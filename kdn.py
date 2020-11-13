# Copyright 2013 James McCauley
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Installs forwarding rules based on topologically significant IP addresses.

We also issue those addresses by DHCP.  A host must use the assigned IP!
Actually, the last byte can be almost anything.  But addresses are of the
form 10.switchID.portNumber.x.

This is an example of a pretty proactive forwarding application.

The forwarding code is based on l2_multi.

Depends on openflow.discovery
Works with openflow.spanning_tree (sort of)
"""

from pox.core import core
import pox.openflow.libopenflow_01 as of
import pox.lib.packet as pkt

from pox.lib.addresses import IPAddr,EthAddr,parse_cidr
from pox.lib.addresses import IP_BROADCAST, IP_ANY
from pox.lib.revent import *
from pox.lib.util import dpid_to_str
from pox.proto.dhcpd import DHCPLease, DHCPD
from collections import defaultdict
from pox.openflow.discovery import Discovery
import time

from pox.lib.recoco import Timer


import numpy as np
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

log = core.getLogger("f.t_p")


# Adjacency map.  [sw1][sw2] -> port from sw1 to sw2
adjacency = defaultdict(lambda:defaultdict(lambda:None))

# Switches we know of.  [dpid] -> Switch and [id] -> Switch
switches_by_dpid = {}
switches_by_id = {}

# [sw1][sw2] -> (distance, intermediate)
path_map = defaultdict(lambda:defaultdict(lambda:(None,None)))


# Dim
s_dim = 8
a_dim = 10

# Statistic
bytes_by_dpid = {}
count_flag = 0
count_sf = 0


# Link weights. [sw1.dpid][sw2.dpid] -> link's weight
lw = defaultdict(lambda:defaultdict(lambda:None))

# rl_POX map. [dpid] -> index. [dpid][dpid] -> index
s_map = defaultdict(lambda:None) # connectionup
a_map = defaultdict(lambda:defaultdict(lambda:None)) #lldp
s_flag = 0
a_flag = 0

rl_s = np.ones(s_dim, dtype=np.float32) # mapped by bytes_by_dpid
rl_a = np.ones(a_dim, dtype=np.float32) # map to lw
rl_s0 = np.ones(s_dim, dtype=np.float32)



#DRL initial
BUFFER_SIZE = 160
BATCH_SIZE = 32
GAMMA = 0.99
EPISODE_COUNT = 100
MAX_STEPS = 64
EXPLORE = EPISODE_COUNT * MAX_STEPS * 0.8
loop = 0
epsilon = 1



ou = OU(a_dim, 0.0, 0.2, 0.4)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
actor = ActorNetwork(sess, s_dim, a_dim)
critic = CriticNetwork(sess, s_dim, a_dim)
buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

graph = tf.get_default_graph()


def reward (states):
  diff = np.std(states)
  return diff
def dpid_to_mac (dpid):
  return EthAddr("%012x" % (dpid & 0xffFFffFFffFF,))


def _calc_paths ():
  """
  Essentially Floyd-Warshall algorithm
  """

  def dump ():
    for i in sws:
      for j in sws:
        a = path_map[i][j][0]
        #a = adjacency[i][j]
        if a is None: a = "*"
        print a,
      print

  sws = switches_by_dpid.values()
  path_map.clear()
  for k in sws:
    for j,port in adjacency[k].iteritems():
      if port is None: continue
      if lw[k.dpid][j.dpid] is None:
        path_map[k][j] = (1,None)
      else:
        path_map[k][j] = (lw[k.dpid][j.dpid], None)  # links' weight
    path_map[k][k] = (0,None) # distance, intermediate

  #dump()

  for k in sws:
    for i in sws:
      for j in sws:
        if path_map[i][k][0] is not None:
          if path_map[k][j][0] is not None:
            # i -> k -> j exists
            ikj_dist = path_map[i][k][0]+path_map[k][j][0]
            if path_map[i][j][0] is None or ikj_dist < path_map[i][j][0]:
              # i -> k -> j is better than existing
              path_map[i][j] = (ikj_dist, k)

  #print "--------------------"
  #dump()


def _get_raw_path (src, dst):
  """
  Get a raw path (just a list of nodes to traverse)
  """
  if len(path_map) == 0: _calc_paths()
  if src is dst:
    # We're here!
    return []
  if path_map[src][dst][0] is None:
    return None
  intermediate = path_map[src][dst][1]
  if intermediate is None:
    # Directly connected
    return []
  return _get_raw_path(src, intermediate) + [intermediate] + \
         _get_raw_path(intermediate, dst)


def _get_path (src, dst):
  """
  Gets a cooked path -- a list of (node,out_port)
  """
  # Start with a raw path...
  if src == dst:
    path = [src]
  else:
    path = _get_raw_path(src, dst)
    if path is None: return None
    path = [src] + path + [dst]

  # Now add the ports
  r = []
  for s1,s2 in zip(path[:-1],path[1:]):
    out_port = adjacency[s1][s2]
    r.append((s1,out_port))
    in_port = adjacency[s2][s1]

  return r


def ipinfo (ip):
  parts = [int(x) for x in str(ip).split('.')]
  ID = parts[1]
  port = parts[2]
  num = parts[3]
  return switches_by_id.get(ID),port,num


class TopoSwitch (DHCPD):
  _eventMixin_events = set([DHCPLease])
  _next_id = 100

  def __repr__ (self):
    try:
      return "[%s/%s]" % (dpid_to_str(self.connection.dpid),self._id)
    except:
      return "[Unknown]"


  def __init__ (self):
    self.log = log.getChild("Unknown")

    self.connection = None
    self.ports = None
    self.dpid = None
    self._listeners = None
    self._connected_at = None
    self._id = None
    self.subnet = None
    self.network = None
    self._install_flow = False
    self.mac = None

    self.ip_to_mac = {}

    # Listen to our own event... :)
    self.addListenerByName("DHCPLease", self._on_lease)

    core.ARPHelper.addListeners(self)


  def _handle_ARPRequest (self, event):
    if ipinfo(event.ip)[0] is not self: return
    event.reply = self.mac


  def send_table (self):
    if self.connection is None:
      self.log.debug("Can't send table: disconnected")
      return

    clear = of.ofp_flow_mod(command=of.OFPFC_DELETE)
    self.connection.send(clear)
    self.connection.send(of.ofp_barrier_request())

    # From DHCPD
    msg = of.ofp_flow_mod()
    msg.match = of.ofp_match()
    msg.match.dl_type = pkt.ethernet.IP_TYPE
    msg.match.nw_proto = pkt.ipv4.UDP_PROTOCOL
    #msg.match.nw_dst = IP_BROADCAST
    msg.match.tp_src = pkt.dhcp.CLIENT_PORT
    msg.match.tp_dst = pkt.dhcp.SERVER_PORT
    msg.actions.append(of.ofp_action_output(port = of.OFPP_CONTROLLER))
    #msg.actions.append(of.ofp_action_output(port = of.OFPP_FLOOD))
    self.connection.send(msg)

    core.openflow_discovery.install_flow(self.connection)

    src = self
    for dst in switches_by_dpid.itervalues():
      if dst is src: continue
      p = _get_path(src, dst)
      if p is None: continue
      for port in self.ports:
        pn = port.port_no
        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match()
        msg.match.dl_type = pkt.ethernet.IP_TYPE
        msg.match.in_port = pn
        #msg.match.nw_dst = "%s/%s" % (dst.network, dst.subnet)
        msg.match.nw_dst = "%s/%s" % (dst.network, "255.255.0.0")

        msg.actions.append(of.ofp_action_output(port=p[0][1]))
        self.connection.send(msg)

    """
    # Can just do this instead of MAC learning if you run arp_responder...
    for port in self.ports:
      p = port.port_no
      if p < 0 or p >= of.OFPP_MAX: continue
      msg = of.ofp_flow_mod()
      msg.match = of.ofp_match()
      msg.match.dl_type = pkt.ethernet.IP_TYPE
      msg.match.nw_dst = "10.%s.%s.0/255.255.255.0" % (self._id,p)
      msg.actions.append(of.ofp_action_output(port=p))
      self.connection.send(msg)
    """

    for ip,mac in self.ip_to_mac.iteritems():
      self._send_rewrite_rule(ip, mac)

    flood_ports = []
    for port in self.ports:
      p = port.port_no
      if p < 0 or p >= of.OFPP_MAX: continue

      if core.openflow_discovery.is_edge_port(self.dpid, p):
        flood_ports.append(p)

      msg = of.ofp_flow_mod()
      msg.priority -= 1
      msg.match = of.ofp_match()
      msg.match.dl_type = pkt.ethernet.IP_TYPE
      msg.match.nw_dst = "10.%s.%s.0/255.255.255.0" % (self._id,p)
      msg.actions.append(of.ofp_action_output(port=of.OFPP_CONTROLLER))
      self.connection.send(msg)

    msg = of.ofp_flow_mod()
    msg.priority -= 1
    msg.match = of.ofp_match()
    msg.match.dl_type = pkt.ethernet.IP_TYPE
    msg.match.nw_dst = "255.255.255.255"
    for p in flood_ports:
      msg.actions.append(of.ofp_action_output(port=p))
    self.connection.send(msg)


  def _send_rewrite_rule (self, ip, mac):
    p = ipinfo(ip)[1]
    for port in self.ports:
      pn = port.port_no
      msg = of.ofp_flow_mod()
      msg.match = of.ofp_match()
      msg.match.dl_type = pkt.ethernet.IP_TYPE
      msg.match.in_port = pn
      msg.match.nw_dst = ip
      msg.actions.append(of.ofp_action_dl_addr.set_src(self.mac))
      msg.actions.append(of.ofp_action_dl_addr.set_dst(mac))
      msg.actions.append(of.ofp_action_output(port=p))
      self.connection.send(msg)


  def disconnect (self):
    if self.connection is not None:
      log.debug("Disconnect %s" % (self.connection,))
      self.connection.removeListeners(self._listeners)
      self.connection = None
      self._listeners = None


  def connect (self, connection):
    if connection is None:
      self.log.warn("Can't connect to nothing")
      return
    if self.dpid is None:
      self.dpid = connection.dpid
    assert self.dpid == connection.dpid
    if self.ports is None:
      self.ports = connection.features.ports
    self.disconnect()
    self.connection = connection
    self._listeners = self.listenTo(connection)
    self._connected_at = time.time()

    label = dpid_to_str(connection.dpid)
    self.log = log.getChild(label)
    self.log.debug("Connect %s" % (connection,))

    if self._id is None:
      if self.dpid not in switches_by_id and self.dpid <= 254:
        self._id = self.dpid
      else:
        self._id = TopoSwitch._next_id
        TopoSwitch._next_id += 1
      switches_by_id[self._id] = self

    self.network = IPAddr("10.%s.0.0" % (self._id,))
    self.mac = dpid_to_mac(self.dpid)

    # Disable flooding
    con = connection
    log.debug("Disabling flooding for %i ports", len(con.ports))
    for p in con.ports.itervalues():
      if p.port_no >= of.OFPP_MAX: continue
      pm = of.ofp_port_mod(port_no=p.port_no,
                          hw_addr=p.hw_addr,
                          config = of.OFPPC_NO_FLOOD,
                          mask = of.OFPPC_NO_FLOOD)
      con.send(pm)
    con.send(of.ofp_barrier_request())
    con.send(of.ofp_features_request())

    # Some of this is copied from DHCPD's __init__().
    self.send_table()

    self.ip_addr = IPAddr("10.%s.0.1" % (self._id,))
    #self.router_addr = self.ip_addr
    self.router_addr = None
    self.dns_addr = None #fix_addr(dns_address, self.router_addr)

    self.subnet = IPAddr("255.0.0.0")
    self.pools = {}
    for p in connection.ports:
      if p < 0 or p >= of.OFPP_MAX: continue
      self.pools[p] = [IPAddr("10.%s.%s.%s" % (self._id,p,n))
                       for n in range(1,255)]

    self.lease_time = 60 * 60 # An hour
    #TODO: Actually make them expire :)

    self.offers = {} # Eth -> IP we offered
    self.leases = {} # Eth -> IP we leased

    global s_flag
    if s_flag < s_dim and self.dpid not in s_map.keys():
      s_map[self.dpid] = s_flag
      s_flag += 1



  def _get_pool (self, event):
    pool = self.pools.get(event.port)
    if pool is None:
      log.warn("No IP pool for port %s", event.port)
    return pool


  def _handle_ConnectionDown (self, event):
    self.disconnect()


  def _mac_learn (self, mac, ip):
    if ip.inNetwork(self.network,"255.255.0.0"):
      if self.ip_to_mac.get(ip) != mac:
        self.ip_to_mac[ip] = mac
        self._send_rewrite_rule(ip, mac)
        return True
    return False


  def _on_lease (self, event):
    if self._mac_learn(event.host_mac, event.ip):
        self.log.debug("Learn %s -> %s by DHCP Lease",event.ip,event.host_mac)


  def _handle_PacketIn (self, event):
    packet = event.parsed
    arpp = packet.find('arp')
    if arpp is not None:
      if event.port != ipinfo(arpp.protosrc)[1]:
        self.log.warn("%s has incorrect IP %s", arpp.hwsrc, arpp.protosrc)
        return

      if self._mac_learn(packet.src, arpp.protosrc):
        self.log.debug("Learn %s -> %s by ARP",arpp.protosrc,packet.src)
    else:
      ipp = packet.find('ipv4')
      if ipp is not None:
        # Should be destined for this switch with unknown MAC
        # Send an ARP
        sw,p,_= ipinfo(ipp.dstip)
        if sw is self:
          log.debug("Need MAC for %s", ipp.dstip)
          core.ARPHelper.send_arp_request(event.connection,ipp.dstip,port=p)

    return super(TopoSwitch,self)._handle_PacketIn(event)


class topo_addressing (object):
  def __init__ (self):
    self._timer = None
    core.listen_to_dependencies(self, listen_args={'openflow':{'priority':0}})
    #core.openflow.addListenerByName("FlowStatsReceived", self._Handle_flow_Stats)
    core.openflow.addListenerByName("AggregateFlowStatsReceived", self._Handle_aggregateflow_stats)

  def _handle_ARPHelper_ARPRequest (self, event):
    pass # Just here to make sure we load it

  def _handle_openflow_discovery_LinkEvent (self, event):
    def flip (link):
      return Discovery.Link(link[2],link[3], link[0],link[1])

    l = event.link
    sw1 = switches_by_dpid[l.dpid1]
    sw2 = switches_by_dpid[l.dpid2]

    global a_flag
    if a_flag < a_dim:
      if l.dpid1 not in a_map.keys():
        a_map[l.dpid1][l.dpid2] = a_flag
	a_map[l.dpid2][l.dpid1] = a_flag
        a_flag += 1
      elif l.dpid2 not in a_map[l.dpid1]:
        a_map[l.dpid1][l.dpid2] = a_flag
	a_map[l.dpid2][l.dpid1] = a_flag
        a_flag += 1
      

    # Invalidate all flows and path info.
    # For link adds, this makes sure that if a new link leads to an
    # improved path, we use it.
    # For link removals, this makes sure that we don't use a
    # path that may have been broken.
    #NOTE: This could be radically improved! (e.g., not *ALL* paths break)
    clear = of.ofp_flow_mod(command=of.OFPFC_DELETE)
    for sw in switches_by_dpid.itervalues():
      if sw.connection is None: continue
      sw.connection.send(clear)
    path_map.clear()

    if event.removed:
      # This link no longer okay
      if sw2 in adjacency[sw1]: del adjacency[sw1][sw2]
      if sw1 in adjacency[sw2]: del adjacency[sw2][sw1]

      # But maybe there's another way to connect these...
      for ll in core.openflow_discovery.adjacency:
        if ll.dpid1 == l.dpid1 and ll.dpid2 == l.dpid2:
          if flip(ll) in core.openflow_discovery.adjacency:
            # Yup, link goes both ways
            adjacency[sw1][sw2] = ll.port1
            adjacency[sw2][sw1] = ll.port2
            # Fixed -- new link chosen to connect these
            break
    else:
      # If we already consider these nodes connected, we can
      # ignore this link up.
      # Otherwise, we might be interested...
      if adjacency[sw1][sw2] is None:
        # These previously weren't connected.  If the link
        # exists in both directions, we consider them connected now.
        if flip(l) in core.openflow_discovery.adjacency:
          # Yup, link goes both ways -- connected!
          adjacency[sw1][sw2] = l.port1
          adjacency[sw2][sw1] = l.port2

    for sw in switches_by_dpid.itervalues():
      sw.send_table()


  def _handle_openflow_ConnectionUp (self, event):
    self._set_timer()
    sw = switches_by_dpid.get(event.dpid)

    if sw is None:
      # New switch

      sw = TopoSwitch()
      switches_by_dpid[event.dpid] = sw
      sw.connect(event.connection)
    else:
      sw.connect(event.connection)

  def _set_timer (self):
    if self._timer: self._timer.cancel()
    interval = 10
    self._timer = Timer(interval, self._timer_handler, recurring=True)

  def _timer_handler (self):
    stats_request = of.ofp_aggregate_stats_request()
    stats_request.match = of.ofp_match()
    #stats_request.match.dl_type = pkt.ethernet.ARP_TYPE #IP_TYPE
    for sw in switches_by_dpid.itervalues():
      if sw.connection is None: continue
      #print("send status request")
      #sw.connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
      sw.connection.send(of.ofp_stats_request(body=stats_request))

  """
  def _Handle_flow_Stats(self, event):

    sum = 0
    for f in event.stats:
      sum += f.byte_count
    bytes_by_dpid[event.dpid] = sum
    print("Switch %s has %s bytes", event.dpid, sum)

  """

  def _Handle_aggregateflow_stats (self, event):
    global count_sf
    global rl_s0
    global count_flag
    global loop
    global epsilon
    global rl_a
    global MAX_STEPS


    bytes_by_dpid[event.dpid] = event.stats.byte_count - count_sf
    count_sf = event.stats.byte_count
    print("Switch %s has %s bytes", event.dpid, event.stats.byte_count)
    count_flag += 1

    if count_flag == s_dim:
      count_flag = 0
      count_sf = 0

      print("\n *******Update the replaybuffer at loop %d ************" %(loop))
      for sw_dpid in switches_by_dpid.keys():
        if s_map[sw_dpid] is not None:
          rl_s[s_map[sw_dpid]] = bytes_by_dpid[sw_dpid]

      rl_r = reward(rl_s)

      if loop != 0:
	if loop % MAX_STEPS != 0:
          buff.add(rl_s0, rl_a, rl_r, rl_s, 0)
	else:
	  buff.add(rl_s0, rl_a, rl_r, rl_s, 1)

      global graph
      with graph.as_default():
      ################      train       ################
        if loop > 3:
          scale = lambda x: x
          batch = buff.getBatch(BATCH_SIZE)
          states = scale(np.asarray([e[0] for e in batch]))
          actions = scale(np.asarray([e[1] for e in batch]))
          rewards = scale(np.asarray([e[2] for e in batch]))
          new_states = scale(np.asarray([e[3] for e in batch]))
          dones = np.asarray([e[4] for e in batch])

          y_t = np.zeros([len(batch), a_dim])
          target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

          for k in range(len(batch)):
            if dones[k]:
              y_t[k] = rewards[k]
            else:
              y_t[k] = rewards[k] + GAMMA * target_q_values[k]

          if len(batch) >= BATCH_SIZE:
            loss = critic.model.train_on_batch([states, actions], y_t)
            a_for_grad = actor.model.predict(states)
            grads = critic.gradients(states, a_for_grad)
            actor.train(states, grads)
            actor.target_train()
            critic.target_train()
        ################      train       ################

        ################      New Action       ################
        rl_a_original = actor.model.predict(rl_s.reshape(1, rl_s.shape[0]))

        epsilon -= 1.0 / EXPLORE
        noise_t = np.zeros([1, a_dim])
        if epsilon > 0 and (loop % 1000) // 100 != 9:
          noise_t[0] = epsilon * ou.evolve()

        a = rl_a_original[0]
        n = noise_t[0]
        rl_a = np.where((a + n > 0) & (a + n < 1), a + n, a - n).clip(min=0, max=1)
	
	
        for sw1_dpid in switches_by_dpid.keys():
          for sw2_dpid in switches_by_dpid.keys():
	    #print ("a_map[sw1_dpid][sw2_dpid] = " + str(a_map[sw1_dpid][sw2_dpid]))
            if sw1_dpid != sw2_dpid and a_map[sw1_dpid][sw2_dpid] is not None:
	      lw[sw1_dpid][sw2_dpid] = rl_a[a_map[sw1_dpid][sw2_dpid]]
            else:
              lw[sw1_dpid][sw2_dpid] = 1



      for sw in switches_by_dpid.itervalues():
        sw.send_table()
      ################      New Action       ################

      rl_s0 = np.copy(rl_s)

      loop += 1








def launch (debug = False):
  core.registerNew(topo_addressing)
  from proto.arp_helper import launch
  launch(eat_packets=False)
  if not debug:
    core.getLogger("proto.arp_helper").setLevel(99)

  # initial DRL

