"""
CriticNetwork.py
"""
__author__ = "giorgio@ac.upc.edu"
__credits__ = "https://github.com/yanpanlau"

from keras.initializers import normal, glorot_normal
from keras.activations import relu
from keras.layers import Dense, Input, merge, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
import keras

from helper import selu


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size):
        self.HIDDEN1_UNITS = 91
        self.HIDDEN2_UNITS = 42

        self.sess = sess
        self.BATCH_SIZE = 32
        self.TAU = 0.001
        self.LEARNING_RATE = 0.0001
        self.action_size = action_size

        self.h_acti = relu
        self.h_acti = selu

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        print("Now we build the Critic model")
        S = Input(shape=[state_size], name='c_S')
        A = Input(shape=[action_dim], name='c_A')
        w1 = Dense(self.HIDDEN1_UNITS, activation=self.h_acti, kernel_initializer='glorot_normal', name='c_w1')(S)
        a1 = Dense(self.HIDDEN2_UNITS, activation='linear', kernel_initializer='glorot_normal', name='c_a1')(A)
        h1 = Dense(self.HIDDEN2_UNITS, activation='linear', kernel_initializer='glorot_normal', name='c_h1')(w1)
        added = keras.layers.Add()([h1, a1])
        # h2 = merge([h1, a1], mode='sum', name='c_h2')
        h3 = Dense(self.HIDDEN2_UNITS, activation=self.h_acti, kernel_initializer='glorot_normal', name='c_h3')(added)
        V = Dense(action_dim, activation='linear', kernel_initializer='glorot_normal', name='c_V')(h3)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
