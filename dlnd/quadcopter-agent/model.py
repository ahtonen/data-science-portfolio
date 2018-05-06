import numpy as np
import tensorflow as tf
import random as rn
import os
#
# Set random seeds according this guide:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
#
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
from keras import layers, models, optimizers, initializers
import keras.regularizers


class DenseModel:
    """Base class for both models."""
    def __init__(self, name, lr=0.001, beta_1=0.9, decay=0.0):
        self.name = name
        self.lr = lr
        self.beta_1 = beta_1
        self.decay = decay
        self.model = models.Sequential()

    def build_dense_layer(self, input, n, init_offset, activation=None, use_batch_norm=False,
        use_dropout=False, dropout_rate=0.2):
        """Helper method for building fully connected layers with bells and whistles."""
        # Initialize weights
        init = initializers.RandomUniform(-init_offset, init_offset)

        # Bias not needed when using batch norm
        if use_batch_norm:
            x = layers.Dense(units=n, kernel_initializer=init, use_bias=False)(input)
            output = layers.BatchNormalization()(x)
        else:
            output = layers.Dense(units=n, kernel_initializer=init, bias_initializer=init)(input)

        # Activation
        if activation is not None:
            output = layers.Activation(activation)(output)

        # Dropout
        if use_dropout:
            output = layers.Dropout(dropout_rate)(output)

        return output

    def save(self):
        """Save model weights as HDF5."""
        self.model.save_weights(self.name+".h5")

    def load(self):
        """Load weights for this model."""
        self.model.load_weights(self.name+".h5")


class Actor(DenseModel):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, name='Actor'):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        super().__init__(name)

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # batch normalization not good idea here...

        # Add hidden layers
        net = self.build_dense_layer(states, 256, 1./np.sqrt(self.state_size), 'relu')
        net = self.build_dense_layer(net, 256, 1./np.sqrt(256), 'relu')

        # Add final output layer with sigmoid activation
        raw_actions = self.build_dense_layer(net, self.action_size, 3e-3, 'tanh')
        #raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
        #    kernel_regularizer=keras.regularizers.l2(0.01), name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range/2) + self.action_range/2,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.lr, beta_1=self.beta_1, decay=self.decay)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic(DenseModel):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, name='Critic'):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super().__init__(name, lr=0.0001)
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = self.build_dense_layer(states, 128, 1./np.sqrt(self.state_size), 'relu',
            use_batch_norm=False)
        net_states = self.build_dense_layer(net_states, 256, 1./np.sqrt(128), 'relu',
            use_batch_norm=False)

        # Add hidden layer(s) for action pathway
        net_actions = self.build_dense_layer(actions, 128, 1./np.sqrt(self.action_size), 'relu',
            use_batch_norm=False)
        net_actions = self.build_dense_layer(actions, 256, 1./np.sqrt(128), 'relu',
            use_batch_norm=False)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        #net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add 3rd hidden layer for combined pathway
        #net = self.build_dense_layer(net, 300, 1./np.sqrt(300), 'relu', use_batch_norm=True)

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr, beta_1=self.beta_1, decay=self.decay)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
