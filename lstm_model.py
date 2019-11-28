import tensorflow as tf
import json

class LSTMModel:
    def __init__(self):
        self.model = None
        self.embedding_size = None
        self.num_clusters = None
        self.cluster_data_file_name = './cluster_data.json'
        self.create_model()
        self.read_data()

    def create_model(self):
        model = tf.keras.Sequential()
        first_layer = tf.keras.layers.Dense(50, input_shape=(self.num_clusters, self.embedding_size))
        model.add(first_layer)
        dense_layer = tf.keras.layers.Dense(600, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        model.add(dense_layer)
        # add time distributed layer that outputs num_tags outputs from the sequence for each term in the input
        dense_layer = tf.keras.layers.Dense(1)
        model.add(dense_layer)
        model.add(tf.keras.layers.Activation('softmax'))
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2), metrics = ['mse'])
        self.model = model
        print(model.summary())

    def read_data(self):
        with open(self.cluster_data_file_name) as fh:
            data = json.load(fh)
            self.embedding_size = data['embedding_size']
            self.num_clusters = data['num_clusters']
            fh.close()

lstm_model = LSTMModel()