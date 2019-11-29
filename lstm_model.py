import tensorflow as tf
import json
import tensorflow.keras.backend as K
import numpy as np
import logging

class LSTMModel:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.model = None
        self.embedding_size = None
        self.num_clusters = None
        self.cluster_data_file_name = './cluster_data.json'
        self.train_test_data_file = './train_test_data.json'
        self.read_data()
        self.create_model()
        self.train_model()
        self.calculate_accuracy()

    def create_model(self):
        logging.info('Creating Model')
        model = tf.keras.Sequential()
        input_layer = tf.keras.layers.InputLayer(batch_size=128, input_shape=(self.num_clusters, self.embedding_size))
        model.add(input_layer)
        # rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 200, activation = 'tanh', dropout = 0.1, recurrent_dropout = 0.1, implementation = 1, return_sequences = False))
        # model.add(rnn_layer)
        dense_layer = tf.keras.layers.Dense(5000, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        model.add(dense_layer)
        dense_middle = tf.keras.layers.Dense(3000, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        model.add(dense_middle)
        dense_layer_2 = tf.keras.layers.Dense(1, activation='tanh')
        model.add(dense_layer_2)
        flatten_layer = tf.keras.layers.Flatten()
        model.add(flatten_layer)
        dense_layer_3 = tf.keras.layers.Dense(1, activation='tanh')
        model.add(dense_layer_3)

        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), metrics = ['mse', 'mae'])
        self.model = model
        print(model.summary())

    def read_data(self):
        logging.info('Reading data')
        with open(self.cluster_data_file_name) as fh:
            data = json.load(fh)
            self.embedding_size = data['embedding_size']
            self.num_clusters = data['num_clusters']
            fh.close()
        
        with open(self.train_test_data_file) as fh:
            data = json.load(fh)
            self.train_data = np.array(data['train_data'], dtype=np.float64)
            self.train_labels = np.array(data['train_labels'], dtype=np.float64)
            self.test_data = np.array(data['test_data'], dtype=np.float64)
            self.test_labels = np.array(data['test_labels'], dtype=np.float64)
            fh.close()
    
    def train_model(self):
        logging.info('Training Model')
        self.model.fit(self.train_data, self.train_labels, batch_size = 128, epochs = 5, validation_split = 0.3)

    def calculate_accuracy(self):
        logging.info('Hurrah')
        predictions = self.model.predict(self.test_data)

        correct = 0
        total = len(self.test_labels)

        for index, prediction in enumerate(predictions):
            if abs(prediction - self.test_labels[index]) < 0.15:
                correct = correct + 1
        
        accuracy = correct/total
        print('Final accuracy is ' + str(accuracy))

lstm_model = LSTMModel()