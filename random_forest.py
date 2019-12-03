import logging
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForest:
    def __init__(self):
        self.cluster_data_file_name = './cluster_data.json'
        self.train_test_data_file = './train_test_data.json'
        logging.basicConfig(level=logging.INFO)
        self.read_data()
        self.train_model()
        self.calculate_accuracy()

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
            self.train_data = np.average(self.train_data, axis=1)
            self.train_labels = np.array(data['train_alternate_labels'], dtype=np.float64)
            self.test_data = np.array(data['test_data'], dtype=np.float64)
            self.test_data = np.average(self.test_data, axis=1)
            self.test_labels = np.array(data['test_alternate_labels'], dtype=np.float64)
            fh.close()

    def train_model(self):
        self.regressor = RandomForestRegressor(n_estimators=480, random_state=0)
        self.regressor.fit(self.train_data, self.train_labels)
    
    def calculate_accuracy(self):
        predictions = self.regressor.predict(self.test_data)
        correct = 0
        total = len(self.test_labels)

        for index, prediction in enumerate(predictions):
            # if (prediction > 15 and self.test_labels[index] > 15):
            if abs(prediction - self.test_labels[index]) < 10:
                correct = correct + 1
            # elif(prediction <= 15 and self.test_labels[index] <= 15):
            #     correct = correct + 1
        
        accuracy = correct/total
        print('Final accuracy is ' + str(accuracy))

model = RandomForest()