import gensim
import json
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, SimpleRNN, LSTM
from tensorflow.keras import Sequential

class LSTMModel:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = []
        self.word2vec_models = []
        self.df_text = []
        self.labels_cap = ["Useful", "Funny", "Cool"]
        self.labels = ["useful", "funny", "cool"]
        self.embedding_dim = 50

    def read_data_file(self):        
        # read normalized datatsets
        with open(self.data_file, 'r') as f:
            full_data = json.load(f)
            # read the 3 preprocessed datasets
            for i in range(3):
                df_temp = pd.DataFrame(json.loads(full_data[self.labels[i]]))
                self.df.append(df_temp)
                print("dataset size: {0:d} rows x {1:d} columns".format( \
                    self.df[i].shape[0], self.df[i].shape[1]))

    def generate_histogram(self):
        # generate histograms for each dataset
        for i in range(3): 
            hist = self.df[i][self.labels[i]].hist(bins=10, rwidth=0.6)
            str_title = "Normalized " + self.labels_cap[i] + ' Label Histogram'
            str_xlabel = "Normalized " + self.labels_cap[i] + ' Count Values'
            str_fig_name = "./histogram_" + self.labels[i]
            plt.title(str_title)
            plt.xlabel(str_xlabel)
            plt.ylabel("Number of Reviews")
            plt.savefig(str_fig_name)
            plt.clf()

    def generate_word2vec(self):
        # create word2vec model for each dataset
        #for i in range(3):
        for i in range(1):
            text = []
            for j in range(self.df[i].shape[0]):
                line = gensim.utils.simple_preprocess(self.df[i]['text'].iloc[j])
                text.append(line)

            self.df_text.append(text)
            model = gensim.models.Word2Vec(text, min_count=1, size=self.embedding_dim, workers=3, sg=1)
            model.train(text,total_examples=len(text),epochs=5)

            self.word2vec_models.append(model)

            print("Testing word2vec model ... ")
            print("Word vector for \"food\": ", model['food'])
            print("Top 5 most similar words as \"pasta\": ", model.wv.most_similar(positive='pasta', topn=5))

    def build_model(self, idx):
        # idx denotes which dataset to build the model on

        common_words = ['i', 'you', 'they', 'has', 'have', 'are', 'is', 'a', 's', 'the', 'there', 'of', 'was', 'were', 'to', 'and', 'it', 'we', 're']

        for i in range(len(self.df_text[idx])):
            j = 0
            while j < len(self.df_text[idx][i]):
                word = self.df_text[idx][i][j]
                if word in common_words:
                    self.df_text[idx][i].pop(j)
                else:
                    j += 1

        max_length = max([len(s) for s in self.df_text[idx]])
        num_reviews = len(self.df_text[idx])
        print(max_length)

        text = np.zeros((num_reviews, max_length, self.embedding_dim))
        for i in range(len(self.df_text[idx])):
            train_idx = 0
            for j in range(len(self.df_text[idx][i])):
                # get jth word of ith review
                word = self.df_text[idx][i][j]
                #print("word: ", word)
                if word in self.word2vec_models[idx].wv.vocab:
                    vec = self.word2vec_models[idx][word]
                    text[i,train_idx,:] = vec
                    train_idx += 1
                #else:
                #    print("review[{0:d}], word[{1:d}]: not in vocab".format(i, j))

        print("Finished creating text... ")
        print("text size: ", text.shape)
        batch_size = 24
        rnn_dim = 100
        # model
        model = Sequential()
        model.add(Dense(batch_size, input_shape=(max_length, self.embedding_dim)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Bidirectional(LSTM(rnn_dim, return_sequences=False)))
        model.add(Dense(1, activation='relu'))
        print(model.summary)

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.003), metrics=['acc'])

        labels = self.df[idx]['useful'].values
        train_text, test_text, train_labels, test_labels = train_test_split(text, labels, test_size=0.1)

        print("train data: ", train_text.shape, train_labels.shape)
        print("test data: ", test_text.shape, test_labels.shape)

        i = num_reviews // batch_size
        i *= batch_size
        train_text = train_text[:i,:,:]
        train_labels = train_labels[:i]

        model.fit(train_text, train_labels, batch_size=batch_size, shuffle=True, epochs=10)

        predicted = model.predict(test_text)

        print("predicted: ", predicted[:30])
        print("real: ", test_labels[:30])

        loss, acc = model.evaluate(test_text, test_labels)
        print("loss: ", loss, " accuracy; ", acc)

    def build_classification_model(self, idx):
        # idx denotes which dataset to build the model on

        common_words = ['i', 'you', 'they', 'has', 'have', 'are', 'is', 'a', 's', 'the', 'there', 'of', 'was', 'were', 'to', 'and', 'it', 'we', 're']

        for i in range(len(self.df_text[idx])):
            j = 0
            while j < len(self.df_text[idx][i]):
                word = self.df_text[idx][i][j]
                if word in common_words:
                    self.df_text[idx][i].pop(j)
                else:
                    j += 1

        max_length = max([len(s) for s in self.df_text[idx]])
        num_reviews = len(self.df_text[idx])
        print(max_length)

        text = np.zeros((num_reviews, max_length, self.embedding_dim))
        for i in range(len(self.df_text[idx])):
            train_idx = 0
            for j in range(len(self.df_text[idx][i])):
                # get jth word of ith review
                word = self.df_text[idx][i][j]
                #print("word: ", word)
                if word in self.word2vec_models[idx].wv.vocab:
                    vec = self.word2vec_models[idx][word]
                    text[i,train_idx,:] = vec
                    train_idx += 1
                #else:
                #    print("review[{0:d}], word[{1:d}]: not in vocab".format(i, j))

        print("Finished creating text... ")
        print("text size: ", text.shape)

        batch_size = 24
        rnn_dim = 100
        # model
        model = Sequential()
        model.add(Dense(batch_size, input_shape=(max_length, self.embedding_dim)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Bidirectional(LSTM(rnn_dim, return_sequences=False)))
        model.add(Dense(3, activation='softmax'))
        print(model.summary)

        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.05), metrics=['acc'])

        class_labels = []
        labels = self.df[idx]['useful'].values
        for l in labels:
            if l < 0.33:
                class_labels.append('not useful')
            elif l >= 0.33 and l < 0.66:
                class_labels.append('quite useful')
            elif l >= 0.66 and l <= 1.0:
                class_labels.append('very useful')
                
        print("class_labels shape: ", len(class_labels))
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(class_labels)
        encoded_labels = encoder.transform(class_labels)
        # convert integers to dummy variables (i.e. one hot encoded)
        one_hot_labels = np_utils.to_categorical(encoded_labels)

        print("one_hot_labels shape: ", one_hot_labels.shape)

        train_text, test_text, train_labels, test_labels = train_test_split(text, one_hot_labels, test_size=0.1)

        print("train data: ", train_text.shape, train_labels.shape)
        print("test data: ", test_text.shape, test_labels.shape)

        i = num_reviews // batch_size
        i *= batch_size
        train_text = train_text[:i,:,:]
        train_labels = train_labels[:i]

        model.fit(train_text, train_labels, batch_size=batch_size, shuffle=True, epochs=10)

        predicted = model.predict(test_text)

        print("predicted: ", predicted[:30])
        print("real: ", test_labels[:30])

        loss, acc = model.evaluate(test_text, test_labels)
        print("loss: ", loss, " accuracy; ", acc)



if __name__ == "__main__":

    model = LSTMModel("normalised_data.json")
    model.read_data_file()
    model.generate_word2vec()
    model.build_model(0)
    



