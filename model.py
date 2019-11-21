import gensim
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LSTMModel:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = []
        self.word2vec_models = []
        self.df_text = []
        self.labels_cap = ["Useful", "Funny", "Cool"]
        self.labels = ["useful", "funny", "cool"]

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
            model = gensim.models.Word2Vec(text, min_count=5, size= 100, workers=3, sg=1)
            model.train(text,total_examples=len(text),epochs=5)

            self.word2vec_models.append(model)

            print("Testing word2vec model ... ")
            print("Word vector for \"food\": ", model['food'])
            print("Top 5 most similar words as \"pasta\": ", model.wv.most_similar(positive='pasta', topn=5))

    def build_model(self, i):
        # i denotes which dataset to build the model on
        max_length = max([len(s) for s in self.df_text[i]])
        print(max_length)


if __name__ == "__main__":

    model = LSTMModel("normalised_data.json")
    model.read_data_file()
    model.generate_word2vec()
    model.build_model(0)
    



