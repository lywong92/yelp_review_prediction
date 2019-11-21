import gensim
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def build_model(data_file):

    labels_cap = ["Useful", "Funny", "Cool"]
    labels = ["useful", "funny", "cool"]
    df = []
    
    # read normalized datatsets
    with open(data_file, 'r') as f:
        full_data = json.load(f)
        # read the 3 preprocessed datasets
        for i in range(3):
            df_temp = pd.DataFrame(json.loads(full_data[labels[i]]))
            df.append(df_temp)
            print("dataset size: {0:d} rows x {1:d} columns".format( \
                df[i].shape[0], df[i].shape[1]))

    # generate histograms for each dataset
    for i in range(3): 
        hist = df[i][labels[i]].hist(bins=10, rwidth=0.6)
        str_title = "Normalized " + labels_cap[i] + ' Label Histogram'
        str_xlabel = "Normalized " + labels_cap[i] + ' Count Values'
        str_fig_name = "./histogram_" + labels[i]
        plt.title(str_title)
        plt.xlabel(str_xlabel)
        plt.ylabel("Number of Reviews")
        plt.savefig(str_fig_name)
        plt.clf()

    # create word2vec model for each dataset
    word2vec_models = []
    for i in range(3):
        text = []
        for j in range(df[i].shape[0]):
            line = gensim.utils.simple_preprocess(df[i]['text'].iloc[j])
            text.append(line)

        model = gensim.models.Word2Vec(text, min_count=5, size= 100, workers=3, sg=1)
        model.train(text,total_examples=len(text),epochs=5)

        word2vec_models.append(model)

        print("Testing word2vec model ... ")
        print("Word vector for \"food\": ", model['food'])
        print("Top 5 most similar words as \"pasta\": ", model.wv.most_similar(positive='pasta', topn=5))
