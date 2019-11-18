import gensim
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn import preprocessing

data_file = "normalised_data.json"
with open(data_file, 'r') as f:
    full_data = json.load(f)
    df = pd.DataFrame(json.loads(full_data['useful']))
    
print("dataset size: {0:d} rows x {1:d} columns".format(df.shape[0], df.shape[1]))
#print(df.head())

text = []
for i in range(df.shape[0]):
    line = gensim.utils.simple_preprocess(df['text'].iloc[i])
    text.append(line)

model = gensim.models.Word2Vec(text, min_count=5, size= 100, workers=3, sg=1)
model.train(text,total_examples=len(text),epochs=5)

#print(model['food'])
print(model.wv.similarity('eat', 'eats'))
print(model.wv.similarity('chinese', 'asian'))
print(model.wv.similarity('burgers', 'fries'))
print(model.wv.similarity('delicious', 'tasty'))
print(model.wv.similarity('drinks', 'alcohol'))
print(model.wv.most_similar(positive='pasta', topn=6))
