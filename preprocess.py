import json
import numpy as np
import pandas as pd
from sklearn import preprocessing

def write_to_file(useful, funny, cool):
    output_dict = {
        'useful': useful.to_json(orient = 'records'),
        'funny': funny.to_json(orient = 'records'),
        'cool': cool.to_json(orient = 'records')
    }

    with open('normalised_data.json', 'w') as op_handle:
        json.dump(output_dict, op_handle)
        op_handle.close()
        
def preprocess_data(data_file):

    df = pd.read_json(data_file)
    df['useful'] = df['useful'].astype(float)
    df['funny'] = df['funny'].astype(float)
    df['cool'] = df['cool'].astype(float)
    print("dataset size: {0:d} rows x {1:d} columns".format(df.shape[0], df.shape[1]))

    vote_labels = ['useful', 'funny', 'cool']

    useful_min, useful_max = df['useful'].min(), df['useful'].max()
    funny_min, funny_max = df['funny'].min(), df['funny'].max()
    cool_min, cool_max = df['cool'].min(), df['cool'].max()
    print("[useful] min:{0:.2f}, max:{1:.2f}".format(useful_min, useful_max))
    print("[funny] min:{0:.2f}, max:{1:.2f}".format(funny_min, funny_max))
    print("[cool] min:{0:.2f}, max:{1:.2f}".format(cool_min, cool_max))

    # print out distribution of reviews by their vote counts
    for i in range(len(vote_labels)):
        ranges = np.arange(-1, 100, 10)
        counts = df.groupby(pd.cut(df[vote_labels[i]], ranges)).count()[vote_labels[i]]
        print(counts)
        print("out of range counts: ", df.shape[0]-counts.sum())

    # clip vote counts to a range
    df['useful_values'] = df['useful']
    df['funny_values'] = df['funny']
    df['cool_values'] = df['cool']
    df['useful'] = df['useful'].clip(3.0, 27.0)
    df['funny'] = df['funny'].clip(3.0, 20.0)
    df['cool'] = df['cool'].clip(3.0, 24.0)

    # normalize data with min max scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    useful_scaled = min_max_scaler.fit_transform(np.array(df['useful']).reshape(-1,1))
    df['useful'] = pd.DataFrame(useful_scaled)
    funny_scaled = min_max_scaler.fit_transform(np.array(df['funny']).reshape(-1,1))
    df['funny'] = pd.DataFrame(funny_scaled)
    cool_scaled = min_max_scaler.fit_transform(np.array(df['cool']).reshape(-1,1))
    df['cool'] = pd.DataFrame(cool_scaled)

    ranges = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    processed_data = []

    # balance dataset for each of vote category
    for i in range(len(vote_labels)):
        counts = df.groupby(pd.cut(df[vote_labels[i]], ranges)).count()[vote_labels[i]]
        min_count = counts.min()
        new_df = pd.DataFrame()
        for j in range(10):
            lower = j/10
            upper = (j+1)/10 if j < 9 else 1.1
            new_data = df[(df[vote_labels[i]] >= lower) & (df[vote_labels[i]] < upper)]
            new_data = new_data.sample(frac=1)
            new_data = new_data.head(min_count)
            new_df = new_df.append(new_data)

        processed_data.append(new_df)

    # write balanced datasets to a file
    write_to_file(processed_data[0], processed_data[1], processed_data[2])


