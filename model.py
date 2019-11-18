import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


data_file = "dataset.json"
df = pd.read_json(data_file)
print("dataset size: {0:d} rows x {1:d} columns", df.shape[0], df.shape[1])

df['useful'] = df['useful'].astype(float)
df['funny'] = df['funny'].astype(float)
df['cool'] = df['cool'].astype(float)

vote_labels = ['useful', 'funny', 'cool']

useful_min, useful_max = df['useful'].min(), df['useful'].max()
funny_min, funny_max = df['funny'].min(), df['funny'].max()
cool_min, cool_max = df['cool'].min(), df['cool'].max()
print("[useful] min:{0:.2f}, max:{1:.2f}".format(useful_min, useful_max))
print("[funny] min:{0:.2f}, max:{1:.2f}".format(funny_min, funny_max))
print("[cool] min:{0:.2f}, max:{1:.2f}".format(cool_min, cool_max))

# plot histograms for votes
#hist = df[['useful','funny','cool']].hist(bins=20)
#hist.subtitle("Number of Reviews")
#plt.show()

for i in range(len(vote_labels)):
    ranges = np.arange(-1, 100, 2)
    counts = df.groupby(pd.cut(df[vote_labels[i]], ranges)).count()[vote_labels[i]]
    print(counts)
    print("{0:s} counts out of range: {1:d}".format(vote_labels[i], df.shape[0]-counts.sum()))

# clip vote values to a range
df['useful'] = df['useful'].clip(3.0, 27.0)
df['funny'] = df['funny'].clip(3.0, 20.0)
df['cool'] = df['cool'].clip(3.0, 24.0)

min_max_scaler = preprocessing.MinMaxScaler()
useful_scaled = min_max_scaler.fit_transform(np.array(df['useful']).reshape(-1,1))
df['useful'] = pd.DataFrame(useful_scaled)
funny_scaled = min_max_scaler.fit_transform(np.array(df['funny']).reshape(-1,1))
df['funny'] = pd.DataFrame(funny_scaled)
cool_scaled = min_max_scaler.fit_transform(np.array(df['cool']).reshape(-1,1))
df['cool'] = pd.DataFrame(cool_scaled)

for i in range(len(vote_labels)):
    ranges = np.arange(-0.1, 1.1, 0.1)
    counts = df.groupby(pd.cut(df[vote_labels[i]], ranges)).count()[vote_labels[i]]
    print(counts)
    percent_counts = (df.groupby(pd.cut(df[vote_labels[i]], ranges)).count()[vote_labels[i]] / df.shape[0]) * 100
    print(percent_counts)
    print("total reviews: ", counts.sum())


#new_data = df[(df['useful'] > 0.6) & (df['funny'] > 0.6) & (df['cool'] > 0.6) & \
#    (df['useful'] <= 0.8) & (df['funny'] <= 0.8) & (df['cool'] <= 0.8)]
#print("size: ", new_data.shape)

"""
fig, ax = plt.subplots()

u_heights, u_bins = np.histogram(df['useful'], bins=[0, 0.333, 0.666, 1.0])
f_heights, f_bins = np.histogram(df['funny'], bins=[0, 0.333, 0.666, 1.0])
c_heights, c_bins = np.histogram(df['cool'], bins=[0, 0.333, 0.666, 1.0])

width = (u_bins[1] - f_bins[0])/4

ax.bar(u_bins[:-1], u_heights, width=width, facecolor='cornflowerblue')
ax.bar(f_bins[:-1]+width, f_heights, width=width, facecolor='seagreen')
ax.bar(c_bins[:-1]+2*width, c_heights, width=width, facecolor='purple')
plt.show()
"""