import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

data_file = "normalised_data.json"
with open(data_file, 'r') as f:
    full_data = json.load(f)
    print(full_data['useful'][0])
    df = pd.DataFrame(full_data['useful'])
    
print("dataset size: {0:d} rows x {1:d} columns".format(df.shape[0], df.shape[1]))