import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

BASE_DIR = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(BASE_DIR, "data/mammographic_masses.data.txt")
FEATURES_PATH = os.path.join(BASE_DIR, "data/features.npy")
CLASSES_PATH = os.path.join(BASE_DIR, "data/classes.npy")

header = ['BI-RADS', 'age', 'shape', 'margin', 'density', "severity"]
df = pd.read_csv(DATA_PATH, names=header, na_values='?', sep=',')
df.dropna(axis=0, inplace=True)



features = df[['age', 'shape', 'margin', 'density']].values
classes = df["severity"].values


scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(features)
print(FEATURES_PATH)
print(CLASSES_PATH)
print(np.save(FEATURES_PATH, scaled_features))
print(np.save(CLASSES_PATH, classes))
