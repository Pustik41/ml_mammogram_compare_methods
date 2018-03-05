import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.abspath(os.getcwd())
FEATURES_PATH = os.path.join(BASE_DIR, "data/features.npy")
CLASSES_PATH = os.path.join(BASE_DIR, "data/classes.npy")

feature_names = ['age', 'shape', 'margin', 'density']
features = np.load(FEATURES_PATH)
classes = np.load(CLASSES_PATH)

clf = LogisticRegression()
lr_score = cross_val_score(clf, features, classes, cv=10).mean()
print("LogisticRegression. Accuracy: {}".format(lr_score))