import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import operator

BASE_DIR = os.path.abspath(os.getcwd())
FEATURES_PATH = os.path.join(BASE_DIR, "data/features.npy")
CLASSES_PATH = os.path.join(BASE_DIR, "data/classes.npy")

feature_names = ['age', 'shape', 'margin', 'density']
features = np.load(FEATURES_PATH)
classes = np.load(CLASSES_PATH)

clf = KNeighborsClassifier(n_neighbors=10)
knn_score = cross_val_score(clf, features, classes, cv=10).mean()
print("K-Nearest-Neighbors. K = 10. Accuracy: {}".format(knn_score))

# check 5 best results with diff k
results = {}
for i in range (1, 50):
    clf = KNeighborsClassifier(n_neighbors=i)
    knn_score = cross_val_score(clf, features, classes, cv=10).mean()
    results[i] = knn_score
results = sorted(results.items(), key=operator.itemgetter(1))[-5:]

print("Top 5 results")
for k, v in results:
    print("K value = {}, Accuracy: {}".format(k, v))