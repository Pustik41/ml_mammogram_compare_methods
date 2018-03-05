import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.abspath(os.getcwd())
FEATURES_PATH = os.path.join(BASE_DIR, "data/features.npy")
CLASSES_PATH = os.path.join(BASE_DIR, "data/classes.npy")

feature_names = ['age', 'shape', 'margin', 'density']
features = np.load(FEATURES_PATH)
classes = np.load(CLASSES_PATH)

clf = svm.SVC(kernel='linear')
linear_score = cross_val_score(clf, features, classes, cv=10).mean()
print("Support Vector Classification. Kernel Linear. Accuracy: {}".format(linear_score))

clf = svm.SVC(kernel='rbf')
rbf_score = cross_val_score(clf, features, classes, cv=10).mean()
print("Support Vector Classification. Kernel RBF. Accuracy: {}".format(rbf_score))

clf = svm.SVC(kernel='poly')
poly_score = cross_val_score(clf, features, classes, cv=10).mean()
print("Support Vector Classification. Kernel Poly, degree 3. Accuracy: {}".format(poly_score))

