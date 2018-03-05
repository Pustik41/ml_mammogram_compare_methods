import os
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

BASE_DIR = os.path.abspath(os.getcwd())
FEATURES_PATH = os.path.join(BASE_DIR, "data/features.npy")
CLASSES_PATH = os.path.join(BASE_DIR, "data/classes.npy")

feature_names = ['age', 'shape', 'margin', 'density']
features = np.load(FEATURES_PATH)
classes = np.load(CLASSES_PATH)

X, x_test, Y, y_test = train_test_split(features, classes, test_size=0.25, random_state=1)

clf = tree.DecisionTreeClassifier(random_state=1)
clf = clf.fit(X, Y)

accuracy = clf.score(x_test, y_test)
print("Decision Tree Classifier. Accuracy: {}".format(accuracy))

# use cross_val_score
clf = tree.DecisionTreeClassifier(random_state=1)
cvs_score = cross_val_score(clf, features, classes, cv=10).mean()
print("Decision Tree Classifier. Used cross_val_score. Accuracy: {}".format(cvs_score))


clf = RandomForestClassifier(n_estimators=10, random_state=1)
cvs_score = cross_val_score(clf, features, classes, cv=10).mean()
print("Random Forest Classifier. Accuracy: {}".format(cvs_score))


