import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


# loading the data
data1= pd.read_csv("statsfinalhgg.csv")
data2=pd.read_csv("statsfinallgg.csv")

# labelling the data into two different classes
ones = np.ones((207,1), dtype=int)
df = pd.DataFrame(data1)
df['labels'] = ones
zeros = np.zeros((74,1), dtype=int)
df2 = pd.DataFrame(data2)
df2['labels'] = zeros

data = [df, df2]
data = pd.concat(data)
data = data.sample(frac=1).reset_index(drop=True)

# preprocessing the data
data = data.drop(["diagnostics_Image-original_Minimum", "NAME"], 1)
X = data.drop(["labels"], 1)
Y = data["labels"]
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
model = sel.fit(X, Y)
X_new = model.transform(X)
X_new.shape
X = pd.DataFrame(X_new)

# splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Applying different models

# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, y_train) * 100, 2)
# acc_log

# svc = SVC()
# svc.fit(X_train, y_train)
# Y_pred = svc.predict(X_test)
# # acc_svc = round(svc.score(X_train, y_train) * 100, 2)
# # acc_svc
# accuracy_score(y_test, Y_pred)

# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, y_train)
# Y_pred = knn.predict(X_test)
# # acc_knn = round(knn.score(X_train, y_train) * 100, 2)
# # acc_knn
# accuracy_score(y_test, Y_pred)

# gaussian = GaussianNB()
# gaussian.fit(X_train, y_train)
# Y_pred = gaussian.predict(X_test)
# # acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
# # acc_gaussian
# accuracy_score(y_test, Y_pred)

# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, y_train)
# Y_pred = decision_tree.predict(X_test)
# # acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
# # acc_decision_tree
# accuracy_score(y_test, Y_pred)

random_forest = RandomForestClassifier(n_estimators=75)
model = random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
# random_forest.score(X_train, y_train)
# acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
# acc_random_forest
accuracy_score(y_test, Y_pred)

# Save the model
filename = 'finalized_model.sav'

import pickle
pickle.dump(model, open(filename, 'wb'))