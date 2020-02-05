import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("statsfinalhgg.csv")

survival_data = pd.read_csv("survival_data.csv")

df['AGE'] = -999
df['Survival'] = -999 

for x in range(0,len(df)):
    for i in range(0,len(survival_data)):
        if(df['NAME'][x]==survival_data['Brats17ID'][i]):
            df['AGE'][x] = survival_data['Age'][i]
            df['Survival'][x] = survival_data['Survival'][i]
df.to_csv()

df = pd.read_csv("sdf.csv")
for x in range (0,len(df)):
    
    if(df['AGE'][x]==-999):        
        df=df.drop([x], axis=0)

df = df.drop(["NAME", "diagnostics_Image-original_Minimum" ], 1)
df = pd.DataFrame(df)
df = np.array(df)
df = pd.DataFrame(df)
df = df.drop([0], 1)
df = df.rename(columns={113: 'Survival'})

df.loc[df.Survival < 360, 'Survival'] = 0
df.loc[df.Survival >= 360, 'Survival'] = 1
X = df.drop(["Survival"], 1)
Y = df["Survival"]


scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X)

sel = SelectFromModel(RandomForestClassifier(n_estimators = 85))
model = sel.fit(X, Y)
X_new = model.transform(X)
X = pd.DataFrame(X_new)


X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=0.3)

# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# Y_pred = logreg.predict(X_test)
# # acc_log = round(logreg.score(X_train, y_train) * 100, 2)
# # acc_log
# accuracy_score(y_test, Y_pred)

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

random_forest = RandomForestClassifier(n_estimators=70)
model = random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
# random_forest.score(X_train, y_train)
# acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
# acc_random_forest
print("random_forest", accuracy_score(y_test, Y_pred))

import pickle
filename = 'Survival_finalized.sav'
pickle.dump(model, open(filename, 'wb'))