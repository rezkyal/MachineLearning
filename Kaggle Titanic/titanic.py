import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, Normalizer, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors, svm, decomposition
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv('../Dataset/titanic/train.csv')

#1. Preprocessing
data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

# Handling categorical data
data = pd.get_dummies(data,columns=['Sex'])
data = pd.get_dummies(data,columns=['Embarked'])
#check if missval exist
#print(data.isnull().sum())

#replace missval in attribute age with mean
imp = Imputer(strategy='mean')
data['Age']=imp.fit_transform(data[['Age']])
index=data.columns

#check if missval still exist
#print(data.isnull().sum())

#normalization
data=MinMaxScaler().fit_transform(data)
data = pd.DataFrame(data)
data.columns=index

#split attribute and target class
X=data.drop(['Survived'],axis=1)
y=data['Survived']

#find outliers
FS=IsolationForest()
FS.fit(X)

# FS=EllipticEnvelope()
# FS.fit(X)

outliers=FS.predict(X)

drop=[]
for index,num in enumerate(outliers.tolist()):
    if(num==-1):
        drop.append(index)

X=X.drop(drop)
y=y.drop(drop)


# print(data.head())

# pca = decomposition.PCA(n_components=7)
# principalComponents = pca.fit_transform(X)
# principalDf = pd.DataFrame(data=principalComponents)

# estimator = svm.SVR(kernel='linear')
# selector = RFE(estimator, 5, step=1)
# principalDF=selector.fit_transform(X,y)
# principalDf=X

#split data
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#build model and evaluation
dtc = DecisionTreeClassifier(max_depth=5)
knn = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
svmm = svm.SVC(kernel='linear', C=1)

scores1 = cross_val_score(dtc, X_train, y_train, cv=10)
print("Crossval DTC: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
scores2 = cross_val_score(knn, X_train, y_train, cv=10)
print("Crossval KNN: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
scores3 = cross_val_score(svmm, X_train, y_train, cv=10)
print("Crossval SVM: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))

# model test
y_preddtc = dtc.fit(X_train, y_train).predict(X_test)
y_predknn = knn.fit(X_train, y_train).predict(X_test)
y_predsvm = svmm.fit(X_train, y_train).predict(X_test)

print("Accuracy DTC: %0.2f" % (accuracy_score(y_test, y_preddtc)))
print("Accuracy KNN: %0.2f" % (accuracy_score(y_test, y_predknn)))
print("Accuracy SVM: %0.2f" % (accuracy_score(y_test, y_predsvm)))

dtc_matrix = confusion_matrix(y_test, y_preddtc)
knn_matrix = confusion_matrix(y_test, y_predknn)
svm_matrix = confusion_matrix(y_test, y_predsvm)

print("Confusion matrix DTC:\n",dtc_matrix)
print("Confusion matrix KNN:\n",knn_matrix)
print("Confusion matrix SVM:\n",svm_matrix)
