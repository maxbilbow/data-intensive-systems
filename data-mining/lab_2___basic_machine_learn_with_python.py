import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
# print(cancer.DESCR)
df=pd.DataFrame(cancer.data,columns=[cancer.feature_names])
df["target"] = cancer.target

X=df.iloc[:,0:30]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=426,test_size=143,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1) #loading
model.fit(X_train,y_train) #training

tmp = model.predict(X_test)
# print(tmp)

from sklearn import datasets
import matplotlib.pyplot as plt
digits = datasets.load_digits()

X=digits.data
y=digits.target
print(X[0])

plt.gray()
plt.matshow(digits.images[0])
plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

from sklearn import tree
DecTree = tree.DecisionTreeClassifier()
DecTree.fit(X_train, y_train)
print(DecTree.score(X_test,y_test))
