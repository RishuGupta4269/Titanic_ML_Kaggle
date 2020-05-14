import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing training set
dataset=pd.read_csv('train.csv')
X=dataset.iloc[:,[2,4,5,6,7,9]].values
y=dataset.iloc[:,1].values

#replacing missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='median',axis=0)
X[:,[2]]=imputer.fit_transform(X[:,[2]])

#encoding
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder_X=LabelEncoder()
X[:,1]=label_encoder_X.fit_transform(X[:,1])
ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X=X[:,1:]

X_train=X
y_train=y

#dimesionality reduction
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
explained_variance=pca.explained_variance_ratio_

# Training the Logistic Regression model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=6 , weights='distance',metric='minkowski' , p=2)
classifier.fit(X_train, y_train)

#Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier , X=X_train,y=y_train,cv=10)
m=accuracies.mean()

#Applying Grid Search
from sklearn.model_selection import GridSearchCV
parameters=[{'n_neighbors':[3,4,5,6,7],'weights':['uniform']},{'n_neighbors':[3,4,5,6,7],'weights':['distance']}]
grid_search=GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search=grid_search.fit(X_train, y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

#Importing test set
dataset2=pd.read_csv('test.csv')
d=dataset2['PassengerId']
X2=dataset2.iloc[:,[1,3,4,5,6,8]].values

#replacing missing data
imputer2=Imputer(missing_values='NaN',strategy='median',axis=0)
X2[:,2:]=imputer2.fit_transform(X2[:,2:])

#encoding
label_encoder_X2=LabelEncoder()
X2[:,1]=label_encoder_X2.fit_transform(X2[:,1])
ohe2=OneHotEncoder(categorical_features=[1])
X2=ohe2.fit_transform(X2).toarray()
X2=X2[:,1:]

#dimensionality reduction
X_test=X2
X_test=pca.transform(X_test)

#predicting the values
y_pred=classifier.predict(X_test)

#creating output data
sub=pd.DataFrame({'PassengerId':d,'Survived':y_pred})
filename = 'Titanic Predictions 1.csv'
sub.to_csv(filename,index=False)
print('Saved file: ' + filename)