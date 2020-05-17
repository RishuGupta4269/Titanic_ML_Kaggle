import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing training and test set
dataset1=pd.read_csv('train.csv')
dataset2=pd.read_csv('test.csv')
out=dataset2

#Dropping unnecessary features
dataset1=dataset1.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
dataset2=dataset2.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)

#replacing missing values
full=[dataset1,dataset2]
for i in full:
    i['Age']=i.Age.fillna(value=i.Age.mode()[0])
    i['Embarked']=i.Embarked.fillna(value=i.Embarked.mode()[0])
dataset2['Fare'] = dataset2.Fare.fillna(dataset2.Fare.mean())
#Encoding
dataset1=pd.get_dummies(dataset1,columns=['Sex','Embarked'],drop_first=True)
dataset2=pd.get_dummies(dataset2,columns=['Sex','Embarked'],drop_first=True)

#finding correlation
k=dataset1.corr()

#removing components with highest correlation using pca
from sklearn.decomposition import PCA

#first pair
df_1 = dataset1.loc[:,['Fare','Pclass']]
df_2 = dataset2.loc[:,['Fare','Pclass']]
pca =  PCA(n_components=1)
col_1 = pca.fit_transform(df_1)
col_2 = pca.fit_transform(df_2)

dataset1['Mod_col_1']=col_1[:,0]
dataset2['Mod_col_1']=col_2[:,0]

dataset1=dataset1.drop(['Fare','Pclass'], axis=1)
dataset2=dataset2.drop(['Fare','Pclass'], axis=1)

#second pair
df_3 = dataset1.loc[:,['SibSp','Parch']]
df_4 = dataset2.loc[:,['SibSp','Parch']]
pca =  PCA(n_components=1)
col_3 = pca.fit_transform(df_3)
col_4 = pca.fit_transform(df_4)

dataset1['Mod_col_2']=col_3[:,0]
dataset2['Mod_col_2']=col_4[:,0]

dataset1=dataset1.drop(['SibSp','Parch'], axis=1)
dataset2=dataset2.drop(['SibSp','Parch'], axis=1)

#Classification
X_train=dataset1.drop(['Survived'],axis=1)
y_train=dataset1['Survived']
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(dataset2)

#output
out=out.loc[:,['PassengerId']]
out['Survived']=y_pred[:]
name='Titanic_2.csv'
out.to_csv(name,index=False)
print('Saved file:'+name)