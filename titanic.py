#%%
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import scipy
import numpy
import sys
import os



warnings.filterwarnings('ignore')
os.chdir('D:\Project\Titanic')

#%%
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
#%%
print(df_train.columns)
print(df_test.columns)
#%%
df_train.isnull().sum()
df_test.isnull().sum()
#%%
df_train['Age'].fillna((df_train['Age'].mean()), inplace=True)
df_test['Fare'].fillna((df_test['Fare'].mean()), inplace=True)

df_train.dropna()

#%%
print(type(df_train))
type(df_test)
#%%
ax= sns.boxplot(x="Pclass", y="Age", data=df_train)
ax= sns.stripplot(x="Pclass", y="Age", data=df_train, jitter=True, edgecolor="gray")
plt.show()
#%%

df_train['Age'].hist()


#%%
pd.plotting.scatter_matrix(df_train, figsize=(10,10))
plt.figure()
#%%
sns.violinplot(data=df_train, x='Sex',y="Age")

sns.FacetGrid(df_train, hue="Survived",size=5).map(sns.kdeplot, 'Fare').add_legend()

#%%
sns.jointplot(x='Fare',y="Age", data=df_train, kind='reg')
plt.show()
#%%
sns.swarmplot(x='Pclass',y="Age", data=df_train)

#%%
df_train.isnull().sum()
#%%
df_train.where(df_train ['Age']>=30).head(10)

# X = df_train.iloc[:, :-1].values
# y = df_train.iloc[:, -1].values

#%%
Pclass=pd.get_dummies(df_train['Pclass'],drop_first=True)
Pclass1=pd.get_dummies(df_test['Pclass'],drop_first=True)
#%%
Sex=pd.get_dummies(df_train['Sex'],drop_first=True)
Sex1=pd.get_dummies(df_test['Sex'],drop_first=True)
#%%
Embarked=pd.get_dummies(df_train['Embarked'],drop_first=True)
Embarked1=pd.get_dummies(df_test['Embarked'],drop_first=True)
#%%

#%%
df_train.head()
#%%
df_train=pd.concat([df_train,Pclass,Sex,Embarked],axis=1)
print(df_train)
print(df_train.head())
#%%
df_train.drop(['Sex','Embarked','Pclass','PassengerId','Cabin','Name', 'Ticket' ],axis=1,inplace=True)
df_test.drop(['Sex','Embarked',"Pclass",'PassengerId','Cabin','Name',"Ticket"], axis=1, inplace=True)

#%%
X=df_train.drop('Survived',axis=1)
y=df_train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
#%%
X_train.head()


#%%
model=LogisticRegression()
model.fit(X_train, y_train)
prediction=model.predict(X_test)
print(prediction)
#%%
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test, prediction)
print(score)
#%%
from sklearn.metrics import classification_report
classification_report(y_test, prediction)

#%%
