# importing the libraries
#%%
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import sklearn
import sys
import os 

os.chdir("D:\Project\Bank loan")
warnings.filterwarnings('ignore')

#%%   importing the data
df=pd.read_csv('credit_risk.csv')
print(df.info())

#%% converting into train and test as there is no test file 
df_train, test= train_test_split(df, test_size=0.2, random_state=1) # we will not make changes to test file

#%% for columns
df_train.columns   # all columns of data
#%%
df_train.shape   # shape of data
#%%
df_train.size   # shape of data
#%%
df_train.head()   # for first 5 rows and all columns
#%%
df_train.describe()
#%%
sns.heatmap(df_train.isnull())  # plotting graph a to check for the missing value
plt.savefig('heatmap.png')
plt.show()
plt.savefig('heatmap.PNG')
#%%
df_train.isnull().sum()
#%%
'''
df_train.isnull().sum()
Loan_ID               0
Gender                9
Married               2
Dependents           10
Education             0
Self_Employed        24
ApplicantIncome       0
CoapplicantIncome     0
LoanAmount           17
Loan_Amount_Term     10
Credit_History       44
Property_Area         0
Loan_Status           0'''
#%%
gender=df.Gender.value_counts()
gender
#%%
# _____________________---visualization of data______________
# check the distribution of persons who takes the loan male or female
def male_percent(df):
    gender=df.Gender.value_counts()
    male=gender.loc['Male']
    female=gender.loc['Female']
    male_ratio=male/(male+female)
    return male_ratio

male_ratio=male_percent(df_train)
female_ratio=1-male_ratio
male_ratio
#%%
df_train.head()

#%%
sns.scatterplot(y='LoanAmount', x="ApplicantIncome", data=df_train)
plt.savefig('scatterplot.png')
plt.show()


#%%
sns.swarmplot(x='LoanAmount',y="ApplicantIncome", data=df_train)
plt.savefig('swarmplot.PNG')

#%%
ax= sns.boxplot(x="LoanAmount", y="ApplicantIncome", data=df_train)
ax= sns.stripplot(x="LoanAmount", y="ApplicantIncome", data=df_train, jitter=True, edgecolor="gray")
plt.show()

#%%
pd.plotting.scatter_matrix(df_train, figsize=(10,10))
plt.figure()

#%%
sns.distplot(df_train.LoanAmount.dropna(), bins=10)
plt.show()
#%%
sns.distplot(df_train['ApplicantIncome'], bins=10)  # Clearly distribution is not normal
plt.show()

#%%
df_train['Gender'].hist()
#%%
gender_mode=df_train.Gender.mode()
married_mode=df_train.Married.mode()
dependent_mode=df_train.Dependents.mode()
education_mode=df_train.Education.mode()
self_employed_mode=df_train.Self_Employed.mode()
loam_anmout=df_train.LoanAmount.median()
Loan_Amount_Term_median=df_train.Loan_Amount_Term.median()
Credit_History_mode=df_train.Credit_History.mode()


def fill_gender(df):
    df.Gender.fillna(gender_mode[0], inplace=True)
    return df

def fill_married(df):
    df.Married.fillna(married_mode[0], inplace=True)
    return df

def fill_dependent(df):
    df.Dependents.fillna(dependent_mode[0], inplace=True)
    return df

def fill_education(df):
    df.Education.fillna(education_mode[0], inplace=True)
    return df

def fill_self_employed(df):
    df.Self_Employed.fillna(self_employed_mode[0], inplace=True)
    return df

def fill_LoanAmount(df):
    df.LoanAmount.fillna(loam_anmout, inplace=True)
    return df

def fill_Loan_Amount_Term(df):
    df.Loan_Amount_Term.fillna(Loan_Amount_Term_median, inplace=True)
    return df

def fill_Credit_History(df):
    df.Credit_History=df.Credit_History.fillna(Credit_History_mode[0], inplace=False)
    return df

def multiply(df):
    df.LoanAmount=df.LoanAmount*100
    df.ApplicantIncome=df.ApplicantIncome*10
    df.CoapplicantIncome=df.CoapplicantIncome*10
    return df

def label_encoder(df):
    label=LabelEncoder()
    columns=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
    for column in columns:
        df[column]=label.fit_transform(df[column])
    return df

def encode_data(df):
    df=fill_gender(df)
    df=fill_married(df)
    df=fill_dependent(df)
    df=fill_education(df)
    df=multiply(df)
    df=fill_self_employed(df)
    df=fill_LoanAmount(df)
    df=fill_Loan_Amount_Term(df)
    df=fill_Credit_History(df)
    df=label_encoder(df)
    return df

df_train=encode_data(df_train)

#%%
df_train.head()
#%%
x=df_train.drop(['Loan_ID','Loan_Status'], axis=1)
y=df_train['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=12, test_size=0.2)

#%%
log_model=LogisticRegression()
log_model.fit(x_train, y_train)
predict=log_model.predict(x_train)
score_x_train=accuracy_score(y_train, predict)
print(score_x_train)

#%%
test=encode_data(test)
testx=test.drop(['Loan_ID','Loan_Status'], axis=1)
testy=test['Loan_Status']

#%%
predict_testx=log_model.predict(testx)
final_test=accuracy_score(testy, predict_testx)
print(final_test)
#%%
