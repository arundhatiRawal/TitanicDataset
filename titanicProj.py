# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sb
sb.set() # setting seaborn default for plots
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
dataset1=pd.read_csv(r"C:\Users\PRANAV\Titanic\train.csv")
dataset=pd.read_csv(r"C:\Users\PRANAV\Titanic\train.csv")
dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.')

dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
le=LabelEncoder()
dataset.Title=le.fit_transform(dataset.Title)
dataset.Sex=le.fit_transform(dataset.Sex)
dataset.Embarked.unique()
dataset['Embarked'] = dataset['Embarked'].fillna('S')#because S was maximum we replaced nan with S
dataset.Embarked=le.fit_transform(dataset.Embarked)
dataset['Embarked']
age_avg = dataset['Age'].mean()
age_std = dataset['Age'].std()
age_null_count = dataset['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
dataset['Age'] = dataset['Age'].astype(int)
    
dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
dataset.AgeBand=le.fit_transform(dataset.AgeBand)
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)
dataset.FareBand=le.fit_transform(dataset.FareBand)
dataset['FareBand']
dataset1['Fare'] = dataset1['Fare'].fillna(dataset1['Fare'].median())
dataset1['FareBand'] = pd.qcut(dataset1['Fare'], 4)

dataset1['FareBand']
dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1
dataset['IsAlone'] = 0
dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
dataset.IsAlone=le.fit_transform(dataset.IsAlone)
dataset['IsAlone']
dataset1['FamilySize'] = dataset1['SibSp'] +  dataset1['Parch'] + 1
dataset1['IsAlone'] = 0
dataset1.loc[dataset1['FamilySize'] == 1, 'IsAlone'] = 1
dataset1.IsAlone=le.fit_transform(dataset1.IsAlone)

#dropping unnecessary features
features_drop = ['PassengerId','Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
dataset = dataset.drop(features_drop, axis=1)
dataset = dataset.drop('Age', axis=1)
dataset = dataset.drop('Fare',axis=1)
dataset.head()
#seggregate
x=dataset.iloc[:,1:]
y=dataset.iloc[:,0]
x.head()
y.head()
#splitting
x_train=x[:713]
y_train=y[:713]
x_test=x[713:]
y_test=y[713:]
dataset.shape
from sklearn.svm import SVC
teacher = SVC()
learner=teacher.fit(x_train, y_train)
y_pred_svc = learner.predict(x_test)
y_acc=y_test
from sklearn.metrics import accuracy_score
acc= accuracy_score(y_acc,y_pred_svc)*100
print (acc)
p= (learner.predict([[1,0,2,1,3,2,1]]))
if p==1:
    print('Survived')
else:
    print('Not Survived')
import easygui as eg


def func1():
    
    option=["ACCURACY","PCLASS DEPENDENCY","GENDER DEPENDENCY","EMBARKED  DEPENDENCY","SEX & AGE DEPENDENCY","SibSp Dependency","FARE  DEPENDENCY","ALONE  DEPENDENCY"]
    button = eg.buttonbox("CHOOSE A BUTTON",choices=option)
    if button == option[0]:
        eg.msgbox(msg=acc, title=' ', ok_button='OK')
    elif button == option[1]:
        dataset1[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
        sb.countplot('Pclass',hue='Survived',data=dataset1)
        plt.show()
    elif button == option[2]:
        dataset1[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
        sb.countplot('Sex',hue='Survived',data=dataset1)
        plt.show()
    elif button == option[3]:
        dataset1[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar()
        sb.countplot('Embarked',hue='Survived',data=dataset1)
        plt.show()
    elif button == option[4]:
        f,ax=plt.subplots(1,2,figsize=(18,8))
        sb.violinplot("Sex","Age", hue="Survived", data=dataset1,split=False,ax=ax[0])
        ax[0].set_title('Sex and Age vs Survived')
        ax[0].set_yticks(range(0,110,10))
        plt.show()
    elif button == option[5]:
        f,ax=plt.subplots(1,2,figsize=(20,8))
        sb.barplot('SibSp','Survived', data=dataset1,ax=ax[0])
        ax[0].set_title('SipSp vs Survived in BarPlot')
        sb.factorplot('SibSp','Survived', data=dataset1,ax=ax[1])
        ax[1].set_title('SibSp vs Survived in FactorPlot')
        plt.close(2)
        plt.show()
    elif button == option[6]:
        dataset1[['FareBand','Survived']].groupby(['FareBand']).mean().plot.bar()
        sb.countplot('FareBand',hue='Survived',data=dataset1)
        plt.show()
    elif button == option[7]:
        dataset1[['IsAlone','Survived']].groupby(['IsAlone']).mean().plot.bar()
        sb.countplot('IsAlone',hue='Survived',data=dataset1)
        plt.show()
     
    

def func2():
    msg = "TITANIC DATASET PREDICTION"
    title = "Titanic "
    fieldNames = ["PCLASS 0:Ist 1:2nd 2:3rd","GENDER 0:Female 1:Male","EMBARKED C:0 Q:1 S:2","TITLE 0-Master 1-Miss 2-Mr 3-Mrs 4-Other","AGE (0-16)-0 (16-32)-1 (32-48)-2 (48-64)-3 (>64)-4","FARE (0-7.9)-0 (7.91-14.45)-1 (14-31)-2 (31-512)-3","ARE YOU ALONE YES-1 NO-0"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = eg.multenterbox(msg,title, fieldNames)
    while 1:
        if fieldValues is None: 
            break
        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg += ('"%s" is a required field.\n\n' % fieldNames[i])
        if errmsg == "":
            pr= (learner.predict([fieldValues]))
            if pr==1:
                eg.msgbox(msg='(SURVIVED TITANIC)', title=' ', ok_button='OK')
            else:
                eg.msgbox(msg='(NOT SURVIVED)', title=' ', ok_button='OK')
        break # no problems found
option1=["TITANIC DATASET PREDICTION","TITANIC DATASET ANALYSIS"]
button1 = eg.buttonbox("CHOOSE A BUTTON",choices=option1)
if button1 == option1[0]:
    
    func2()
    
elif button1== option1[1]:
    func1()
