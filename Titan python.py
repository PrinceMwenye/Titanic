# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:13:11 2020

@author: Prince
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer


# you have to fit then transform


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
fulltitan = pd.concat([train, test], keys = ['train', 'test'])

fulltitan.head()
fulltitan.isna().sum() #many missing values by column

#FARE

nn = fulltitan[fulltitan['Fare'] == 0] #just interesting
#Those who didnt pay embarked from S and are all males. Let's check age distribution by Embarked
fulltitan.groupby('Embarked')['Fare'].mean()
fulltitan[fulltitan['Fare'].isna()]
fulltitan['Fare'].fillna(27.418, inplace = True) #replace missing Fare with mean of Embarked S
 
#EMBARKED

fulltitan[fulltitan['Embarked'].isna()]

#2 females, same ticket with missing Embarkation point
fulltitan.groupby(['Pclass', 'Embarked']).size()

#majority are in S class
fulltitan['Embarked'].fillna('S', inplace = True)

#CABIN AND SAME TICKET
fulltitan['Cabin'].isna().sum()
fulltitan['Cabin'].fillna('No_cabin', inplace = True)
fulltitan[(fulltitan['Ticket'] == 'PC 17757') & (fulltitan['Cabin'] == 'No_cabin')]
fulltitan.iloc[380,1] = 'C62'
fulltitan.iloc[557,1] = 'C62'
fulltitan.iloc[856,1] = 'C7'
fulltitan.iloc[217,1] = 'C7'
fulltitan.iloc[708, 1] = 'C4'
fulltitan.iloc[141,1] = 'C54'

import re
f = lambda x: x[0:1] if x != 'No_cabin' else 'No_room' #good for 
fulltitan['Area'] = fulltitan['Cabin'].apply(f) 


#AGE

#input age based on class
fulltitan['Age'].fillna(0, inplace = True)
def age_input(row):
    if row['Age'] == 0 and row['Pclass'] == 1:
        return 39.2
    elif row['Age'] == 0  and row['Pclass'] == 2:
        return 29.5
    elif row['Age'] == 0 and row['Pclass'] == 3:
        return 24.8
    else:
        return row['Age']

fulltitan['Age'] = fulltitan.apply(lambda row: age_input(row), axis =1 )
#group by ticket to find tickets with more than 1
helper = pd.DataFrame({'count': fulltitan.groupby('Ticket').size()}).reset_index()
helper = helper[helper['count'] >1]
fulltitan = pd.merge(fulltitan, helper, how = 'left')
fulltitan['count'].fillna('No_helper', inplace = True)

g = lambda f: 'Helper' if f != 'No_helper' else 'No_helper'
fulltitan['count'] = fulltitan['count'].apply(g)

fulltitan['Family_size'] = fulltitan['Parch'] + fulltitan['SibSp'] + 1
fulltitan = fulltitan.drop(['Cabin', 'Parch', 'SibSp', 'Ticket'], axis = 'columns')

import math
lo = lambda c: np.log10(c)
fulltitan['Fare'] = fulltitan['Fare'].apply(lo)

fulltitan['Fare']=fulltitan['Fare'].replace([-np.inf],0)

#bin age groups
Bins = [0,18,65,101]
fulltitan['Age_level'] = pd.cut(fulltitan['Age'], 
         bins = Bins, 
         labels = ['minor', 'adult', 'senior'])



fulltitan = fulltitan.drop(['Age'], axis = 'columns')



#titles

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
    
titles = sorted(set([x for x in train.Name.map(lambda x: get_title(x))]))

def replace_titles(x):
    title = x['Title']
    if title in ['Capt', 'Master', 'Lady', 'the Countess', 'Col', 'Jonkheer', 'Major', 'Rev', 'Sir' , 'Dr']:
        return 'Elite'
    elif title in [ 'Mme', 'Mrs', 'Mlle', 'Ms' 'Don', 'Dona', 'Mr', 'Miss']:
        return 'Regular'
    else:
        return title
    
fulltitan['Title'] = fulltitan['Name'].map(lambda x: get_title(x))
fulltitan['Title'] = fulltitan.apply(replace_titles, axis=1)
fulltitan = fulltitan.drop(['Name'], axis = 'columns')
fulltitan['Survived'].fillna(2, inplace = True)
ffam = [0,3,6,11]
fulltitan['Family_size'] = pd.cut(fulltitan['Family_size'], 
         bins = ffam, 
         labels = ['Small', 'Medium', 'Large'])

#OneHotEcode() if nominal categories
#LabelEncode() if ordinal categories
 
#Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', 
                                        OneHotEncoder(), 
                                        [0,2,3,4,5,6,7,8])] , 
                                       remainder = 'passthrough' )

fulltitanX = fulltitan.drop(['Survived'], axis = 'columns')
fulltitanX = fulltitanX.drop(['PassengerId'], axis = 'columns')
fulltitanY = fulltitan['Survived']

# independent variables encoded
fulltitanX = np.array(ct.fit_transform(fulltitanX))

#encoding response variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
fulltitanY = le.fit_transform(fulltitanY)


x_train = fulltitanX[0:890,]
x_test = fulltitanX[891:,]

y_train =fulltitanY[0:890,]
y_test = fulltitan.iloc[891:,]

#TRAIN/TEST sets #only for unknown train/test sets!!!!
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(X,y,)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100,
                                    criterion = 'entropy',
                                    random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

predictions = pd.DataFrame(y_pred, columns = ['Survived'])

y_test['Survived'] = y_pred 


