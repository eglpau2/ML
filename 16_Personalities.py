import numpy as np
import pandas as pd
import pathlib
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics


df = pathlib.Path(r'C:\Users\egle0\OneDrive\Dokumentai\data science mokykla\DUOMENYS\16P.csv')
df1 = pd.read_csv(df, encoding='cp1252')

df1.head()

#cheak for nan
print(df1.isnull().sum())



y = df1['Personality']
x = df1.drop(['Response Id', 'Personality'], axis=1)

#pd.plotting.scatter_matrix(x)
#cor = x.corr()

# encoding y
l = LabelEncoder()
y = l.fit_transform(y)
y_ = np.array(y, dtype=np.int16)

print(x.shape, y_.shape)


#spliting to train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size= 0.2, random_state = 1)

print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_test dimension= ', y_test.shape)





#####################################lr####################

lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
lm.fit(X_train, y_train)

lm.score(X_test, y_test)


#Creating matplotlib axes object to assign figuresize and figure title

y_pred = lm.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

print(metrics.classification_report(y_test, lm.predict(X_test)))
