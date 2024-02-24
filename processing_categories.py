from ucimlrepo import fetch_ucirepo
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

# fetch dataset
census_income = fetch_ucirepo(id=20)

# data (as pandas dataframes)
X = census_income.data.features
y = census_income.data.targets

dx = pd.DataFrame(X)
dy = pd.DataFrame(y)

df = dx.merge(dy, left_index=True, right_index=True)

#check for nan
print(X.isnull().sum())
print(y.isnull().sum())
# metadata
print(census_income.metadata)
# variable information
print(census_income.variables)

#removing nan
df.dropna(inplace=True)

#summary of data
df.describe()
df.dtypes

# Count of Diffrent Text Elements in the Column
df['workclass'].value_counts()
df['education'].value_counts()
df['marital-status'].value_counts()
df['occupation'].value_counts()
df['relationship'].value_counts()
df['race'].value_counts()
df['sex'].value_counts()
df['native-country'].value_counts()
df['income'].value_counts()


#one_hot_encoded_data
one_hot_encoded_data = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                                   'race', 'native-country'], dtype='int')
print(one_hot_encoded_data)

one_hot_encoded_data.dtypes

# sex
one_hot_encoded_data['sex'] = one_hot_encoded_data['sex'].apply(lambda x: (x == 'Male')*1)
one_hot_encoded_data['sex'].value_counts()
# income
one_hot_encoded_data['income'] = one_hot_encoded_data['income'].apply(lambda x: (x == '<=50K')*1 | (x == '<=50K.')*1)
one_hot_encoded_data['income'].value_counts()

Y = one_hot_encoded_data['income']
features = [name for name in one_hot_encoded_data.columns if name not in ['income']]
X = one_hot_encoded_data[features]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def get_predictions(model):
    prediction = model.predict(X_test)
    output_df = pd.DataFrame(columns=['pred'], data=prediction)
    output_df['prob'] = model.predict_proba(X_test)[:, 1]
    return output_df

def print_stats(predictions, title):
    print(f"\t {title}")
    print(confusion_matrix(predictions['pred'], Y_test))
    print(classification_report(predictions['pred'], Y_test))
    print(f"F1 score: {f1_score(Y_test, predictions['pred'])}")
    print(f"AUC: {roc_auc_score(Y_test, predictions['prob'])}")

#########################################################LR###########################################
LR_model = Pipeline([('predictor', LogisticRegression(n_jobs=-1))])

LR_model.fit(X_train, Y_train)
LR_prediction = get_predictions(LR_model)
print_stats(LR_prediction, LR_model.named_steps['predictor'])

RocCurveDisplay.from_predictions(Y_test, LR_prediction['prob'])
plt.grid()
plt.axis('equal')
plt.show()

