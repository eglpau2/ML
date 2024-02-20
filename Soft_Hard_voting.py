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
import seaborn as sns

#reading data
df = pathlib.Path(r'C:\Users\egle0\OneDrive\Dokumentai\data science mokykla\DUOMENYS\data_banknote_authentication.txt')
df1 = pd.read_csv(df, header=None).copy()
#summary
df1.describe()


# adding column name to the respective columns
df1.columns =['variance', 'skewness', 'curtosis', 'entropy', 'class']
print(df1.dtypes)
#cheak for nan
print(df1.isnull().sum())

pd.plotting.scatter_matrix(df1)
cor = df1.corr()

scaler = preprocessing.StandardScaler().fit(df1[['variance', 'skewness', 'curtosis', 'entropy']])
t = scaler.transform(df1[['variance', 'skewness', 'curtosis', 'entropy']])



df2 = df1.copy()

Y = df2['class']
features = [name for name in df2.columns if name not in ['class']]
X =  t#df2[['variance', 'skewness', 'curtosis', 'entropy']]
X1 = df2[features]

#train and test
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

################################################RF###########################################
RF_model = Pipeline([#('scaler', StandardScaler()),
                    ('predictor', RandomForestClassifier(n_estimators=100,
                                                        min_samples_split=20,
                                                        min_samples_leaf=5,
                                                        n_jobs=-1))])

RF_model.fit(X_train, Y_train)

RF_prediction = get_predictions(RF_model)
print_stats(RF_prediction, RF_model.named_steps['predictor'])

RocCurveDisplay.from_predictions(Y_test, RF_prediction['prob'])
plt.grid()
plt.axis('equal')
plt.show()

f1_dict = {'RF': f1_score(Y_test, RF_prediction['pred'])}
auc_dict = {'RF': roc_auc_score(Y_test, RF_prediction['prob'])}

#########################################################LR###########################################
LR_model = Pipeline([ #('scaler', StandardScaler()),
                   ('predictor', LogisticRegression(n_jobs=-1))])

LR_model.fit(X_train, Y_train)
LR_prediction = get_predictions(LR_model)
print_stats(LR_prediction, LR_model.named_steps['predictor'])

f1_dict['LR'] = f1_score(Y_test, LR_prediction['pred'])
auc_dict['LR'] = roc_auc_score(Y_test, LR_prediction['prob'])

RocCurveDisplay.from_predictions(Y_test, LR_prediction['prob'])
plt.grid()
plt.axis('equal')
plt.show()

######################## SOFT VOTING #########################################################

df_soft3 = pd.DataFrame(columns=['LR_poly_prob'], data=list(LR_prediction['prob']))
df_soft3['LR_poly_prob'] = LR_prediction['prob']
df_soft3['RF_prob'] = RF_prediction['prob']
df_soft3['prob'] = df_soft3.apply(lambda x:np.mean(x), axis=1)
df_soft3['pred'] = df_soft3['prob'].apply(lambda x:(x>0.5)*1)

print_stats(df_soft3, "soft  voting")

######################## HARD VOTING #########################################################
df_h3 = pd.DataFrame(columns=['LR_poly_pred'], data=list(LR_prediction['pred']))
df_h3['LR_poly_pred'] = LR_prediction['pred']
df_h3['RF_pred'] = RF_prediction['pred']
df_h3['prob'] = df_h3.apply(lambda x:np.mean(x), axis=1)
df_h3['pred'] = df_h3['prob'].apply(lambda x:(x>0.5)*1)

print_stats(df_h3, "hard voting")