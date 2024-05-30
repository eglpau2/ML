import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, RocCurveDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib




#apmokymo duomenų nusiskaitymas------------------------------------------------------------
train = pathlib.Path(r'C:/Users/egle0/OneDrive/Dokumentai/duomenuanalizevirusu/train.csv')
traind = pd.read_csv(train).copy()

#testavimo duomenu nusiskaitymas-------------------------------------------------------
test = pathlib.Path(r'C:/Users/egle0/OneDrive/Dokumentai/duomenuanalizevirusu/test.csv')
testd = pd.read_csv(test).copy()

#-	Duomenų tyrinėjimas. Įvertinamas duomenų pasiskirstymas ir sąryšiai, nusakomos galimos duomenų transformacijos.
#duomenų aprašymas apmokymo ir testavimo-----------------
aptrai = traind.describe()
aptest = testd.describe()

# patikriname apmokymo duomenis dėl praleistų reikšmių----
traind.isnull().sum()  #praleistų reikšmių nėra

pd.plotting.scatter_matrix(traind)

#koreliacijos ir grafikai duomenu pasiskirstymo
cor = traind.corr()
sns.heatmap(cor, cmap = 'viridis', vmin = -1, vmax = 1, center = 0, annot=True, fmt=".2f", square=True, linewidths=.1)
plt.show() #ram turi stiprią koreliaciją su price_range

traind.hist()
testd.hist()
#-Duomenų paruošimas. Atliekamas duomenų 'valymas' bei transformavimas.


#suskirstome duomenis kas bus mūsų: y ir x
Y = traind['price_range']
X = traind.drop(['price_range'], axis=1)

#suvienodiname skales duomenų
scaler = preprocessing.StandardScaler().fit(X)
t = scaler.transform(X)
scaled_data = pd.DataFrame(t)
scaled_data.columns = X.columns

#išskirstome duomenis į mokymo ir testavimo rinkinius
X_train, X_test, Y_train, Y_test = train_test_split(scaled_data, Y, test_size=0.2, random_state=42)


#modelio prametrai
# rezultatų prognozavimas įrašomas į output_df
def get_predictions(model):
    prediction = model.predict(X_test)
    output_df = pd.DataFrame(columns=['pred'], data=prediction)
    return output_df


# modelio vertinimas:
# atspausdinama sumaišymo matrica
# klasifikavimo rezultatų aprašas
# paskaičiuojamas f1 parametras - vertinami modelio prognozavimo įgūdžiai, atsižvelgiant į jo klasių rezultatus

def print_stats(predictions, title):
    print(f"\t {title}")
    print(confusion_matrix(predictions['pred'], Y_test))
    print(classification_report(predictions['pred'], Y_test))
    print(f"F1 score: {f1_score(Y_test, predictions['pred'], average='macro')}")

# paskaičiuojamas plotas po kreive 
def acu_roc(model):
    y_score = model.predict_proba(X_test)
    print(f"AUC: {roc_auc_score(Y_test, y_score, multi_class='ovr')}")


#####################prenkame du modelius#############################
#1.Random forest-------------------------------------------------------------
#max_depth - medžio gylis  10
#n_estimators - medžių skaičius
#
RF_model = Pipeline([('predictor', RandomForestClassifier(n_estimators=300,
                                                          max_depth=15,
                                                          random_state=0
                                                          ))])


RF_model.fit(X_train, Y_train)

RF_prediction = get_predictions(RF_model)
print_stats(RF_prediction, RF_model.named_steps['predictor'])
acu_roc(RF_model)
#             precision    recall  f1-score   support
#           0       0.96      0.94      0.95       107
#           1       0.85      0.87      0.86        89
#           2       0.83      0.81      0.82        94
#           3       0.91      0.93      0.92       110
#    accuracy                           0.89       400
#   macro avg       0.89      0.89      0.89       400
#weighted avg       0.89      0.89      0.89       400
#F1 score: 0.8861272410572472
#AUC: 0.9847158193226493


# išsaugome modelį ---------------------------------------------------------------
filename = 'C:/Users/egle0/OneDrive/Dokumentai/duomenuanalizevirusu/RF_model.sav'
joblib.dump(RF_model, filename)



#2.GradientBoosting---------------------------------------------------------------

GLF_model = Pipeline([('predictor', GradientBoostingClassifier(n_estimators=300,
                                                               learning_rate=1.0,
                                                               max_depth=1,
                                                               random_state=0))])

GLF_model.fit(X_train, Y_train)

CLF_prediction = get_predictions(GLF_model)
print_stats(CLF_prediction, GLF_model.named_steps['predictor'])
acu_roc(GLF_model)
#              precision    recall  f1-score   support
#            0       0.96      0.97      0.97       104
#            1       0.93      0.91      0.92        93
#            2       0.89      0.89      0.89        92
#            3       0.94      0.95      0.94       111
#     accuracy                           0.93       400
#    macro avg       0.93      0.93      0.93       400
# weighted avg       0.93      0.93      0.93       400
# F1 score: 0.93085715105307
# AUC: 0.9928561608412656


# išsaugojame modelį
filename2 = 'C:/Users/egle0/OneDrive/Dokumentai/duomenuanalizevirusu/CLF_model.sav'
joblib.dump(GLF_model, filename)

#įkeliame modelį
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

#3.XGBClassifier--------------------------------------------------------------
XGB_model = Pipeline([('predictor', XGBClassifier(n_estimators= 300,
                                                  random_state=0))])

XGB_model.fit(X_train, Y_train)

XGB_prediction = get_predictions(XGB_model)
print_stats(XGB_prediction, XGB_model.named_steps['predictor'])
acu_roc(XGB_model)
#             precision    recall  f1-score   support
#            0       0.95      0.96      0.96       104
#            1       0.93      0.89      0.91        95
#            2       0.87      0.85      0.86        94
#            3       0.89      0.93      0.91       107
#     accuracy                           0.91       400
#    macro avg       0.91      0.91      0.91       400
# weighted avg       0.91      0.91      0.91       400
# F1 score: 0.9110933391406448
# AUC: 0.9923548500248273






#pratestuojame su vienu iš modelių ant test duomenų
#td = testd.drop(columns='id')
#t2 = scaler.transform(td)
#scaled_data2 = pd.DataFrame(t2)
#scaled_data2.columns = td.columns
#pt1 = GLF_model.predict(scaled_data2)
#testd['price_range'] = pt1
#testd.to_csv(train.parent.joinpath('EP_python.csv'))