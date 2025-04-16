import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, RocCurveDisplay, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle



Train_d = "C:/Users/egle0/Documents/DUOMENYS/paskolos/train.csv"
Test_d = "C:/Users/egle0/Documents/DUOMENYS/paskolos/test.csv"


train_df = pd.read_csv(Train_d).copy()
print(train_df.describe())

test_df = pd.read_csv(Test_d).copy()
print(test_df.describe())


#turime 4 tekstinius ir 8 - skaitinius
print(train_df.dtypes)


#praleistu reiksmiu nera
print(train_df.isnull().sum())


#sutvarkysime tekstines reiksmes:
#1 person_home_ownership
def stulpelis(df):
   df = pd.get_dummies(df, columns=['person_home_ownership'], prefix='', dtype=int)
   df.rename(columns={'RENT': 'is_rent', 'OWN': 'is_own', 'MORTGAGE': 'is_morgage'}, inplace=True)
   #df= df.drop(columns=['OTHER'])
   return df

df = stulpelis(train_df)


#2 loan_intent
def loan_intent(df):
   df = pd.get_dummies(df, columns=['loan_intent'], prefix='', dtype=int)
   df.rename(columns={'EDUCATION': 'intent_education', 'MEDICAL': 'intent_medical', 'PERSONAL': 'intent_personal', 'VENTURE': 'intent_venture','DEBTCONSOLIDATION': 'intent_debtconsol', 'HOMEIMPROVEMENT': 'intent_homeimprove' }, inplace=True)
   return df
df2 = loan_intent(df)

#3 loan_grade
# Create a dictionary to map values to numbers
value_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F':6, 'G':7}
# Map the values in the 'category' column using the dictionary
df2['loan_grade_numeric'] = df2['loan_grade'].map(value_mapping)
df2['loan_grade'].unique()
df2['loan_grade_numeric'] = df2['loan_grade_numeric'].astype(int)


#4 cb_person_default_on_file
df2['Y_cb_person_default_on_file'] = (df2['cb_person_default_on_file'] =='Y').astype(int)


#normalizavimas
# sukuriame min max saklės objektą
scaler = MinMaxScaler()


# Normalize the selected columns
df2[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_grade_numeric']] = scaler.fit_transform(df2[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_grade_numeric']])
print(df2.dtypes)
df3 = df2[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', '_MORTGAGE', '_OWN', '_RENT', '_DEBTCONSOLIDATION', '_EDUCATION', '_HOMEIMPROVEMENT', '_MEDICAL', '_PERSONAL', '_VENTURE', 'loan_grade_numeric', 'Y_cb_person_default_on_file']]
Y = df2['loan_status']


print(Y.shape)
#sukurmiame testavimo ir mokymo imtis
train_x, test_x, train_y, test_y = train_test_split(df3, Y, test_size=0.2, random_state=42)


#ML algoritmai


def get_predictions(model, X_test):
   #prognozes duomenys
   prediction = model.predict(X_test)
   #is prediction issima stulpeli
   output_df = pd.DataFrame(columns=['pred'], data=prediction)
   #tikimybe
   output_df['prob'] = model.predict_proba(X_test)[:, 1]
   return output_df


def print_stats(predi, Y_test, title):
   print(f"\t {title}")
   print(confusion_matrix(predi['pred'], Y_test))
   print(classification_report(predi['pred'], Y_test))
   print(f"F1 score: {f1_score(Y_test, predi['pred'])}")
   print(f"AUC: {roc_auc_score(Y_test, predi['prob'])}")




#LR###
LR_model = Pipeline([ ('predictor', LogisticRegression(n_jobs=-1, max_iter=100000))])


LR_model.fit(train_x, train_y)
LR_prediction = get_predictions(LR_model, test_x)
print_stats(LR_prediction, test_y, 'LR_model')

#F1 score: 0.5676190476190476
#AUC: 0.895240095698483




#RF#
RF_model = Pipeline([ ('predictor', RandomForestClassifier(n_estimators=1000))])


RF_model.fit(train_x, train_y)
RF_model_prediction = get_predictions(RF_model, test_x)
print_stats(RF_model_prediction, test_y, 'RF_model')

#F1 score: 0.8106448311156602
#AUC: 0.9437145615121645

RocCurveDisplay.from_predictions(test_y, RF_model_prediction['prob'])
plt.grid()


plt.axis('equal')
plt.show()


#DF###
DF_model = Pipeline([('predictor', DecisionTreeClassifier(max_depth=1000, min_samples_split=1000))])


DF_model.fit(train_x, train_y)
DF_prediction = get_predictions(DF_model, test_x)
print_stats(DF_prediction, test_y, 'DF_model')

#F1 score: 0.7697160883280757
#AUC: 0.9338724171570914

#ETR#
ETR_model = Pipeline([('predictor', ExtraTreesClassifier(max_depth=100, min_samples_split=10))])


ETR_model.fit(train_x, train_y)
ETR_prediction = get_predictions(ETR_model, test_x)
print_stats(ETR_prediction, test_y, 'ETR_model')

#
BA_model = Pipeline([('predictor', BaggingClassifier(KNeighborsClassifier(), max_samples=0.1, max_features=0.5))])


BA_model.fit(train_x, train_y)
BA_prediction = get_predictions(BA_model, test_x)
print_stats(BA_prediction, test_y, 'BA_model')

#F1 score: 0.65443186255369
#AUC: 0.9157640645748613

#IŠSAUGOJAME GERIAUSIA MODELI
modelis_geriausias = "modelis.pkl"

with open (modelis_geriausias, "wb") as file:
   pickle.dump(RF_model, file)

#uzsikrauti modeli

with open(modelis_geriausias, "rb") as file:
   modelis = pickle.load(file)




#BANDYSIME GAUTI PREDICT REIKSMES
#1 person_home_ownership
def stulpelis(df):
   df = pd.get_dummies(df, columns=['person_home_ownership'], prefix='', dtype=int)
   df.rename(columns={'RENT': 'is_rent', 'OWN': 'is_own', 'MORTGAGE': 'is_morgage'}, inplace=True)
   #df= df.drop(columns=['OTHER'])
   return df


df_te = stulpelis(test_df)

def loan_intent(df):
   df = pd.get_dummies(df, columns=['loan_intent'], prefix='', dtype=int)
   df.rename(columns={'EDUCATION': 'intent_education', 'MEDICAL': 'intent_medical', 'PERSONAL': 'intent_personal', 'VENTURE': 'intent_venture','DEBTCONSOLIDATION': 'intent_debtconsol', 'HOMEIMPROVEMENT': 'intent_homeimprove' }, inplace=True)
   return df
df_te2 = loan_intent(df_te)

#3 loan_grade
# Create a dictionary to map values to numbers
value_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F':6, 'G':7}
# Map the values in the 'category' column using the dictionary
df_te2['loan_grade_numeric'] = df_te2['loan_grade'].map(value_mapping)
df_te2['loan_grade'].unique()
df_te2['loan_grade_numeric'] = df_te2['loan_grade_numeric'].astype(int)

#4 cb_person_default_on_file
df_te2['Y_cb_person_default_on_file'] = (df_te2['cb_person_default_on_file'] =='Y').astype(int)

#normalizavimas
# Normalize the selected columns
df_te2[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_grade_numeric']] = scaler.fit_transform(df_te2[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_grade_numeric']])
print(df_te2.dtypes)
df_te3 = df_te2[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', '_MORTGAGE', '_OWN', '_RENT', '_DEBTCONSOLIDATION', '_EDUCATION', '_HOMEIMPROVEMENT', '_MEDICAL', '_PERSONAL', '_VENTURE', 'loan_grade_numeric', 'Y_cb_person_default_on_file']]

#modelio paleidimas
_prediction = get_predictions(modelis, df_te3)
tikimybe = _prediction[["pred"]]
tiketi = _prediction[["prob"]]

test_df = [tikimybe, df_te]
df_su_tikimybe2 = pd.concat(test_df, axis=1)

frames = [tikimybe, df_te2]
df_su_tikimybe = pd.concat(frames, axis=1)

rez = "C:/Users/egle0/Documents/DUOMENYS/paskolos/sample_submission.csv"

re = pd.read_csv(rez)
re['prognoze'] = tikimybe
re['tiketinumas'] = tiketi
re.to_csv(rez, index=False)
