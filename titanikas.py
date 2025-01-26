#paketai
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from titanikas_klase import duomenu_tvarkymas #klases iskvietimas

#duomenys
test = "C:/Users/egle0/Documents/titanic/test.csv"
train = "C:/Users/egle0/Documents/titanic/train.csv"
Y =  "C:/Users/egle0/Documents/titanic/gender_submission.csv"

#testavimo duomenys
testd = pd.read_csv(test).copy()
#summary
print(testd.describe())
column = testd.columns.values
print(len(column))

#apmokymo duomenys
traind = pd.read_csv(train).copy()
#ar yra praleistų reikšmių
traind.isnull().sum()
#duomenų tipai
print(traind.dtypes)



# klase
p = duomenu_tvarkymas(traind)
#duomenys
traind_new1 = p.stulpeliu_tvark()
traind_new2 = p.Ticket().copy()
#pasaliname raides
traind_new2['Ticket'] = traind_new2['Ticket'] .str.replace(r'[a-zA-Z]', '', regex=True)

#pasaliname simbolius
def ticket_simboliai(text):
    text = text.replace(".", "")  # Remove single dots
    text = text.replace("/", "")  # Remove double dots
    text = text.replace(" ", "")  # Remove double dots
    return text

traind_new2['Ticket'] = traind_new2['Ticket'].apply(ticket_simboliai)
#pakeiciame tipa ticket i  int
traind_new2['Ticket'] = traind_new2['Ticket'].astype(int)
print(traind_new2.dtypes)

#sujungiame i viena rinkini y ir train duomenis
df_clean = traind_new2.dropna()
apibendrinimas = df_clean.describe()
print(df_clean.describe())

#duomenų normalizavimas arba standartizavimas
#transformuojamos į panašią skalę, užtikrinant,
#kad visi požymiai vienodai prisidėtų prie modelio mokymosi proceso.
#modeliai gali teikti pirmenybę didesnėms reikšmėms, todėl prognozės gali būti iškreiptos.
#Tai gali lemti prastą modelio veikimą ir lėtesnę konvergenciją mokymo metu.

#atsiskiriame duoemnis į X ir Y
X = df_clean[['Pclass', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Is_Female', 'Is_male', 'Is_C', 'Is_Q', 'Is_S']]
print(X.dtypes)
Y = df_clean['Survived']

##GRAFIKAI#########################
# Plotting the histogram.
pd.plotting.scatter_matrix(X, figsize=(8, 8))
plt.show()
X.hist(figsize=(10, 5), bins=10)
plt.show()

#normalizacija nes duomenys nėra normaliai pasiskirste
# Create a MinMaxScaler object
scaler = MinMaxScaler()
X_nor = pd.DataFrame(X)
# Normalize the selected columns
X_nor[['Pclass', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare']] = scaler.fit_transform(X_nor[['Pclass', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare']])

###Grafikai#############################
X_nor.hist(figsize=(10, 5), bins=10)
plt.show()
# Create box plot for all numeric columns
X.boxplot(figsize=(8, 6))
plt.show()


#koreliaciju matrica kas daro vienas kitam itaka- FAKTISKA NERA KORELIACIJOS DUOMENYSE
cormat = X_nor.corr()
round(cormat,2)
sns.heatmap(cormat, annot=True)

##Skeliame duomenis į ampmokymo ir testavimo imtis############
X_train, X_test, Y_train, Y_test = train_test_split(X_nor, Y, test_size=0.2, random_state=42)

#ALGORITMU KURIMAS

def get_predictions(model):
    #prognozes duomenys
    prediction = model.predict(X_test)
    #is prediction issima stulpeli
    output_df = pd.DataFrame(columns=['pred'], data=prediction)
    #tikimybe
    output_df['prob'] = model.predict_proba(X_test)[:, 1]
    return output_df

def print_stats(predi, title):
    print(f"\t {title}")
    print(confusion_matrix(predi['pred'], Y_test))
    print(classification_report(predi['pred'], Y_test))
    print(f"F1 score: {f1_score(Y_test, predi['pred'])}")
    print(f"AUC: {roc_auc_score(Y_test, predi['prob'])}")




print(X_nor.columns)
X.isnull().sum()
Y.isnull().sum()
###Logistic regresion####
#Labai mažų duomenų rinkinių atveju lygiagretinimo sąnaudos gali būti didesnės už galimą pagreitinimą.
# Tokiais atvejais efektyviau naudoti n_jobs=1.

LR_model = Pipeline([  ('predictor', LogisticRegression(n_jobs=1))])

LR_model.fit(X_train, Y_train)
LR_prediction = get_predictions(LR_model)
print_stats(LR_prediction, LR_model.named_steps['predictor'])

RocCurveDisplay.from_predictions(Y_test, LR_prediction['prob'])
plt.grid()

plt.axis('equal')
plt.show()

f1_dict = {'RF': f1_score(Y_test, LR_prediction['pred'])}
auc_dict = {'RF': roc_auc_score(Y_test, LR_prediction['prob'])}

# F1 score: 0.8421052631578948
# AUC: 0.8516666666666668

####GBC###################
GBC_model = Pipeline([ ('predictor', GradientBoostingClassifier(max_depth=5))])

GBC_model.fit(X_train, Y_train)
GBC_prediction = get_predictions(GBC_model)
print_stats(GBC_prediction, GBC_model.named_steps['predictor'])

RocCurveDisplay.from_predictions(Y_test, GBC_prediction['prob'])
plt.grid()

plt.axis('equal')
plt.show()

f1_dict = {'RF': f1_score(Y_test, GBC_prediction['pred'])}
auc_dict = {'RF': roc_auc_score(Y_test, GBC_prediction['prob'])}

#F1 score: 0.8333333333333334
#AUC: 0.84

#Bayes classifiers have worked quite well in many real-world situations,
# famously document classification and spam filtering.
# They require a small amount of training data to estimate the necessary parameters.
# (For theoretical reasons why naive Bayes works well, and on which types of data it does, see the references below.)
##GaussianNB##

GNB_model = Pipeline([ ('predictor', GaussianNB())])

GNB_model.fit(X_train, Y_train)
GNB_prediction = get_predictions(GNB_model)
print_stats(GNB_prediction, GNB_model.named_steps['predictor'])

RocCurveDisplay.from_predictions(Y_test, GNB_prediction['prob'])
plt.grid()

plt.axis('equal')
plt.show()

f1_dict = {'RF': f1_score(Y_test, GNB_prediction['pred'])}
auc_dict = {'RF': roc_auc_score(Y_test, GNB_prediction['prob'])}

#F1 score: 0.8421052631578948
#AUC: 0.75


##RF###

RF_model = Pipeline([('predictor', RandomForestClassifier(n_estimators=100))])

RF_model.fit(X_train, Y_train)
RF_prediction = get_predictions(RF_model)
print_stats(RF_prediction,RF_model.named_steps['predictor'])

RocCurveDisplay.from_predictions(Y_test, RF_prediction['prob'])
plt.grid()

plt.axis('equal')
plt.show()

f1_dict = {'RF': f1_score(Y_test, RF_prediction['pred'])}
auc_dict = {'RF': roc_auc_score(Y_test, RF_prediction['prob'])}

#F1 score: 0.8571428571428572
#AUC: 0.90

##VOTING_ MODEL###

eclf = VotingClassifier(estimators=[('rf', RF_model), ('gnb', GNB_model), ('GBC', GBC_model)],voting='soft')

for clf, label in zip([LR_model, RF_model, GNB_model, GBC_model, eclf], ['Random Forest', 'naive Bayes', 'GradientBoostingClassifier', 'Ensemble']):
    scores = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=5)

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


eclf = eclf.fit(X_train, Y_train)
eclf_prediction = get_predictions(eclf)
print_stats(eclf_prediction, "VotingClassifier")

#F1 score: 0.8771929824561403
#AUC: 0.87
