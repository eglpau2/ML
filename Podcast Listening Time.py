import matplotlib.pyplot as plt
import numpy as np
from paketas import Paket
from klase import Nustatymai
from data_view import Data_v
import pandas as pd
import matplotlib.pyplot as pyl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro
import matplotlib as pl
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
import seaborn as sb
import joblib
import tkinter
pl.use('Qt5Agg')

#class
p = Paket()
path = Nustatymai()
#data
traind = p.pd.read_csv(path.getT()).copy()
testd = p.pd.read_csv(path.getTest()).copy()
aprasa = Data_v(traind)

df1 = traind.copy()
df2test = testd.copy()


#data view missing values, data type
aprasa.aprasas()
#turime tris kintamuosius kuriuose yra praleistos reiksmes:
#Episode_Length_minutes
#Guest_Popularity_percentage
#Number_of_Ads
#Turime 6 tekstinius kintamuosius
aprasa.aspras_st()

#grafikai- pasizvalgymas po duomenis
axs = df1[df1.columns[1:12]].plot.area(figsize=(6, 6), subplots=True)
pyl.savefig('C:/Users/egle0/Documents/DUOMENYS/Predict Podcast Listening Time/axs.png')
pyl.close()


pd.plotting.scatter_matrix(df1, alpha=0.2, figsize=(8, 8), diagonal='kde' )
# Rotating X,Y-axis labels
pyl.tick_params(axis='x', labelrotation = 60)
pyl.tick_params(axis='y', labelrotation = 60)
pyl.savefig('C:/Users/egle0/Documents/DUOMENYS/Predict Podcast Listening Time/2.png')
pyl.close()

#praleitu  reiksmiu sutvarkymas
#Episode_Length_minutes - negalime suzinoti koki tiksliai ten turetu buti duomenys todel strinam juos
#df1 = df1.dropna(subset=['Episode_Length_minutes'])
df1 = df1.dropna(subset=['Number_of_Ads'])
df1 = df1.dropna(subset=['Guest_Popularity_percentage'])
df1 = df1.dropna(subset=['Episode_Length_minutes'])
#priskiriame 0 trukstamoms reiksmems kuri galesi reiksti arba is vis neturi arba nezinome ar apskritai turejo
#df1['Guest_Popularity_percentage'] = df1['Guest_Popularity_percentage'].fillna(-1)
#df1['Episode_Length_minutes'] = df1['Episode_Length_minutes'].fillna(-1)


#pakeiciame ka galime i skaitinius duomenis
#Episode_Title
df1['Episode_Title'] = df1['Episode_Title'].str.replace(r'Episode', '', regex=True).astype(int)
print(df1.head())
#Publication_Day
savaite = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
laikas = {'Morning':1, 'Afternoon':2, 'Evening':3, 'Night':4}
nuomone = {'Negative':-1, 'Neutral':0, 'Positive':1}
df1['Publication_Day'] = df1['Publication_Day'].apply(lambda x: savaite.get(x,-1)).astype(int)
df1['Publication_Time'] = df1['Publication_Time'].apply(lambda x: laikas.get(x,-1)).astype(int)
df1['Episode_Sentiment'] = df1['Episode_Sentiment'].apply(lambda x: nuomone.get(x,-1)).astype(int)
dfG = pd.get_dummies(df1['Genre']).astype(int)
DF2 = pd.concat([df1, dfG], axis=1)
aprasa2 = Data_v(DF2)
aprasa2.aprasas()

#normality cheak

def normality(data):
    result = {}
    for i in data.select_dtypes(['float64', 'int64']).columns:
        sample = data[i].sample(n=5000, random_state=42)
        stat, p = shapiro(sample)
        result[i] = {'p-value':p, 'Normal': p>0.05}
        a = result[i] if p>0.05 else print('Nėra normalumo:', sample.name)
    return a

#normalumo nėra
normality(DF2)


#reikia duomenis normalizuoti
scaler = MinMaxScaler()
DF2[['Episode_Title','Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time','Guest_Popularity_percentage', 'Number_of_Ads','Episode_Sentiment']] = scaler.fit_transform(DF2[['Episode_Title','Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time','Guest_Popularity_percentage', 'Number_of_Ads','Episode_Sentiment']])

#ISKIRCIU PASALINIMAS
#PASALINAME TUOS KURIE TURI IŠSKIRČIŲ
DF2 = DF2[DF2['Number_of_Ads']<0.1]
DF2 = DF2[DF2['Episode_Length_minutes']<0.8]
#SUTVARKYTI DUOMENYS
X = DF2.drop(columns=['Podcast_Name','Genre', 'Listening_Time_minutes', 'id', 'Episode_Title'])
Y = DF2[['Listening_Time_minutes']]

#isskirtys
def iskirtis(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers =(data < (Q1-1.5*IQR)) | (data > (Q3+1.5*IQR))
    return outliers

print(Y[iskirtis(Y).any(axis=1)]) #parodo eilute su isskirtimis
print(X[iskirtis(X).any(axis=1)]) #parodo eilute su isskirtimis

Y.plot(kind='box', subplots=True, layout=(3,4), figsize=(15,10), sharex=False, sharey=False)
pyl.title('box plot for outliers')
pyl.show()



pyl.figure(figsize=(15,8))
pyl.xticks(rotation =45)
sns.boxplot(data=[X['Number_of_Ads'], X['Episode_Length_minutes']] )
pyl.show()

#corelation
plt.figure(figsize=(10, 10))
sb.heatmap(X.corr() > 0.8,
           annot=True,
           cbar=False)
plt.show()


#padalinam duomenis i test ir train
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

def ivertis(model, x_test,y_test, y_pred):
    s = model.score(x_test, y_test)
    mae = mean_absolute_error(y_true= y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    rmse = np.sqrt(mse)
    return print("Score: ",s,"MAE: ", mae, "MSE:  ", mse, "RMSE: ", rmse)

def grafikas(model,y_pred,x_test,y_test):
    pyl.scatter(x_test,y_test,color = 'b')
    pyl.scatter(y_test, y_pred, color='k')
    pyl.savefig('C:/Users/egle0/Documents/DUOMENYS/Predict Podcast Listening Time/3.png')
    pyl.close()

#metodai

#LR
LR = Pipeline([('predictor', LinearRegression(n_jobs=1))])
LR.fit(x_train,y_train)
y_pred = LR.predict(x_test)
ivertis(LR, x_test,y_test, y_pred)

#XGB
XGB = Pipeline(steps=[('regressor', XGBRegressor(n_estimators=100, learning_rate=1, max_depth=3))])
XGB.fit(x_train,y_train)
y_predg = XGB.predict(x_test)
ivertis(XGB, x_test,y_test, y_predg)

#save model
model = 'C:/Users/egle0/Documents/DUOMENYS/Predict Podcast Listening Time/XGB.sav'
joblib.dump(XGB, model)

#LASO
LA = Pipeline(steps=[('regressor', Lasso(alpha=0.01))])
LA.fit(x_train,y_train)
y_predL = LA.predict(x_test)
ivertis(LA, x_test,y_test, y_predL)

#RIDGE
RI = Pipeline(steps=[('regressor', Ridge(alpha=0.01))])
RI.fit(x_train,y_train)
y_predR = RI.predict(x_test)
ivertis(RI, x_test,y_test, y_predR)


RF = Pipeline(steps=[('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
RF.fit(x_train,y_train)
y_predf = RF.predict(x_test)
ivertis(RF, x_test,y_test, y_predf)


##Test
aprasa_test = Data_v(testd)
aprasa_test.aspras_st()
aprasa_test.aprasas()

df2test = testd.copy()
#praleistos reikšmės
#priskiriame 0 trukstamoms reiksmems kuri galesi reiksti arba is vis neturi arba nezinome ar apskritai turejo
df2test['Guest_Popularity_percentage'] = df2test['Guest_Popularity_percentage'].fillna(-1)
df2test['Episode_Length_minutes'] = df2test['Episode_Length_minutes'].fillna(-1)

#keiciame tekstinius kintamuosius i skaicius
#pakeiciame ka galime i skaitinius duomenis
#Episode_Title
df2test['Episode_Title'] = df2test['Episode_Title'].str.replace(r'Episode', '', regex=True).astype(int)
print(df2test.head())
#Publication_Day

df2test['Publication_Day'] = df2test['Publication_Day'].apply(lambda x: savaite.get(x,-1)).astype(int)
df2test['Publication_Time'] = df2test['Publication_Time'].apply(lambda x: laikas.get(x,-1)).astype(int)
df2test['Episode_Sentiment'] = df2test['Episode_Sentiment'].apply(lambda x: nuomone.get(x,-1)).astype(int)
df2testG = pd.get_dummies(df2test['Genre']).astype(int)
DF3 = pd.concat([df2test, df2testG], axis=1)
aprasa3 = Data_v(DF3)
aprasa3.aprasas()

#reikia duomenis normalizuoti
DF3[['Episode_Title','Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time','Guest_Popularity_percentage', 'Number_of_Ads','Episode_Sentiment']] = scaler.fit_transform(DF3[['Episode_Title','Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time','Guest_Popularity_percentage', 'Number_of_Ads','Episode_Sentiment']])

#ISKIRCIU PASALINIMA
#SUTVARKYTI DUOMENYS
X1 = DF3.drop(columns=['Podcast_Name','Genre', 'id', 'Episode_Title'])

#uzsikraunam modeli
loaded_model = joblib.load(model)
ytest_pre = loaded_model.predict(X1)


results = pd.DataFrame({'Predict': ytest_pre})

#irasom rezultatą
rezultatas = 'C:/Users/egle0/Documents/DUOMENYS/Predict Podcast Listening Time/sample_submission - Copy.csv'
results.to_csv(rezultatas, mode='w', index=False)
