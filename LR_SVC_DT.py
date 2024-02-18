import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import preprocessing


dfp = pathlib.Path(r'C:\Users\egle0\OneDrive\Desktop\water_potability.csv')

df = pd.read_csv(dfp)
print(df.columns.values)

#pasaliname praleistas reiksmes

df2 = df.dropna()
print(df2.columns.values)

df3 = df2[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']].copy()



xc = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
yc = "Potability"




train, test0 = train_test_split(df3, test_size=0.4)
val, test = train_test_split(test0, test_size=0.6)
# Train 0.7, validation 0.15, test 0.15

def train_and_display_stats(model):
    scalerX = preprocessing.StandardScaler().fit(train[xc])
    X_train = scalerX.transform(train[xc])
    scalerXv = preprocessing.StandardScaler().fit(val[xc])
    X_val = scalerXv.transform(val[xc])
    model.fit(X_train, train[yc])
    preds = model.predict(X_val)
    print("\n" + model.__class__.__name__)
    print(confusion_matrix(val[yc], preds))
    print(classification_report(val[yc], preds))
    #roc kreive
    fpr, tpr, _ = metrics.roc_curve(val[yc], preds)
    auc = metrics.roc_auc_score(val[yc], preds)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


# LinearRegression
from sklearn.linear_model import LogisticRegression
ln = LogisticRegression(intercept_scaling=0.1, max_iter=1000)
train_and_display_stats(ln)
#ln.fit(train[xc], train[yc])
#preds = ln.predict(val[xc])
#print(preds)
#c = confusion_matrix(val[yc], preds)
#print(c)


#fig, ax = plt.subplots(figsize=(8, 8))
#ax.imshow(c)
#ax.grid(False)
#ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
#ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
#ax.set_ylim(1.5, -0.5)
#for i in range(2):
 #   for j in range(2):
  #      ax.text(j, i, c[i, j], ha='center', va='center', color='red')
#plt.show()

# Support vector machine
from sklearn.svm import SVC
clf = SVC()
train_and_display_stats(clf)

# Decision tree
from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
train_and_display_stats(dc)
tree.plot_tree(dc)

# DecisionTreeClassifier
# [[135  51]
#  [ 71  65]]
#               precision    recall  f1-score   support
#            0       0.66      0.73      0.69       186
#            1       0.56      0.48      0.52       136
#     accuracy                           0.62       322
#    macro avg       0.61      0.60      0.60       322
# weighted avg       0.62      0.62      0.62       322

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
train_and_display_stats(rf)

# RandomForestClassifier (komentuoti  - ctrl /)
# [[163  23]
#  [ 87  49]]
#               precision    recall  f1-score   support
#            0       0.65      0.88      0.75       186
#            1       0.68      0.36      0.47       136
#     accuracy                           0.66       322
#    macro avg       0.67      0.62      0.61       322
# weighted avg       0.66      0.66      0.63       322
