from math import gamma
from random import random
import pandas as pd
from sklearn import svm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

data_dir = 'data/'
red_wine_data = 'winequality-red.csv'
white_wine_data = 'winequality-red.csv'

df_red = pd.read_csv(data_dir+red_wine_data)
df_white = pd.read_csv(data_dir+white_wine_data)

net_wine_data = pd.concat([df_red,df_white])

fixed_acidity_list = []
residual_sugar_list = []
free_sulfur_list = []
density = []

X = []
Y = []
max_df_row = net_wine_data.max()
for index, row in net_wine_data.iterrows():
    # fixed_acidity_list.append(row['fixed acidity'])
    # residual_sugar_list.append(row['residual sugar'])
    # density.append(row['density'])
    X.append([row['fixed acidity']/max_df_row[0],row['volatile acidity']/max_df_row[1],row['citric acid']/max_df_row[2],row['residual sugar']//max_df_row[3]
    ,row['chlorides']/max_df_row[4],row['free sulfur dioxide']/max_df_row[5],row['total sulfur dioxide']/max_df_row[6],row['density']/max_df_row[7],row['pH']/max_df_row[8],row['sulphates']/max_df_row[9],row['alcohol']/max_df_row[10]])

    if(row['quality']>5):
        Y.append('good')
    else:
        Y.append('bad')
    
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=1)

# model = svm.SVC(kernel = 'rbf', gamma=40, C=1000)
# model = SVR(kernel = 'poly', degree=2, C=100, epsilon=0.1)
model = DecisionTreeClassifier(max_depth = 12)

model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(Y_test, y_pred))

# print(min(fixed_acidity_list))
# print(max(residual_sugar_list))
# print(min(residual_sugar_list))
# print(max(density))
# print(min(density))

