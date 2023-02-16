import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = './pulsar_star_dataset/pulsar_data_train.csv'

df = pd.read_csv(data)

# print(df.head)
# remove leading spaces from column names

df.columns = df.columns.str.strip()

# This removes the rows that contain infinity or Nan values
df = df[np.isfinite(df).all(1)]
# rename column names
df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness', 'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']
# print(df.columns)
# print(df['target_class'].value_counts())

# view the percentage distribution of target_class column
# print(df['target_class'].value_counts()/np.float(len(df)))

print(df.isnull().sum())
# view summary statistics in numerical variables
round(df.describe(),2)
# df[df['columns'] == ''].index
# df = df.dropna(subset=['columns'])
X = df.drop(['target_class'], axis=1)

y = df['target_class']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# print(X_train.shape, X_test.shape)
cols = X_train.columns
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

print(X_train.info())


# instantiate classifier with rbf kernel and C=100
svc=SVC(C=1000.0) 
# fit classifier to training set
svc.fit(X_train,y_train)
# make predictions on test set
y_pred=svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1000.0) 
# fit classifier to training set
linear_svc.fit(X_train,y_train)
# make predictions on test set
y_pred_test=linear_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
y_pred_train = linear_svc.predict(X_train)
print(y_pred_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))