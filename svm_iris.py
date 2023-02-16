import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

data_dir = 'data/'
file_path = 'iris.csv'
df = pd.read_csv(data_dir + file_path)

sepal_length_X1 = []
sepal_width_X2 = []
petal_length_X3 = []
petal_width_X4 = []
X = []
Y = []
df_max = df.max()
for index, row in df.iterrows():
    X.append([row['sepal_length']/df_max[0],row['sepal_width']/df_max[1],row['petal_length']/df_max[2],row['petal_width']/df_max[3]])
    Y.append(row['species'])


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1) 
model = svm.SVC(kernel='rbf')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))
