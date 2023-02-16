
import pandas as pd
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
dir_path = 'data/'
file_path = dir_path + 'DMVWrittenTests.csv'

df = pd.read_csv(file_path)

print(df.shape)
X=[]
Y=[]

# get max row of panda
df_max = df.max()
for index, row in df.iterrows():
    X.append([row['DMV_Test_1']/df_max[0], row['DMV_Test_2']/df_max[1]])
    Y.append(row['Results'])

split_size = 0.2

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_size, random_state=1) 

clf = svm.SVC(kernel='rbf') # rbf Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))

