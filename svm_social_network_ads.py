import pandas as pd
from sklearn import svm
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn import metrics


data_dir = "data/"
file_path = data_dir +'SocialNetworkAds.csv'
df = pd.read_csv(file_path)

#print(df.shape)
#print(df.head)

list_x1 = []
list_x2 = []
list_y = []

X = []
Y = []

df_max = df.max()
print(df_max[0])

for index, row in df.iterrows():
    X.append([row['Age'] / df_max[0], row['EstimatedSalary'] / df_max[1]])
    Y.append(row['Purchased'])

split_size = 0.2

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_size, random_state=100) 

clf = svm.SVC(kernel='rbf') # rbf Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))
