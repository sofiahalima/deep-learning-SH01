#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
cancer = datasets.load_breast_cancer()

print(cancer.data.shape)

# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

# print the cancer data features (top 5 records)
print(cancer.data[0:5])

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)


# Import train_test_split function
from sklearn.model_selection import train_test_split

split_size = 0.3

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=split_size, random_state=109) # 70% training and 30% test

#Import svm model
from sklearn import svm

#Create a svm Classifier
#clf = svm.SVC(kernel='linear') # Linear Kernel
clf = svm.SVC(kernel='rbf') # rbf Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

print("training done!!")

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#for i in y_pred:
#    print(i)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


