#Support Vector Machine for cancer classification


#Importing the libraries
import numpy as np
import pandas as pd

#Importing the data set
dataset = pd.read_csv('cancerdata.csv')
x = dataset.iloc[:,2:24].values
y = dataset.iloc[:, -1].values

#Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)

#splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state =1)

#Training the SVM model on training data
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

#Predicting test results
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#The confusion matrix to evaluate accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
      



