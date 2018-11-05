import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

df = pd.read_excel('UseCase_4Test.xlsx')

#Data exploration and analysis
df.shape #enclose with print to print output on terminal
df.head(5) #enclose with print to print output on terminal

#Data preprocessing
df = df1 = df.drop(['Init_st_ts','init_end_ts','LocoID','Ops_Id'],1) # dropping variables of string type,which are not categorical so cannot be converted to classes
X = df.drop('Test_Success', axis=1)  
y = df['Test_Success'] 

#split data into training and testing sets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Training the algorithm
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)

#Making Predictions
y_train_pred = svclassifier.predict(X_train)
y_pred = svclassifier.predict(X_test) 

#Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix 
print('**************Printing training metrics*******************')
print(confusion_matrix(y_train,y_train_pred))  
print(classification_report(y_train,y_train_pred))
print('**************Printing testing metrics********************')
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  

