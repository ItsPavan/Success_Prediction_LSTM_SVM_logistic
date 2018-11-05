import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

df = pd.read_excel('UseCase_4Test.xlsx')
#print (df.head(6))
#print (df.info())
correlation_matrix = df.drop(['Init_st_ts','init_end_ts','LocoID','Ops_Id'], axis=1).corr(method='spearman')
#print (correlation_matrix)
#sns.heatmap(df.corr(),linewidths = .5,annot=True,xticklabels=2,yticklabels=3,cmap="YlGnBu")
#plt.show()
#print (df.corr())
df1 = df.drop(['Init_st_ts','init_end_ts','LocoID','Ops_Id'],1)
#print (df1.corr())
df2 = df.drop(['Init_st_ts','init_end_ts','LocoID','Ops_Id','Init_time_taken','FLT_cnt','EMP_1000_cnt','EMP_2080_cnt','TRN_cnt','EMP_1011_cnt'],1)
print (df2.corr())
#print (df.isnull().sum())
#df1['Test_Success'] = df['Test_Success'].map({True: 1, False: 0})
#y = df1['Test_Success']
#df2['Test_Success'] = df['Test_Success'].map({True: 1, False: 0})
y = df2['Test_Success']

#print (df['Test_Success'])
#print (df.head(3))
# Barplot for dependant variable
#sns.countplot(x='Test_Success',data=df, palette='hls')
#plt.show()
#X_train, X_test, y_train, y_test = train_test_split(df1, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.2)
#Data Preprocessing
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)
#print (y_test)
# fit a model
from sklearn.metrics import confusion_matrix
logit1 = LogisticRegression(random_state=0)
logit1.fit(X_train, y_train)
train_y_pred = logit1.predict(X_train)
test_y_pred = logit1.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

#**************Printing training metrics*******************
print(confusion_matrix(y_train,train_y_pred))  
print(classification_report(y_train,train_y_pred))
#**************Printing testing metrics*******************
print(confusion_matrix(y_test,test_y_pred))  
print(classification_report(y_test,test_y_pred))  
#Below is the alternative way of computing confusion matrix using numpy
'''
train_confusion_matrix = np.array(confusion_matrix(y_train, train_y_pred))
print(train_confusion_matrix)
true_pos = np.diag(train_confusion_matrix)
false_pos = np.sum(train_confusion_matrix, axis=0) - true_pos
false_neg = np.sum(train_confusion_matrix, axis=1) - true_pos
train_acc = (np.sum(true_pos / (np.sum(train_confusion_matrix, axis=0) + np.sum(train_confusion_matrix, axis=1))))
train_acc_percentage = (train_acc)*100
print ('training accuracy : %.2f' %(train_acc_percentage)) # 99.77% - df1 100% - df2

test_y_pred = classifier.predict(X_test)
test_confusion_matrix = confusion_matrix(y_test, test_y_pred)
print(test_confusion_matrix)
true_pos = np.diag(test_confusion_matrix)
false_pos = np.sum(test_confusion_matrix, axis=0) - true_pos
false_neg = np.sum(test_confusion_matrix, axis=1) - true_pos
train_acc = (np.sum(true_pos / (np.sum(test_confusion_matrix, axis=0) + np.sum(test_confusion_matrix, axis=1))))
test_acc_percentage = (train_acc)*100
print ('testing accuracy : %.2f' %(test_acc_percentage)) #98.19% - df1 99.07% - df2
'''

