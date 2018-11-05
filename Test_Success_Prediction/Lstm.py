import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_add(file_name):
    
    dataset = pd.read_csv(file_name)

    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    return dataset,X,y

dataset_train,X,y = load_and_add('data_train.csv')

def make_encoding(X):
    for i in range(len(X[1])):
        if str(type(X[:,i][0])) == "<class 'str'>":
            labelencoder = LabelEncoder()
            X[:, i] = labelencoder.fit_transform(X[:, i])
    return X


X = make_encoding(X)

X = X.astype(float)
#y = make_encoding(y)

labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)


## if needed we can add one hot encoding
'''
def one_hot_encoding(X):
    onehotencoder1 = OneHotEncoder(categorical_features = [1])
    X = onehotencoder1.fit_transform(X).toarray()
    X = X[:,1:]
    return X

X = one_hot_encoding(X)
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

params = {
            "optimizer": "adam",
			"rnn_activation_1":"relu",
            "rnn_activation_2":"sigmoid",
            "rnn_activation_3":"softmax",
            "kernel_initializer":"truncated_normal",
            "units_lstm" : 16,
			"loss": "binary_crossentropy",
			"dropout_rate": 0.1,
            "output_units":1,
			"epochs": 100,
			"model_name": "lstm_model",
			"batch_size": 20,
			"verbose": True
		}

#params[""]
def build_model():
    model = Sequential()
    
    model.add(Dense(params["units_lstm"], kernel_initializer=params["kernel_initializer"], activation = params["rnn_activation_1"], input_shape = (X.shape[1],)))
    model.add(Dropout(params["dropout_rate"]))
    
    model.add(Dense(params["units_lstm"], kernel_initializer=params["kernel_initializer"], activation = params["rnn_activation_1"], input_shape = (X.shape[1],)))
    model.add(Dropout(params["dropout_rate"]))
    
    model.add(Dense(params["units_lstm"], kernel_initializer=params["kernel_initializer"], activation = params["rnn_activation_1"], input_shape = (X.shape[1],)))
    model.add(Dropout(params["dropout_rate"]))
    
    model.add(Dense(params["output_units"], kernel_initializer=params["kernel_initializer"], activation = params["rnn_activation_2"] ))
    
    model.compile(optimizer = params["optimizer"], loss = params["loss"], metrics = ["accuracy"])
    
    return model

model = build_model()
model.fit(X_train,y_train, batch_size = params["batch_size"], epochs = params["epochs"])
score,acc = model.evaluate(X_test,y_test,verbose = 2,batch_size = params["batch_size"])
print ('test score : %.2f' %(score))
print ('test accuracy: %.2f' %(acc))

def prediction(model,X):
    pred = model.predict_classes(X)
    pred = (pred > 0.5)

    return pred

dataset_test,x_test,y_test = load_and_add('data_train.csv')
x_test = make_encoding(x_test)
#x_test = one_hot_encoding(x_test)
x_test = sc.fit_transform(x_test)
y_test = labelencoder_y.fit_transform(y_test)
pred = prediction(model,x_test)








