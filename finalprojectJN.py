# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
import pandas
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Load Dataset
dataframe=pandas.read_csv('shotslog.csv', sep=",", header = 0, 
                           names = ["location","defender distance","shot number","quarter","game clock","shot clock",
                                    "dribbles","touch time","shot distance","points type","shot result"])
dataframe = dataframe.replace(np.nan,0)
shotdata=dataframe.values
print('data.shape', shotdata.shape)

#split into input(X) and output(Y) variables
x = shotdata[:,0:10]
y= shotdata [:,10]

print(x)
print(y)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
np.random.seed(2)

le = LabelEncoder() 
y_reshape = le.fit_transform(y)
x[:, 0] = le.fit_transform(x[:, 0])

# Scale the Data
sc = StandardScaler()
x_reshape = sc.fit_transform(x)

#Reshape the Data
# x_reshape = np.array(x).reshape(len(x), 10)
# y_reshape = np.array(y).reshape(len(y), 1)



print(x_reshape)
print(y_reshape)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

#create model
def create_model():
    model = Sequential()
    model.add(Dense(16, input_dim=10, activation='relu'))
    # model.add(Dense(32, init = 'uniform', activation='relu'))
    # model.add(Dense(32, init = 'uniform', activation='relu'))
    model.add(Dense(32, init = 'uniform', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    #compile network
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(x_reshape, y_reshape, test_size=0.3)
model=create_model()
history=model.fit(X_train,y_train, batch_size = 500, epochs=50, verbose=2, validation_data=(X_test, y_test))
print('Model Evaluation:', model.evaluate(X_train, y_train))
plt.figure()

plt.plot(history.history['acc'], color = 'red', label = 'Training Accuracy')
plt.plot(history.history['val_acc'], color = 'blue', label = 'Test Accuracy')
plt.title('Loss value')
plt.ylabel('Loss value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time = datetime.datetime.now()
print("Time required for training:", stop_time-start_time)