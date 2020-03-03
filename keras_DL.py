""" import numpy as np

from sklearn.datasets import load_iris

iris = load_iris()

# print(iris.DESCR)

X = iris.data
y = iris.target

from keras.utils import to_categorical

y = to_categorical(y)

print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler_object = MinMaxScaler()

scaler_object.fit(X_train)

scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(scaled_X_train, y_train, epochs=150, verbose=2)

#model.predict(scaled_X_test)
predictions = model.predict_classes(scaled_X_test)

y_test.argmax(axis=1)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test.argmax(axis=1), predictions))
print(confusion_matrix(y_test.argmax(axis=1), predictions))
print(accuracy_score(y_test.argmax(axis=1), predictions))

model.save('myfirstmodel.h5')

from keras.models import load_model

new_model = load_model('myfirstmodel.h5') """

import numpy as np

import pandas as pd

npr = pd.read_csv('C:/Users/cenb/Documents/Camilo/Proyectos/cursos online/nlp-tutorial/UPDATED-NLP-COURSE/UPDATED_NLP_COURSE/TextFiles/transport.csv')

X = npr['Keyword']
y = npr['category']

from keras.utils import to_categorical

y = to_categorical(y)

from keras.preprocessing.text import Tokenizer

# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(X)
X = t.texts_to_matrix(X, mode='count')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler_object = MinMaxScaler()

scaler_object.fit(X_train)

scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_dim=8207, activation='relu'))
model.add(Dense(8, input_dim=8207, activation='relu'))
model.add(Dense(25, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(scaled_X_train, y_train, epochs=150, verbose=2)

#model.predict(scaled_X_test)
predictions = model.predict_classes(scaled_X_test)

y_test.argmax(axis=1)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test.argmax(axis=1), predictions))
print(confusion_matrix(y_test.argmax(axis=1), predictions))
print(accuracy_score(y_test.argmax(axis=1), predictions))

model.save('myfirstmodel.h5')

from keras.models import load_model

new_model = load_model('myfirstmodel.h5')