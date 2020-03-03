import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/cenb/Documents/Camilo/Proyectos/cursos online/nlp-tutorial/UPDATED-NLP-COURSE/UPDATED_NLP_COURSE/TextFiles/smsspamcollection.tsv', sep='\t')

# print(df.isnull().sum())
# print(df['label'].value_counts())

from sklearn.model_selection import train_test_split

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

count_vect.fit(X_train)
X_train_counts = count_vect.transform(X_train) 

X_train_counts = count_vect.fit_transform(X_train)

print(X_train_counts.shape)

""" from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

count_vect.fit(X_train)
X_train_counts = count_vect.transform(X_train) 

X_train_counts = count_vect.fit_transform(X_train)

print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

print(X_train_tfidf.shape)

from sklearn.svm import LinearSVC

clf = LinearSVC() """

""" print(X_train_tfidf.shape, y_train.shape)

clf.fit(X_train_tfidf, y_train) """

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

print(accuracy_score(y_test, predictions))

pred = text_clf.predict(["Hi, how are you doing today?"])
print(pred)