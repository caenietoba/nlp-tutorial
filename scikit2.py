import numpy as np
import pandas as pd

df = pd.read_csv('C:/Users/cenb/Documents/Camilo/Proyectos/cursos online/nlp-tutorial/UPDATED-NLP-COURSE/UPDATED_NLP_COURSE/TextFiles/moviereviews.tsv', sep='\t')

#print(df.head())

#print(df.shape)

#print(df['review'][0])

#print(df.isnull().sum())

#print(df)

df.dropna(inplace=True)

#print(df.isnull().sum())


blanks = []
for i, lb, rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)

#print(blanks)

df.drop(blanks, inplace=True)

#print(df.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))