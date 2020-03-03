import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/cenb/Documents/Camilo/Proyectos/cursos online/nlp-tutorial/UPDATED-NLP-COURSE/UPDATED_NLP_COURSE/TextFiles/moviereviews.tsv', sep='\t')

#print(df['label'].value_counts())

df.dropna(inplace=True)
blanks = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)
df.drop(blanks,inplace=True)

import nltk

#nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda d: d['compound'])
df['comp_scores'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(accuracy_score(df['label'], df['comp_scores']))
print(classification_report(df['label'], df['comp_scores']))
print(confusion_matrix(df['label'], df['comp_scores']))
