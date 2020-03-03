import nltk

#nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

my_string = 'This was the worst movie i have ever seen'

print(sid.polarity_scores(my_string))

import pandas as pd

df = pd.read_csv('C:/Users/cenb/Documents/Camilo/Proyectos/cursos online/nlp-tutorial/UPDATED-NLP-COURSE/UPDATED_NLP_COURSE/TextFiles/amazonreviews.tsv', sep='\t')

#print(df['label'].value_counts())

df.dropna(inplace=True)
blanks = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)
df.drop(blanks,inplace=True)

print(sid.polarity_scores(df.iloc[0]['review']))

df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda d: d['compound'])
df['comp_score'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')

print(df.head())

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(df['label'], df['comp_score']))
print(classification_report(df['label'], df['comp_score']))
print(confusion_matrix(df['label'], df['comp_score']))

