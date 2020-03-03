import pandas as pd

npr = pd.read_csv('C:/Users/cenb/Documents/Camilo/Proyectos/cursos online/nlp-tutorial/UPDATED-NLP-COURSE/UPDATED_NLP_COURSE/05-Topic-Modeling/npr.csv')

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = tfidf.fit_transform(npr['Article'])

from sklearn.decomposition import NMF

nmf_model = NMF(n_components=7, random_state=42)

nmf_model.fit(dtm)

for index, topic in enumerate(nmf_model.components_):
    print(f'The topic 15 words in topic #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n\n')

topic_results = nmf_model.transform(dtm)

npr['topic'] = topic_results.argmax(axis=1)

print(npr.head())