import pandas as pd

npr = pd.read_csv('C:/Users/cenb/Documents/Camilo/Proyectos/cursos online/nlp-tutorial/UPDATED-NLP-COURSE/UPDATED_NLP_COURSE/05-Topic-Modeling/npr.csv')

# print(npr.head())
#print(npr)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')

dtm = cv.fit_transform(npr['Article'])

#print(dtm)

from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=7, random_state=42)

LDA.fit(dtm)

# Voca in the documents
print(cv.get_features_names()[285:310])

# Gra the topics

len(LDA.components_)

for i, topic in LDA.components_:
    print(f'The top 15 words for topic #{i}')
    print([cv.get_features_names()[index] for index in topic.argsort()[-15:]])
    print('\n\n')

""" single_topic = LDA.components_[0]

top_ten_words = single_topic.argsort()[-15:]

for index in top_ten_words:
    print(cv.get_features_names()[index]) """

topic_results = LDA.transform(dtm)

npr['topic'] = topic_results.argmax(axis=1)

print(npr.head())