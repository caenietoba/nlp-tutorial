import spacy

nlp = spacy.load('en_core_web_lg')

#print(nlp(u'LTL').vector)

""" tokens = nlp(u'like love hate')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2)) """

""" tokens = nlp(u'doc cat ltl tl drayage truckload')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov) """

from scipy import spatial

cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2) 

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

new_vector = king - man + woman

computed_similarities = []

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([t[0].text for t in computed_similarities[:10]])

