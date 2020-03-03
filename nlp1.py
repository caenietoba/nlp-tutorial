import spacy

nlp = spacy.load('en_core_web_sm')

""" doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million') """

""" for token in doc:
    print(token.text, token.pos_, token.dep_)
 """

""" doc2 = nlp(u'Tesla isn\'t looking into startups anymore')

for token in doc2:
    print(token.text)

for token in doc2.sents:
    print(token) """

""" my_string = u'Autonomous cars shift insurance liability toward manufacturers.'

doc = nlp(my_string)
for token in doc:
    print(token.text, end = ' | ')
print('\n')
for entity in doc.ents:
    print(entity)
    print(str(spacy.explain(entity.label_)))
    print(entity.label_, '\n')

for chunk in doc.noun_chunks:
    print(chunk) """


""" my_string = u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.'

doc = nlp(my_string)

spacy.displacy.serve(doc, style='ent') """
# spacy.displacy.render(doc, style = 'dep', jupyter=False, options={'distance':110})

""" my_string = u'I\'m a runner running in a race because i love to run since i ran today'

doc = nlp(my_string)

for token in doc:
    print(f'{token.text:{10}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}' )"""

""" my_String = u''

print(nlp.Defaults.stop_words)

print(len(nlp.Defaults.stop_words))

print(nlp.vocab['mystery'].is_stop)

nlp.Defaults.stop_words.add('btw')
nlp.vocab['btw'].is_stop = True

print(nlp.vocab['btw'].is_stop) """

""" from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

# SolarPower
pattern1 = [{'LOWER': 'solarpower'}]
# solar.power
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
# solar power
pattern3 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]

matcher.add('SolarPower', None, pattern1, pattern2, pattern3)

my_string = 'The solar Power Industry continues to grow aa solarpower increase. Solar--power is amaizing'

doc = nlp(my_string)

found_matches = matcher(doc)

print(found_matches)

matcher.remove('SolarPower')

# solarpower
pattern_1 = [{'lower': 'solarpower'}]
# solar.power
pattern_2 = [{'lower': 'solar'}, {'IS_PUNCT': True}, {'OP':'*'}, {'LOWER':'power'}]

matcher.add('SolarPower',None, pattern_1, pattern_2)

found_matches = matcher(doc)

print(found_matches) """

""" from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)

path = 'C:/Users/cenb/Documents/Camilo/Proyectos/cursos online/nlp-tutorial/UPDATED-NLP-COURSE/UPDATED_NLP_COURSE/TextFiles/reaganomics.txt'

with open(path) as f:
    doc = nlp(f.read())

phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']

phrase_patterns = [nlp(text) for text in phrase_list]

matcher.add('EconMatcher', None, *phrase_patterns)

found_matches = matcher(doc)

print(found_matches) 

for token in doc:
    print(f'{token.text}\t{token.pos_}\t{token.tag_}\t{spacy.explain(token.tag_)}')
"""

""" my_string = u'I read books on NLP.'
my_string2 = u'I read a book on NLP.'

doc = nlp(my_string)
doc2 = nlp(my_string2)
token = doc2[1]

print(f'{token.text}\t{token.pos_}\t{token.tag_}\t{spacy.explain(token.tag_)}') """

""" my_string = u'The quick brown fox jumped ove the lazy dog\'s back.'

doc = nlp(my_string)

POS_counts = doc.count_by(spacy.attrs.POS)
TAG_counts = doc.count_by(spacy.attrs.TAG)
DEP_counts = doc.count_by(spacy.attrs.DEP)

for k,v in sorted(POS_counts.items()):
    print(f'{k}\t{doc.vocab[k].text}\t{v}')

for k,v in sorted(TAG_counts.items()):
    print(f'{k}\t{doc.vocab[k].text}\t{v}')

for k,v in sorted(DEP_counts.items()):
    print(f'{k}\t{doc.vocab[k].text}\t{v}') """

""" from spacy import displacy

my_string = u'The quick brown fox jumped ove the lazy dog\'s back.'
my_string2 = u'I read a book on NLP. This is the most longer sentence in all the world'

doc = nlp(my_string)
doc2 = nlp(my_string2)
spans = list(doc2.sents)

options = {
    'distance': 110,
    'compact': 'True',
    'color': 'yellow',
    'bg': '#09a3d5',
    'font': 'Times'
}

displacy.serve(spans, options=options) """

""" def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(f'{ent.text} - {ent.label_} - {str(spacy.explain(ent.label_))}')
    else:
        print('No entities found')

my_string = u'Tesla to build a U.K. factory for $6 million'

doc = nlp(my_string)

show_ents(doc)

from spacy.tokens import Span

ORG = doc.vocab.strings[u'ORG']

new_ent = Span(doc, 0, 1, label=ORG)

doc.ents = list(doc.ents) + [new_ent]

show_ents(doc) """

""" my_String_1 = u'Our company created a brand new vacuum cleaner.'
my_String_2 = u'This new vacuum-cleaner is the best in show.'

doc = nlp(u'Our company created a brand new vacuum cleaner.'
u'This new vacuum-cleaner is the best in show.')

print(doc.text)

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(f'{ent.text} - {ent.label_} - {str(spacy.explain(ent.label_))}')
    else:
        print('No entities found')

#show_ents(doc)

from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
phrase_patterns = [nlp(text) for text in phrase_list]

matcher.add('newproduct', None, *phrase_patterns)

found_matches = matcher(doc)

print(found_matches)

from spacy.tokens import Span

PROD = doc.vocab.strings[u'PRODUCT']

new_ents = [Span(doc, match[1], match[2], label=PROD) for match in found_matches]

doc.ents = list(doc.ents) + new_ents

show_ents(doc)

my_string_3 = u'Originally I paid $29.90 for this car toy, but now it is marked down by 10 dollars.'

doc2 = nlp(my_string_3)

ents = [ent for ent in doc2.ents if ent.label_ == 'MONEY']

print(ents) """

""" from spacy import displacy

my_String = u'''Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.
By contrast, Sony only sold 8 thousand Walkman music players'''

doc = nlp(my_String)

colors = {
    'PRODUCT': 'radial-gradient(yellow, red)'
}

options = {
    'ents': ['PRODUCT', 'ORG'],
    'colors': colors
}

displacy.serve(doc, style='ent', options=options) """

my_string = u'This is a sentence. A second sentence.\n The last\n one.'

doc = nlp(my_string)

def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc

nlp.add_pipe(set_custom_boundaries, before='parser')

doc = nlp(my_string)

for sent in doc.sents:
    print(sent)

my_string = u'This is a sentence. A second sentence.\n The last\n one.'

doc = nlp(my_string)

from spacy.pipeline import SentenceSegmenter

def split_on_newlines(doc):

    start = 0
    seen_newline = False

    for word in doc:
        if seen_newline:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text.startswith('\n'):
            seen_newline = True

    yield doc[start:] 

sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)

nlp.add_pipe(sbd)

doc = nlp(my_string)

for sent in doc.sents:
    print(sent)