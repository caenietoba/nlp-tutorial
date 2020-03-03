import nltk

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer(language='english')

# words = ['run', 'runner', 'ran', 'runs', 'easily', 'fearly']
words = ['generous', 'generation', 'generously', 'generate']

print(f'Porter')
for word in words:
    print(f'{word} ----> {p_stemmer.stem(word)}')

print(f'\nSnowball')
for word in words:
    print(f'{word} ----> {s_stemmer.stem(word)}')

