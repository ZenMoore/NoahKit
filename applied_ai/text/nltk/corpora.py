import nltk
'''
how to install corpora
'''

'method 1 : download selected models and corpora'
nltk.download()

'method 2 : we can also download them from official website and import them'
from nltk import data
data.path.append('xxx/nltk_data')

'method 3 : but it is sufficient and efficient to download one specific corpus'
nltk.download('brown') # download has been finished, we annotate this line from now on.

'Then, have a look at the corpus "brown" '
from nltk.corpus import brown # before importation, we should download it in advance.

# basic information
print(brown.categories())
print(brown.readme())

# vocabulary
print(len(brown.words()))
print(brown.words()[:10])

# sentences
print(len(brown.sents()))
print(brown.sents()[:10])

# tags
print(len(brown.tagged_words()))
print(brown.tagged_words()[:10])

