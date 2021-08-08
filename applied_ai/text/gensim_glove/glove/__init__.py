from __future__ import print_function
import argparse
import pprint
import gensim
from glove import Glove  # todo : difficult to install glove-python on Windows
from glove import Corpus

'''
learning material : https://blog.csdn.net/sinat_26917383/article/details/83029140
api : no api, but github : https://github.com/maciejkula/glove-python
more official version : https://github.com/stanfordnlp/GloVe, but in C
'''

'dataset'
sentense = [['who','are','you'],['I','am','chinese']]  # can also be in Chinese language
corpus_model = Corpus()
corpus_model.fit(sentense, window=10)
#corpus_model.save('corpus.model')
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)

'glove model'
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

'io of model'
glove.save('glove.model')
glove = Glove.load('glove.model')

'io of corpus'
corpus_model.save('corpus.model')
corpus_model = Corpus.load('corpus.model')

'similarity'
glove.most_similar('I', number=10)

'all word vectors'
glove.word_vectors

'specific word vectors'
glove.word_vectors[glove.dictionary['You']]


'corpus matrix'
corpus_model.matrix.todense().tolist()