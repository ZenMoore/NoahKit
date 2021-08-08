import gensim
from gensim import corpora
from gensim import models
from gensim import similarities
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

'''
learning material : https://www.jianshu.com/p/9ac0075cc4c0
api : https://radimrehurek.com/gensim/apiref.html#api-reference
'''

path = '../../../../asset/sonnet.txt'

'preprocessing of corpora'
# sentences : document
doc = []
with open(path, 'r') as f:
    for line in f.readlines():
        doc.append(line.split())
print(doc)

# vocabulary
vocab = corpora.Dictionary(doc)
print(vocab)  # pay attention to his arrangement style

# bag of words
corpus = [vocab.doc2bow(sent) for sent in doc]
print(corpus)

# or use iterator
# def iter():
#     for line in open(path):
#         yield vocab.doc2bow(line.lower().split())

'vector transformation'
# from bow, we have an iterable vector named corpus
tfidf = models.TfidfModel(corpus)  # which returns an iterator
print(tfidf)
text_bow = [(0, 1), (1, 1)]  # must in bow sparse vector form
print(tfidf[text_bow])
# other : lda, lsi, rp, hdp

'io'
tfidf.save('model.tfidf')
tfidf = models.TfidfModel.load('model.tfidf')


'similarity'
lsi = models.LsiModel(corpus=corpus, id2word=vocab, num_topics=2)
texts_lsi = lsi[corpus]
query = 'the beauty will die'
query_bow = vocab.doc2bow(query)
query_lsi = lsi[query_bow]

bench = similarities.MatrixSimilarity(texts_lsi)

bench.save('sim.bench')
bench = similarities.MatrixSimilarity.load('sim.bench')

sims = bench[query_lsi]  # similarity matrix between query_lsi and every texts_lsi

'glove2word2vec'
glove_file = datapath('path-to-glove.txt')
tmp_file = get_tmpfile('path-to-output-word2vec.txt')

glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)