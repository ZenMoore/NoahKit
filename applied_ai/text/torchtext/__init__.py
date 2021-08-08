import torchtext
from sklearn.model_selection import train_test_split
import pandas as pd
from torchtext.legacy import datasets, data
import torch
from torch.nn import init
from nltk import word_tokenize

'''
learning material : https://blog.csdn.net/JWoswin/article/details/92821752
api : https://pytorch.org/text/stable/index.html
'''

# pipeline:
# 1. Field : preprocessing
# 2. Datasets : samples are preprocessed
# 3. Vocab : vocab, word2idx, idx2word
# 4. Iterator : training batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'datasets prepare'
train_set = pd.read_csv('../../../asset/movie_train.tsv', sep='\t')
test_set = pd.read_csv('../../../asset/movie_test.tsv', sep='\t')
train_set, val_set = train_test_split(train_set, train_size=0.7)
train_set.to_csv("train.csv", index=False)
val_set.to_csv("val.csv", index=False)
test_set.to_csv('test.csv', index=False)

'Field'
# Field announces how you want to process the data
LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=word_tokenize, lower=True)

'Datasets'
# TabularDataset : Defines a Dataset of columns stored in CSV, TSV, or JSON format.
train, val = data.TabularDataset.splits(
        path='', train='train.csv',validation='val.csv', format='csv',skip_header=True,
        fields=[('PhraseId',None),('SentenceId',None),('Phrase', TEXT), ('Sentiment', LABEL)])

test = data.TabularDataset('test.csv', format='csv', skip_header=True,
                           fields=[('PhraseId',None),('SentenceId',None),('Phrase', TEXT)])
print(train[5])
print(train[5].__dict__.keys())
print(train[5].Phrase, train[5].Sentiment)

'Vocab'
# todo : too large glove
TEXT.build_vocab(train, vectors='glove.6B.100d')  # max_size= 30000, 100d='dim=100'
# when some token not appeared in the vector, we need initialization
TEXT.vocab.vectors.unk_init = init.xavier_uniform
# automatically download glove.6B.100d to /.vector_cache
# word vector supported by torchtext :
# charngram, fasttext, glove
# if we need word2vec, use gensim : glove2word2vec
# if we need ELMo, use bi-LSTM of pytorch
# if we need BERT/RoBERTa/etc., use hugging face
print(TEXT.vocab.itos[1510])
print(TEXT.vocab.stoi['bore'])
print(TEXT.vocab.vectors.shape)
print(TEXT.vocab.vectors[TEXT.vocab.stoi['bore']].shape)

'Iterator'
# BucketIterator : Defines an iterator that batches examples of similar lengths together.
#       Minimizes amount of padding needed while producing freshly shuffled batches for each new epoch.
train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.Phrase), shuffle=True, device=DEVICE)
val_iter = data.BucketIterator(val, batch_size=128, sort_key=lambda x: len(x.Phrase), shuffle=True, device=DEVICE)
test_iter = data.Iterator(test, batch_size=128, sort=False, shuffle=False, train=False, device=DEVICE)

# method 1
batch = next(iter(train_iter))  # must be in iter(), suck.
data = batch.Phrase
label = batch.Sentiment
print(data.shape)  # batch_size is in dimension 2 ! not first ! suck.

# method 2
# for batch in train_iter:
#         data = batch.Phrase
#         label = batch.Sentiment








