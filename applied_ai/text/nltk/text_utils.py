import nltk
import jieba # from chinese processing, we should directly use ltp/jieba/etc., or use Spacy(Chinese), or graft the Stanford NLP(Chinese)

sentence = 'In all external grace, you have some part.'  # from Sonnet by Shakespeare
sentences = ['Lo, in the orient when the gracious light.', 'Lifts up his burning head, each under eye.',
             'Doth homage to his new-appearing sight.', 'Serving with looks his sacred majesty.'] # from Sonnet by Shakespeare
paragraph = '''
It would be fatal for the nation to overlook the urgency of the moment. This sweltering summer of the
 Negro's legitimate discontent will not pass until there is an invigorating autumn of freedom and equality.
  Nineteen sixty-three is not an end, but a beginning. And those who hope that the Negro needed to blow off
   steam and will now be content will have a rude awakening if the nation returns to business as usual. And
    there will be neither rest nor tranquility in America until the Negro is granted his citizenship
     rights. The whirlwinds of revolt will continue to shake the foundations of our nation until the bright
      day of justice emerges.
'''  # from I Have A Dream by Martin Luther King
chinese_sent = '文官之间的冲突，虽然起源于抽象的原则，但也不能减轻情绪的冲突。'  # from 1587, A Year of No Significance by Renyu Huang

'download packages'
nltk.download('punkt') # we should download this package for tokenization in advance.
nltk.download('wordnet') # we should download this package for lemmatization in advance.
# todo : difficult to download
nltk.download('averaged_perceptron_tagger')  # we should download this package for pos in advance
nltk.download('stopwords') # we should download this package for stopwords in advance
nltk.download('state_union')  # we should download this package for trained sentence splitting in advance


'sentence splitting (simple)'
print(nltk.sent_tokenize(paragraph))

'sentence splitting (trained)'
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw("2006-GWBush.txt")
tokenizer = PunktSentenceTokenizer(train_text)
print(tokenizer.tokenize(sample_text))

'phrase'
from nltk.tokenize import MWETokenizer
tokenizer = MWETokenizer([('a', 'little'), ('a', 'lot')], separator='_')
tokenizer.add_mwe(('in', 'spite', 'of'))
print(tokenizer.tokenize(nltk.word_tokenize('a little or a lot or in spite of')))

'tokenize (not Chinese)'
print(nltk.word_tokenize(sentence)) # different from ltp, here the param is a string, not an array

'tokenize (Chinese, jieba)'
seg_list = jieba.cut(chinese_sent, cut_all=True)
print("全模式:","/ ".join(seg_list)) # 全模式
seg_list= jieba.cut(chinese_sent, cut_all=False)
print("精确模式:","/ ".join(seg_list)) # 精确模式, default
seg_list= jieba.cut_for_search(chinese_sent)
print('搜索引擎模式：',','.join(seg_list))

'tokenize (social network language)'
# we can use regex. see more on https://www.cnblogs.com/zhangyafei/p/10618585.html.

'stemming'
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
print(porter_stemmer.stem('selecting'))
print(porter_stemmer.stem('presumably'))
print(porter_stemmer.stem('provision'))

'lemmatizer'
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize('dogs'))
print(wordnet_lemmatizer.lemmatize('corpora'))
print(wordnet_lemmatizer.lemmatize('coming'))
print(wordnet_lemmatizer.lemmatize('insatiably'))
print(wordnet_lemmatizer.lemmatize('are'))
print(wordnet_lemmatizer.lemmatize('is')) # if there is not pos tag, it will be treated as original form.

'part of speech (automatic)'
words = nltk.word_tokenize(sentence)
print(nltk.pos_tag(words))

'part of speech (handmade)'
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize('are', pos='v'))

'named entity recognition'
tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
tree = nltk.ne_chunk(tagged) # ner tree, but print(tree) is a linear one.
print(tree)

'chunk'
tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
# <RB.?>*：零个或多个任何时态的副词，后面是：
# <VB.?>*：零个或多个任何时态的动词，后面是：
# <NNP>+：一个或多个合理的名词，后面是：
# <NN>?：零个或一个名词单数。
chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
chunkParser = nltk.RegexpParser(chunkGram)
chunked = chunkParser.parse(tagged)
chunked.draw()

'stopwords'
from nltk.corpus import stopwords
words = nltk.word_tokenize(sentence)
filtered_words = [word for word in words if word not in stopwords.words('english')]
print(filtered_words)

'frequency'
from nltk import FreqDist
words = nltk.word_tokenize(sentence)
fdist = FreqDist(words)
print(fdist.most_common(50))
print(fdist.items())
for k, v in fdist.items():
    print(k, v)

'tf-idf'
from nltk.text import TextCollection
sents = [nltk.word_tokenize(sent) for sent in sentences]
corpus = TextCollection(sents)

print(corpus.idf('orient'))
print(corpus.tf(term='orient', text='in the orient when the gracious light.'))
print(corpus.tf_idf(term='orient', text='in the orient when the gracious light.'))

