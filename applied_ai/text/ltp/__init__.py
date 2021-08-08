from ltp import LTP

'''
learning material : http://ltp.ai/docs/introduction.html
api : http://ltp.ai/docs/introduction.html
'''

# the sentence is from "Snow Country" (Kawabata Yasunari).

'load ltp model'
ltp = LTP(path='small')

'sentence split'
print(ltp.sent_split(['银平张开右掌挥了挥。这是他边走边激励自己的习惯。']))  # attention : sent must be in an array

'define dictionary'
ltp.init_dict(path='../../../asset/vocab.txt', max_window=4)
print(ltp.trie)  # trie is the intrinsic dictionary

'add words to dictionary'
ltp.add_words(words=['张牙舞爪'], max_window=4)
print(ltp.trie)  # trie is the intrinsic dictionary

'word segmentation'
segment, hidden = ltp.seg(['银平张开右掌挥了挥'])
print(segment)  # segment = ['银平', '张开', '右掌', '挥', '了', '挥']
print(hidden)  # todo dict={'word_cls', 'word_length', 'word_cls_input', 'word_cls_mask'}
segment, hidden = ltp.seg(['银平/张开/右掌/挥了挥'.split('/')], is_preseged=True)

'part of speech'
segment, hidden = ltp.seg(['银平张开右掌挥了挥'])
pos = ltp.pos(hidden)
print(segment)
print(pos)

'named entity recognition'
segment, hidden = ltp.seg(['银平张开右掌挥了挥'])
ner = ltp.ner(hidden)
print(segment)
print(ner)
tag, start, end = ner[0][0]
print(tag + ' : ' + ''.join(segment[0][start: end + 1]))

'semantic role labelling'
segment, hidden = ltp.seg(['他叫银平去拿外衣'])
srl = ltp.srl(hidden)
print(segment)
print(srl)
srl = ltp.srl(hidden, keep_empty=False)
print(segment)
print(srl)

'dependency syntactic parsing'
# the root node is of index 0, hence the beginning index of nodes is 1
segment, hidden = ltp.seg(['他叫银平去拿外衣'])
dep = ltp.dep(hidden)
print(dep)

'semantic dependency parsing'
# beginning index is 1
segment, hidden = ltp.seg(['他叫银平去拿外衣'])
print(ltp.sdp(hidden, graph=False))  # tree
print(ltp.sdp(hidden, graph=True))  # graph

'LTP Server'
# todo
# pip install ltp, tornado
# python utils/server.py serve
