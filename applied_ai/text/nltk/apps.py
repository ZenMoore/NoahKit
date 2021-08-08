import nltk

'''
sentiment analysis : Naive Bayesian Classifier
'''

from nltk.classify import NaiveBayesClassifier
s1= 'this is a good book'
s2= 'this is a awesome book'
s3= 'this is a bad book'
s4= 'this is a terrible book'

def preprocess(s):

    return {word:True for word in s.lower().split()}
    # {'this': True, 'is':True, 'a':True, 'good':True, 'book':True}
    # former one is 'fname', latter one is 'feval'
    # feval is auxiliary information, here True means the words do appear.
    # we can also assign other feval, for example, word2vec

training_data = [
    [preprocess(s1), 'pos'],
    [preprocess(s2), 'pos'],
    [preprocess(s3), 'neg'],
    [preprocess(s4), 'neg'],
]

model = NaiveBayesClassifier.train(training_data)
print(model.classify(preprocess('this is a good book.')))


'''
Chatbot
'''

from nltk.chat.util import Chat
from nltk.chat.eliza import eliza_chat
from nltk.chat.iesha import iesha_chat
from nltk.chat.rude import rude_chat
from nltk.chat.suntsu import suntsu_chat
from nltk.chat.zen import zen_chat

bots = [
    (eliza_chat, "Eliza (psycho-babble)"),
    (iesha_chat, "Iesha (teen anime junky)"),
    (rude_chat, "Rude (abusive bot)"),
    (suntsu_chat, "Suntsu (Chinese sayings)"),
    (zen_chat, "Zen (gems of wisdom)"),
]


def chatbots():
    import sys

    print("Which chatbot would you like to talk to?")
    botcount = len(bots)
    for i in range(botcount):
        print("  %d: %s" % (i + 1, bots[i][1]))
    while True:
        print("\nEnter a number in the range 1-%d: " % botcount, end=" ")
        choice = sys.stdin.readline().strip()
        if choice.isdigit() and (int(choice) - 1) in range(botcount):
            break
        else:
            print("   Error: bad chatbot number")

    chatbot = bots[int(choice) - 1][0]
    chatbot()

if __name__ == '__main__':
    chatbots()
