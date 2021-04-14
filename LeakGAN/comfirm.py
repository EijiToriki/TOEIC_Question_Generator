# coding:shift-JIS
import cPickle

vocab_file = "news.pkl"
word = cPickle.load(open('save/'+vocab_file))

print(word[0])