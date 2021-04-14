#word2vecのモデルを作成するファイル
from gensim.models import word2vec
import nltk

class create(object):
    def __init__(self):
        self.first_file = "./data/for_word2vec.txt"
        self.model_name = "./data/toeic.model"

    def CM(self): 
        self.sentences=[ x.rstrip("\r\n").split() for x in list(open(self.first_file,errors='ignore')) ]
        self.model = word2vec.Word2Vec(self.sentences,  sg=1, size=100, window=5, min_count=1)
        model.save(self.model_name)

    # delete_char_* はいるのか微妙．
    # ピリオドとかをわざわざ消して学習することに意味があるのか．
    def delete_char_file(self,file_name):      # ピリオドとか消す
        self.word_list = []
        self.sentence_list = []
        with open(file_name,"r") as f:
            for sentence in f:
                self.word_list = nltk.word_tokenize(sentence)
                if self.word_list[-1] == '.' or self.word_list[-1] == '?' or self.word_list[-1] == '!':
                    del self.word_list[-1]
                out_str = ''
                for word in self.word_list:
                    if word == self.word_list[0]:
                        out_str = word
                    else:
                        out_str = out_str + ' ' + word
                self.sentence_list.append(out_str+'\n')

        with open("a.txt","w") as fw:
            for sentence in self.sentence_list:
                fw.write(sentence)
    
    def delete_char_list(self,toeic_sen):      # ピリオドとか消す.
        self.word_list = []
        self.sentence_list = []
        for sentence in toeic_sen:
            self.word_list = nltk.word_tokenize(sentence)
            if self.word_list[-1] == '.' or self.word_list[-1] == '?' or self.word_list[-1] == '!':
                del self.word_list[-1]
            out_str = ''
            for word in self.word_list:
                if word == self.word_list[0]:
                    out_str = word
                else:
                    out_str = out_str + ' ' + word
            self.sentence_list.append(out_str+'\n')

        return self.sentence_list

    def updateTrain(self,toeic_sen):
        # 既存モデルの読み出し
        model = word2vec.Word2Vec.load(self.model_name)

        # 追加の学習
        sentences=[ x.rstrip("\r\n").split() for x in toeic_sen ]
        model.build_vocab(sentences, update=True)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(self.model_name)