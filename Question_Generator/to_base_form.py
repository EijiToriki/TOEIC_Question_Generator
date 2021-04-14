from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib import request
import ssl
import mlconjug3
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
ssl._create_default_https_context = ssl._create_unverified_context
#nltk.download('wordnet')

class base_form(object):

    # 不規則動詞の辞書を作る関数
    def __init__(self):
        self.iv = [] # iv = irregular verbs
        self.response = request.urlopen("https://www.englishpage.com/irregularverbs/irregularverbs.html")
        self.soup = BeautifulSoup(self.response)
        self.response.close()

        self.table = self.soup.find_all('tr')

        
        for row in self.table:
            tmp = []
            tmp2 = []
            for item in row.find_all('td'):
                kind = 0
                p1 = ''
                p2 = ''
                word = item.text
                # 余計な文字を消す
                index = word.find('(')
                if index != -1:
                    word = word[:(index-1)]
                index = word.find('REGULAR')
                if index != -1:
                    word = word[:(index-2)]
                index = word.find('[')
                if index != -1:
                    word = word[:(index-1)]
                

                # 過去形/過去分詞が二つある場合
                index = word.find('/')
                if index != -1:
                    p1 = word[:index-1]
                    p2 = word[index+1:]
                    tmp.append(p1)
                    tmp2.append(p2)
                    kind = 1
                else:
                    tmp.append(word)
                    tmp2.append(word)

            if len(tmp) == 3:
                if kind == 0:
                    self.iv.append(tmp)
                else:
                    self.iv.append(tmp)
                    self.iv.append(tmp2)


    # 動詞の原形に戻す関数
    def to_base(self,verb,pos):
        base = ''
        is_base = 0

        if pos == 'VBD':
            for row in self.iv:
                if row[1] == verb:
                    base = row[0]
                    is_base = 1
                    break
        elif pos == 'VBN':
            for row in self.iv:
                if row[2] == verb:
                    base = row[0]
                    is_base = 1
                    break

        lemmatizer = WordNetLemmatizer()

        if is_base == 0:
            base = lemmatizer.lemmatize(verb,pos='v')

        return base