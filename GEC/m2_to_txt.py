import argparse
import re
import nltk

class m2_to_txt(object):
    def __init__(self,m2_file):
        self.m2_file = m2_file
        self.correct_sentence = []   ## 正しい英文を入れておくリスト
        self.error_sentence = []     ## 誤っている英文を入れておくリスト
        
    def create_list_of_er_cor(self):
        words = []              ## 誤り文を正しい文へ変換するための作業用リスト
        error_flag = 0          ## DELETEがないことによる IndexError を制御
        with open(self.m2_file,'r') as fr:
            for text in fr:
                if(re.match(r'^S',text)):   ## 先頭文字が S ならば
                    if len(words) != 0:
                        if error_flag == 1:
                            while "DELETE" in words:
                                words.remove("DELETE")      ## DELETEタグを全て消す
                        self.correct_sentence.append(' '.join(words))
                        error_flag = 0
                    words = nltk.word_tokenize(text)    ## 単語ごとに分割
                    words.pop(0)            ## 先頭の S を削除
                    self.error_sentence.append(' '.join(words))
                if(re.match(r'^A',text)):   ## 先頭文字が A ならば
                    span,type,sub,req,none,annotator = text.split('|||')
                    span = span.replace('A ','')
                    start,end = span.split()
                    
                    for i in range(int(start),int(end)):
                        if sub and i==int(start):
                            words[i] = sub ## 要素の変換
                        else:
                            words[i] = "DELETE"    ## 要素の削除ラベル
                            error_flag = 1          ## DELETEタグを消すためにフラグを立てる
                    if (int(start)==int(end)):
                        if int(start) < len(words): 
                            words[int(start)] = sub + ' ' + words[int(start)]     ## 要素の挿入
                        else:   ## 単語数とstartの値が同じ時がある
                            words.append(sub)

        while "DELETE" in words:
            words.remove("DELETE")      ## DELETEタグを全て消す
        self.correct_sentence.append(' '.join(words))    ## 正しい英文は A タグが終わってから出来上がるのでループ後に一回実行する必要あり