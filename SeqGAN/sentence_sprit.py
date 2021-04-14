from nltk import tokenize
import collections
import json
import sys

def convert_word(sentence):
    return [word2id[word.lower()] for word in sentence.split()]

if __name__ == '__main__':
    r_name = "./data/review.json"
    w_name = "./data/review.txt"
    lower_threshold = 15
    upper_threshold = 25
    word2id = collections.defaultdict(lambda: len(word2id) )
    word2id['<PAD>'] 
    word2id['<S>']
    word2id['</S>']
    word2id['<UNK>']

    with open(r_name,'r',encoding="utf-8_sig") as rf:
        rbytes = 0
        count = 0
        #書き込みファイルオープン
        with open(w_name,'w',encoding="utf-8_sig") as wf:
            #jsonファイルの一要素ずつのループ
            for line in rf:
                rbytes += len(line)
                count += 1
                #1万行書き込むごとに何バイト読み込んだか表示
                if count % 100 == 0:
                    sys.stdout.write('\rRead bytes : {}'.format(rbytes))    #\rは上書き
                    sys.stdout.flush()  #これを書かないとfor終わってから.writeの内容が書かれるらしい
                    break
                #データセットのtextのところだけ持ってきてtextに代入
                datum = json.loads(line)
                text = datum["text"]
                text_list = tokenize.sent_tokenize(text)
                #print(text_list)
                for one_sentence in text_list:
                    if len(one_sentence.split(' ')) >= lower_threshold and len(one_sentence.split(' ')) <= upper_threshold:
                        if len(one_sentence.split('\n')) == 1:
                            wf.write(''.join(str(convert_word(one_sentence))))
    
    print(len(word2id))
