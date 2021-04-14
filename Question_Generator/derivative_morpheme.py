import argparse
import nltk
import enchant
import json
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from nltk.stem.snowball import SnowballStemmer

def remove_duplicate(lst):
    ret_lst = []
    for d in lst:
        if d not in ret_lst:
            ret_lst.append(d)
    return ret_lst

stemmer = SnowballStemmer("english")

derivative_list = ["JJ","JJR","JJS","NN","NNS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]      # 派生語になる可能性がある品詞リスト
derivative_dict = {}

parser = argparse.ArgumentParser()
parser.add_argument("--txt_path",type=str,default="./data/text/")
parser.add_argument("--json_path",type=str,default="./data/json/")
args = parser.parse_args()

checker = enchant.Dict("en_US")

for f in glob.glob(args.txt_path + '*.txt'):
    file = os.path.split(f)[1]
    print(file)     # 進行状況の確認用
    with open(args.txt_path + file,'r') as fr:
        for line in fr:
            morph = nltk.word_tokenize(line)
            pos = nltk.pos_tag(morph)
            for w,p in pos:
                w = w.lower()
                if p in derivative_list and "'" not in w:
                    derivative_dict.setdefault(stemmer.stem(w),[]).append(w)

    derivative_dict = {k:v for k,v in derivative_dict.items() if len(v) > 1}        # 2単語以上の派生語リストを取っておく
    derivative_dict = {k:remove_duplicate(v) for k,v in derivative_dict.items()}    # リストの被り要素を消す
    derivative_dict = {k:v for k,v in derivative_dict.items() if len(v) > 1}

    copy_dd = derivative_dict

    for k,v in derivative_dict.items():
        if k in copy_dd[k]:                         ## 派生語リストとキーに同じ要素があったら
            copy_dd[k].remove(k)                    ## 消す
        if checker.check(k):        ## キーが既知語なら
            copy_dd[k].append(k)    ## 辞書にキーを追加

    derivative_dict = copy_dd

    # print(derivative_dict)

    fw = open(args.json_path + file +'.json','w')
    json.dump(derivative_dict,fw,indent=4)