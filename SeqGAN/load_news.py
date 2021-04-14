import csv
csv.field_size_limit(1000000000)
from nltk.tokenize import sent_tokenize
import argparse
import sys

def load_news(l,h): 
    news_data = []

    with open("./data/articles1.csv", "r") as rf:
        data = csv.reader(rf)
        flag = 0
        for i in data:  # 各行がリストになっている
            if flag == 0:
                flag = 1
            else:
                news_data.append(i[9])

    with open("./data/articles1.txt","w") as wf:
        for one_sentence in news_data:
            for line in sent_tokenize(one_sentence):
                if len(line.split(' ')) >= l and len(line.split(' ')) <= h:
                    wf.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('lower_threshold',type=int, help='lower_threshold')
    parser.add_argument('upper_threshold',type=int, help='upper_threshold')
    args = parser.parse_args()
    
    load_news(args.lower_threshold,args.upper_threshold)
