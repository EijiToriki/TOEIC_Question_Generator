import json
import re
import argparse
import sys

def convert_reviewjson_to_dataset(review_json, review_text, threshold):
    pats = [('\s*\.\s*', ' . \n'), ('\s*\!\s*', ' ! \n'), ('\s*\?\s*', ' ? \n'), ('\s*,\s*', ' , ')]
    with open(review_json,'r') as rf:
        rbytes = 0
        count = 0
        with open(review_text,'w') as wf:
            for line in rf:
                rbytes += len(line)
                count += 1
                if count % 10000 == 0:
                    sys.stdout.write('\rRead bytes : {}'.format(rbytes))
                    sys.stdout.flush()
                datum = json.loads(line)
                text = datum["text"]
                for pat, repl in pats:
                    text = re.sub(pat, repl, text)
                for intext in re.split('\n', text):
                    if len(intext.split(' ')) > threshold:
                        wf.write(intext+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('review_json',type=str, help='review_json')
    parser.add_argument('review_text',type=str, help='review_text')
    parser.add_argument('threshold',type=int, help='threshold')
    args = parser.parse_args()
    
    convert_reviewjson_to_dataset(args.review_json, args.review_text, args.threshold)
