# coding:shift-JIS
import os
import cPickle
import argparse

from create_dict_kai import Vocab

def create_dict(input_data):
    id_input_data = './save/id_text.txt'
    
    
    vocab = Vocab(input_data)
    vocab_size = vocab.vocab_num
    vocab.write_word2id(input_data, id_input_data)
    
    cPickle.dump(vocab.id2word, open('save/news.pkl', 'w'))
    print(vocab_size)
 
def id_encode(input_data):
    vocab_file = "news.pkl"
    output_file = 'speech/test.txt'
    word = cPickle.load(open('save/'+vocab_file))

    with open(output_file, 'w')as fout:
        with open(input_data)as fin:
            for line in fin:
                line = line.split()
                line = [int(x) for x in line]
                line = [word[x] for x in line]
                line = ' '.join(line) + '\n'
                fout.write(line)#.encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('command',type=int, help='1:make dict , 2:transfer id->word')
    parser.add_argument('in_filename', help='if you choose command 1,you must write an input file-name')
    args = parser.parse_args()
    
    if args.command == 1:
        create_dict(args.in_filename)
    elif args.command == 2:
        id_encode(args.in_filename)
    else:
        print("error:check your command number")
