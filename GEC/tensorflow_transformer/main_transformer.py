import argparse
from m2_to_txt import m2_to_txt
from data_creation import data_creation
from multi_head_attention import MultiHeadAttention
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer
from encoder import Encoder
from decoder import Decoder
from transformer import Transformer
from custom_schedule import CustomSchedule
from train_eval import Train_Eval_data

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt


def translate(sentence, data,Train,max_length):
  Train.evaluate(sentence,data.tokenizer_correct,max_length)
  
  predicted_sentence = data.tokenizer_error.decode([i for i in Train.result 
                                            if i < data.tokenizer_error.vocab_size])  

  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(predicted_sentence))
  
#   if plot:
#     plot_attention_weights(attention_weights, sentence, result, plot)


def main(args):
    if args.mode == "train":
        m2 = m2_to_txt(args.m2_file)
        m2.create_list_of_er_cor()
        data = data_creation(m2.correct_sentence, m2.error_sentence, args.buffer_size,args.batch_size,args.max_length)
        data.create_dataset()

        input_vocab_size = data.tokenizer_correct.vocab_size + 2
        target_vocab_size = data.tokenizer_error.vocab_size + 2

        #train(args,input_vocab_size,target_vocab_size,data.dataset)

        Train = Train_Eval_data(args.d_model,args.num_layers,args.num_heads,args.dff,input_vocab_size,target_vocab_size,args.dropout_rate)
        Train.train(args.EPOCHS,data.dataset)

        model = Transformer(args.num_layers,args.d_model,args.num_heads,args.dff,input_vocab_size,target_vocab_size,pe_input=input_vocab_size,pe_target=target_vocab_size)
        model.load_weights("./checkpoints/train")
        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## m2ファイルを指定する
    parser.add_argument("--m2_file",type=str,default="./dataset/test2013.m2")
    ## パラメータに関する引数
    parser.add_argument("--buffer_size",type=int,default=20000)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--max_length",type=int,default=40)
    parser.add_argument("--num_layers",type=int,default=4)
    parser.add_argument("--d_model",type=int,default=128)
    parser.add_argument("--dff",type=int,default=512)
    parser.add_argument("--num_heads",type=int,default=8)
    parser.add_argument("--dropout_rate",type=float,default=0.1)

    parser.add_argument("--EPOCHS",type=int,default=20)
    parser.add_argument("--mode",default="train")       ## train or generate
    parser.add_argument("--input_sen",default=None)     ## モデルを試すときに入力する文を指定
    args = parser.parse_args()
    main(args)