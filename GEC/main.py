import argparse
from m2_to_txt import m2_to_txt
from data_creation import data_creation

# from transformers import T5Tokenizer,TFT5Model,TFTrainer,TFTrainingArguments
from transformers import T5Tokenizer,T5Model

import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

def padding(sentences, max_word_len):
    padding_sentences = []

    for sentence in sentences:
        word_count = sentence.count(' ')
        if word_count < max_word_len:
            for _ in range(max_word_len-word_count):
                sentence = sentence + ' <pad>'
            padding_sentences.append(sentence)

    return padding_sentences

def train_tfds(dataset,args):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = TFT5Model.from_pretrained('t5-small')

    optimizer = optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    metric = metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # train_on_batch を使う
    # for i in range(args.epoch):
    #     print("epoch : ",i+1 ,"/" ,args.epoch)
    #     for correct_batch, error_batch in iter(dataset):
    #         model.train_on_batch(x=correct_batch,y=error_batch)

    # model.save_pretrained('model')  

    # model.fitを使いたかったがエラー
    history = model.fit(dataset, epochs=3, validation_data=dataset)

    # transformers の trainer を使おうとしたがエラー
    # training_args = TFTrainingArguments(
    #     output_dir='./results',
    #     num_train_epochs=3,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=64,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir='./logs'
    # )
    # trainer = TFTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     eval_dataset=dataset
    # )
    # trainer.train()

def train_list(correct_sentences, error_sentences, args):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = TFT5Model.from_pretrained('t5-small')

    optimizer = optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    metric = metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # 1行ずつ学習しているつもり    
    # for i in range(args.epoch):
    #     print("epoch : ",i+1 ,"/" ,args.epoch)
    #     for correct_sentence, error_sentence in zip(correct_sentences, error_sentences):
    #         correct_id = tokenizer.encode(correct_sentence, return_tensors='tf')
    #         error_id = tokenizer.encode(error_sentence, return_tensors='tf')
    #         model(correct_id, decoder_input_ids=error_id)

def train_torch(correct_sentences,error_sentences,args):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Model.from_pretrained('t5-small')
    model.train()
    for i in range(args.epoch):
        print("epoch : ",i+1 ,"/" ,args.epoch)
        for correct_sentence, error_sentence in zip(correct_sentences, error_sentences):
            correct_id = tokenizer.encode(correct_sentence, return_tensors='pt')
            error_id = tokenizer.encode(error_sentence, return_tensors='pt')
            model(correct_id, decoder_input_ids=error_id)

    torch.save(model.state_dict(),'./model/weight.pth')


def main(args):
    if args.mode == "train":
        m2 = m2_to_txt(args.m2_file)
        m2.create_list_of_er_cor()
        # data = data_creation(m2.correct_sentence, m2.error_sentence, args.buffer_size,args.batch_size,args.max_length)
        # data.create_dataset()

        # # リストで学習するとき
        error_sentences = padding(m2.error_sentence, args.max_length)
        correct_sentences = padding(m2.correct_sentence, args.max_length)
        train_torch(correct_sentences, error_sentences,args)

        # # tfds で学習するとき
        # train_tfds(data.dataset, args)
    elif args.mode == "generate":
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = TFT5Model.from_pretrained('./model/')
        inputs = tokenizer.encode(args.input_sen, return_tensors="tf")
        outputs = model.generate(input_ids=inputs,decoder_start_token_id=inputs, max_length=args.max_length, num_beams=4, early_stopping=True)
        print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## m2ファイルを指定する
    parser.add_argument("--m2_file",type=str,default="./dataset/test2013.m2")
    ## パラメータに関する引数
    parser.add_argument("--buffer_size",type=int,default=20000)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--max_length",type=int,default=40)
    parser.add_argument("--epoch",type=int,default=5)
    parser.add_argument("--mode",default="train")       ## train or generate
    parser.add_argument("--input_sen",default=None)     ## モデルを試すときに入力する文を指定
    args = parser.parse_args()
    main(args)