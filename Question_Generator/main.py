#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

#!/usr/bin/env python3
## 使用するGPUを指定 ##
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
###################################################

import argparse
import logging
import re

import numpy as np
import torch

import sys
########################################################
## 問題生成に必要
########################################################
import nltk
from nltk.tokenize import sent_tokenize
import random
import os
import psycopg2
from nltk.stem.snowball import SnowballStemmer

#from arrange_sentence_format import arrange
from create_model import create
from send_DB_pos_word import send
from to_base_form import base_form
from Q_generate import Q_generate
#########################################################

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

## 問題分を作るところ
def sentence_generate(args):
    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    file_name = "./data/toeic_sentense_part5.txt"

    with open(file_name,'r') as fr:
        original_sentences = fr.readlines()
        pattern_big, pattern_small = decide_patern()        # 問題パターン取得 pattern_big : 大区分，pattern_small : 小区分
        while True:
            generated_sequences = []
            prompt_text = random.sample(original_sentences,1)
            prompt_text = prompt_text[0]

            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
                preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

                if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                    tokenizer_kwargs = {"add_space_before_punct_symbol": True}
                else:
                    tokenizer_kwargs = {}

                encoded_prompt = tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
                )
            else:
                encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(args.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]

                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (
                    text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                )

                generated_sequences.append(total_sequence)

            text = "".join(generated_sequences)
            sent_tokenize_list = sent_tokenize(text)

            ### 文章選別器
            q_texts = arrange(sent_tokenize_list,args.min_word,args.max_word) # 単語数と品詞で文を選別

            if len(q_texts) != 0:
                check,q_text = check_pattern(pattern_big,pattern_small,q_texts)
                if check :    # 文中に問題パターンに対応する品詞があるかチェック
                    break
        
    return q_text,pattern_big,pattern_small

# 10文字以下の足切りと改行文字・終端文字の削除 + 固有名詞だらけの文が問題分にならないようにする
def arrange(toeic_sens,low,max):
    qpos_list = ['VB','VBD','VBG','VBN','VBP','VBZ']    ## 動詞リスト
    delete_sen_list = []
    Alpha = [chr(i) for i in range(65, 65+26)]          ## アルファベットの大文字リスト(先頭を大文字の文にするため)
    for sen in toeic_sens:
        sen = sen.replace(r'<.+>',"")       
        sen = sen.replace("\n","")
        sen = sen.replace(":",",")    
        word_list = nltk.word_tokenize(sen)
        try:
            if word_list[-1] != '.' or word_list[0][0] not in Alpha :    # 文末がピリオドでない文 もしくは先頭がアルファベットの大文字でない文は削除
                delete_sen_list.append(sen)
        except IndexError:      # IndexErrorは無視
            None

    toeic_sens = list(set(toeic_sens)-set(delete_sen_list))
    toeic_sens = filter(lambda one_sentence:len(nltk.word_tokenize(one_sentence)) >= low and len(nltk.word_tokenize(one_sentence)) <= max,toeic_sens)
    toeic_sens = list(toeic_sens)

    candidate_sen = []

    if len(toeic_sens) == 0:
        return candidate_sen
    else:
        for sen in toeic_sens:
            morph = nltk.word_tokenize(sen)
            pos = nltk.pos_tag(morph)               # 品詞を見てあげる
            for i,(w,p) in enumerate(pos):
                if i == 0 and w[0] not in Alpha:    # 先頭文字が大文字でなければ変な文だからループ抜ける
                    break
                if p in qpos_list:                  # 問題になる品詞が文に一つでも含まれていれば返す
                    candidate_sen.append(sen)
        return candidate_sen

        # while True:
        #     if toeic_sens == None:    # toeic_sens の中身が無くなったら 0 を返す
        #         return 0
        #     toeic_sen = random.choice(toeic_sens)
        #     morph = nltk.word_tokenize(toeic_sen)
        #     pos = nltk.pos_tag(morph)               # 品詞を見てあげる
        #     for i,(w,p) in enumerate(pos):
        #         if i == 0 and w[0] not in Alpha:    # 先頭文字が大文字でなければ変な文だからループ抜ける
        #             break
        #         if p in qpos_list:                  # 問題になる品詞が文に一つでも含まれていれば返す
        #             return toeic_sen
        #     toeic_sens = toeic_sens.remove(toeic_sen)
        
# 問題の登場確率に沿って，作成する問題パターンを決定する
def decide_patern():
    big = random.randint(0,100)
    small = random.randint(0,100)
    if big <= 15 :      # 15% は動詞の活用問題
        if small <= 45:
            return "v_form","single"        # 動詞1語
        else:
            return "v_form","double"        # 動詞2語
    else:               # 85% は派生語問題
        if small <= 31:
            return "deri","n"               # 派生語 名詞
        elif small <= 48:
            return "deri","v"               # 派生語 動詞
        elif small <= 77:
            return "deri","adj"             # 派生語 形容詞
        else:
            return "deri","adv"             # 派生語 副詞

# 文に問題にしたい品詞が含まれているか判別
def check_pattern(big,small,sens):
    v_list = ['VB','VBD','VBG','VBN','VBP','VBZ']    ## 動詞リスト    
    n_list = ['NN','NNS']  # 名詞リスト
    adj_list = ['JJ','VBG','VBN']  # 形容詞の役割をするリスト
    adv_list = ['RB']

    for sen in sens:
        morph = nltk.word_tokenize(sen)
        pos = nltk.pos_tag(morph)
        if big == "v_form":         # 動詞の活用形問題
            if small == "single":
                for w,p in pos:
                    if p in v_list:
                        return True,sen
            else:
                # 二語以上の動詞があるかチェック
                DC = Q_generate(sen)       ## DC = Double Check
                double_flag, _ = DC.check_2more(pos)

                if double_flag == 1:
                    return True,sen
        else:                       # 派生語問題
            if small == "n":
                for w,p in pos:
                    if p in n_list:
                        if exsist_db(w):       ## 名詞がDBに含まれている場合
                            return True,sen
            elif small == "v":
                for w,p in pos:
                    if p in v_list:
                        if exsist_db(w):       ## 動詞がDBに含まれている場合
                            return True,sen
            elif small == "adj":
                for w,p in pos:
                    if p in adj_list:
                        if exsist_db(w):       ## 形容詞がDBに含まれている場合
                            return True,sen
            elif small == "adv":
                for w,p in pos:
                    if p in adv_list:
                        if exsist_db(w):       ## 副詞がDBに含まれている場合
                            return True,sen

    return False,None

# DB接続
def exsist_db(w):
    # connect postgreSQL
    users = 'torikieiji' # initial user
    dbnames = 'derivative'
    passwords = ''
    conn = psycopg2.connect(" user=" + users +" dbname=" + dbnames +" password=" + passwords)
    stemmer = SnowballStemmer("english")

    cur = conn.cursor()
    stem = stemmer.stem(w)
    stem = stem.replace("'",'"')
    cur.execute("SELECT EXISTS(SELECT * FROM deri_test WHERE original = '" + stem + "');")
    t_f = cur.fetchall()

    if t_f[0][0]:       ## 単語がDBに含まれている場合
        cur.execute("SELECT derivative FROM deri_test WHERE original = '" + stem + "';")
        deri = cur.fetchall()
        deri_list = deri[0][0].split(',')        ## 現在の派生語リストを取得
        if len(deri_list) >= 4:
            return True
        else:
            return False

#
# Functions to prepare models' input
#
def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main(args):
    if(args.format) == 'file':
        q_file = "./data/q_file.txt"
        ans_file = "./data/ans_file.txt"
        try:
            os.remove(q_file)
            os.remove(ans_file)
        except FileNotFoundError:
            None

    set_seed(args)              # 元からあった.GPU関係の設定の処理?
    for i in range(args.mondaisu):
        toeic_sen, big, small = sentence_generate(args)     # gpt2 を用いて文章を生成する
        Q_gen = Q_generate(toeic_sen)       
        Q_gen.generate(i,args.format,big,small)        # 問題を作って出力/ファイル書き出し

if __name__ == "__main__":
    ## 引数の指定
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    #parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=900)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--p", type=float, default=1)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    # 自作の引数
    parser.add_argument("--mondaisu", type=int, default=1, help="quantity of questions that can be printed.")
    parser.add_argument("--format", type=str, default="print", help="print or file")
    parser.add_argument("--q_pattern",type=str, default="rand",help="v_form or mean or rand")
    parser.add_argument("--min_word",type=int,default=10)
    parser.add_argument("--max_word",type=int,default=20)
    ## v_form ⇒ 動詞の活用問題，mean ⇒ 意味問題, デフォルト：ランダム出力
    ## min_word:最小単語数(足切りに使う), max_word:最大単語数(単語数が多すぎる文は問題として不適)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    main(args)

# 単語問題で必要になるかも
# CR = create()               # word2vecモデルに関する処理
# CR_toeic_sen = CR.delete_char_list(toeic_sen)  # 不要文字の削除：CR_toeic_sen ⇒ word2vec学習に使う
# CR.updateTrain(CR_toeic_sen) # word2vecモデルを再学習
# send(toeic_sen)