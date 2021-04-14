import os
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

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import F_05

## テストファイルから誤り分と正しい文のペアを抜き取るメソッド
def deal_testfile(file):    
  with open(file,'r') as fr:
    for line in fr:
      if line != "\n":
        line_list = line.split("\t")
        err_sens.append(line_list[4].replace('\n',''))
        cor_sens.append(line_list[-1].replace('\n',''))

## PIEのテキストファイルからリストを作成
def deal_pie(cor_file,err_file):
  with open(err_file,mode='r') as fe,open(cor_file,mode='r') as fc:
    for err_sen,cor_sen in zip(fe,fc):
      err_sens.append(err_sen)
      cor_sens.append(cor_sen)

## 推論時 トークナイザーを ロードする
def load_tokenizer():
    cor_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("./tokenizer/correct")
    err_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("./tokenizer/error")

    return cor_tokenizer, err_tokenizer  

##################################################################################################
## パラメータの設定(このプログラムではグローバル変数で行う)
##################################################################################################
parser = argparse.ArgumentParser()
## PIE-syntheticのテキストファイルを指定する
parser.add_argument("--pie_file",type=str,default="./dataset/synthetic/")
parser.add_argument("--pie_count",type=int,default=5) # textファイルを何個使うか(メモリ量的に)
## testファイルを指定する
parser.add_argument("--test_file",type=str,default="./dataset/lang-8-en-1.0/lang-8-en-1.0/entries.test")
## m2ファイルを指定する
parser.add_argument("--m2_file",type=str,default="./dataset/test2013.m2")
## checkpointのパス
parser.add_argument("--check_path",type=str,default="./checkpoints_global/train")
## パラメータに関する引数
parser.add_argument("--EPOCHS",type=int,default=20)
parser.add_argument("--buffer_size",type=int,default=20000)
parser.add_argument("--batch_size",type=int,default=16)
parser.add_argument("--max_length",type=int,default=40)

parser.add_argument("--mode",default="train")       ## train or generate or F05:f0.5値の計算
parser.add_argument("--input_sen",default=None)     ## モデルを試すときに入力する文を指定
args = parser.parse_args()

mode = args.mode
EPOCHS = args.EPOCHS

cor_tokenizer = 0
err_tokenizer = 0
input_vocab_size = 0
target_vocab_size = 0

if mode == "train":
  err_sens = [] ## 英語学習者の文章
  cor_sens = [] ## 訂正文章

  ## m2ファイルで学習していたときのコード(データ量少なすぎてゴミ)
  # m2 = m2_to_txt(args.m2_file)
  # m2.create_list_of_er_cor()

  ## PIEファイルで学習する
  ### synthetic内のファイル名を獲得
  # path = args.pie_file
  # files = os.listdir(path)
  # file = [f for f in files if os.path.isfile(os.path.join(path,f))]
  # file.sort()

  # for i in range(args.pie_count):
  #   cor_file = args.pie_file + file[i*2]
  #   err_file = args.pie_file + file[i*2+1]
  #   print(cor_file,err_file)
  #   deal_pie(cor_file,err_file)

  # lang8のtestファイルで学習するコード
  deal_testfile(args.test_file)

  ## データセットを作成(tfds)
  data = data_creation(cor_sens, err_sens, args.buffer_size,args.batch_size,args.max_length)
  data.create_tokenizer()
  data.create_dataset()
  input_vocab_size = data.tokenizer_correct.vocab_size + 2
  target_vocab_size = data.tokenizer_error.vocab_size + 2
  print(input_vocab_size , target_vocab_size)
elif mode == "generate":
  cor_tokenizer, err_tokenizer = load_tokenizer()
  input_vocab_size = cor_tokenizer.vocab_size + 2
  target_vocab_size = err_tokenizer.vocab_size + 2
  print(input_vocab_size , target_vocab_size)

# ハイパラメータ
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')    

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

## チェックポイント関係
checkpoint_path = args.check_path
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# チェックポイントが存在したなら、最後のチェックポイントを復元
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

########################################################################
## 学習に関するメソッド
########################################################################
def train():
  for epoch in range(EPOCHS):
    start = time.time()
      
    train_loss.reset_states()
    train_accuracy.reset_states()
      
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(data.dataset):
      train_step(inp, tar)
        
      if batch % 50 == 0:
        print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, batch, train_loss.result(), train_accuracy.result()))
          
    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))
        
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
    train_loss(loss)
    train_accuracy(tar_real, predictions)

############################################################################################
## 正しい⇒誤りを試す
############################################################################################
def evaluate(inp_sentence):
  start_token = [cor_tokenizer.vocab_size]
  end_token = [cor_tokenizer.vocab_size + 1]
  
  # inp文は正しい英語、開始および終了トークンを追加
  inp_sentence = start_token + cor_tokenizer.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # ターゲットは誤った英語であるため、Transformerに与える最初の単語は英語の
  # 開始トークンとなる
  decoder_input = [err_tokenizer.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(args.max_length):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # seq_len次元から最後の単語を選択
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # predicted_idが終了トークンと等しいなら結果を返す
    if predicted_id == err_tokenizer.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    # 出力にpredicted_idを結合し、デコーダーへの入力とする
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))
  
  sentence = data.tokenizer_correct.encode(sentence)
  
  attention = tf.squeeze(attention[layer], axis=0)
  
  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # アテンションの重みをプロット
    ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))
    
    ax.set_ylim(len(result)-1.5, -0.5)
        
    ax.set_xticklabels(
        ['<start>']+[data.tokenizer_correct.decode([i]) for i in sentence]+['<end>'], 
        fontdict=fontdict, rotation=90)
    
    ax.set_yticklabels([data.tokenizer_error.decode([i]) for i in result 
                        if i < data.tokenizer_error.vocab_size], 
                       fontdict=fontdict)
    
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show()

def translate(sentence, plot=''):
  result, attention_weights = evaluate(sentence)
  
  predicted_sentence = err_tokenizer.decode([i for i in result 
                                            if i < err_tokenizer.vocab_size])  

  print('変換後: {}'.format(predicted_sentence))
  
  if plot:
    plot_attention_weights(attention_weights, sentence, result, plot)



def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

#####################################################################################
## F0.5値での評価
#####################################################################################
def cal_F05():
  m2 = m2_to_txt(args.m2_file)
  m2.create_list_of_er_cor()
  g = []
  e = []
  # F0.5を計算するためのgの用意
  for c_sentence,e_sentence in zip(m2.correct_sentence,m2.error_sentence):
      g.append(F_05.make_dict(c_sentence,e_sentence))
  # F0.5を計算するためのeの用意
  for c_sentence in m2.correct_sentence:
      result, attention_weights = evaluate(c_sentence)  
      predicted_sentence = data.tokenizer_error.decode([i for i in result 
                                            if i < data.tokenizer_error.vocab_size]) 
      print(c_sentence)
      print(predicted_sentence)
      print()
      e.append(F_05.make_dict(c_sentence,predicted_sentence))

  print("F_0.5値：{}".format(F_05.calculate_f05(g,e)))

##########################################################################################
## mode によってやること変える
##########################################################################################
def main():
    if mode == "train":
        train()
    elif mode == "generate":
        input_sen = input("正しい英文：")
        translate(input_sen)
    elif mode == "F05":
        cal_F05()
        


if __name__ == "__main__":
    main()