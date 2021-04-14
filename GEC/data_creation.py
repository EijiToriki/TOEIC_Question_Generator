import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras

class data_creation(object):
    def __init__(self,correct_sentence,error_sentence,buffer_size,batch_size,max_length):
        self.correct_sentence = correct_sentence
        self.error_sentence = error_sentence
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_length = max_length
        ## tf.data の形式に変換
        self.dataset = tf.data.Dataset.from_tensor_slices((self.correct_sentence, self.error_sentence))

        ## ID化のために必要
        # self.tokenizer_correct = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        # (cor.numpy() for cor,er in self.dataset), target_vocab_size=2**13)
        # self.tokenizer_error = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        # (er.numpy() for cor,er in self.dataset), target_vocab_size=2**13)

    def create_tokenizer(self):
        ## ID化のために必要
        self.tokenizer_correct = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (cor.numpy() for cor,er in self.dataset), target_vocab_size=2**13)
        self.tokenizer_error = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (er.numpy() for cor,er in self.dataset), target_vocab_size=2**13)

        self.tokenizer_correct.save_to_file("./tokenizer/correct")
        self.tokenizer_error.save_to_file("./tokenizer/error")

    def encode(self,correct,error):    ## 文の先頭と末尾に目印を付与 + ID変換
        correct = [self.tokenizer_correct.vocab_size] + self.tokenizer_correct.encode(
            correct.numpy()) + [self.tokenizer_correct.vocab_size+1]
        error = [self.tokenizer_error.vocab_size] + self.tokenizer_error.encode(
            error.numpy()) + [self.tokenizer_error.vocab_size+1]
        
        return correct,error

    def tf_encode(self,correct,error):
        result_correct, result_error = tf.py_function(self.encode, [correct,error], [tf.int64, tf.int64])
        result_correct.set_shape([None])
        result_error.set_shape([None])

        return result_correct, result_error

    def filter_max_length(self,x,y):
        max_length=self.max_length
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    def create_dataset(self):
        self.dataset = self.dataset.map(self.tf_encode)
        self.dataset = self.dataset.filter(self.filter_max_length)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.shuffle(self.buffer_size).padded_batch(self.batch_size)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

