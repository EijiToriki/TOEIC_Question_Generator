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

class Train_Eval_data(object):
    def __init__(self,d_model,num_layers,num_heads,dff,input_vocab_size,target_vocab_size,dropout_rate):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.dropout_rate = dropout_rate

        self.learning_rate = CustomSchedule(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, 
                                            epsilon=1e-9)
        self.temp_learning_rate_schedule = CustomSchedule(self.d_model)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        self.transformer = Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
                                self.input_vocab_size, self.target_vocab_size, 
                                pe_input=self.input_vocab_size, 
                                pe_target=self.target_vocab_size,
                                rate=self.dropout_rate)
        
        checkpoint_path = "./checkpoints/train"
        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')



    def train(self,EPOCHS,train_dataset):
        for epoch in range(EPOCHS):
            start = time.time()
          
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            
            # inp -> correct, tar -> error
            for (batch, (inp, tar)) in enumerate(train_dataset):
                train_step(inp, tar, self.transformer, self.optimizer, self.train_loss, self.train_accuracy)
    
                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))
                
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))
                
            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                            self.train_loss.result(), 
                                                            self.train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        


    def evaluate(self,inp_sentence,tokenizer,MAX_LENGTH):
        start_token = [tokenizer.vocab_size]
        end_token = [tokenizer.vocab_size + 1]
        
        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + tokenizer.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [tokenizer.vocab_size]
        output = tf.expand_dims(decoder_input, 0)
            
        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)
        
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
            
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            # return the result if the predicted_id is equal to the end token
            if predicted_id == tokenizer.vocab_size+1:
                self.result = tf.squeeze(output, axis=0)
                self.weights = attention_weights
            
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        self.result = tf.squeeze(output, axis=0)
        self.weights = attention_weights




# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]
# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy):
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



def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

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