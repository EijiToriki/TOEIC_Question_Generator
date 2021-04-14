# coding:shift-JIS
import linecache
import random

class Vocab(object):
    def __init__(self, sentences_path):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.UNK_TOKEN = '<UNK>'
        self.word2id = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.sentences = load_data(sentences_path)
        self.build_vocab(self.sentences)
        self.vocab_num = len(self.word2id)
        self.sentence_num = len(self.sentences)

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1
        for word, count in sorted(
            word_counter.items(), key=lambda x: x[1], reverse=True
        ):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

    def write_word2id(self, sentences_path, output_path):
        ids_sentences = []
        for line in open(sentences_path):
            words = line.strip().split()
            ids_words = [
                str(self.word2id.get(word, self.UNK)) for word in words
            ]
            ids_words = id_padding(ids_words,32)
            ids_sentences.append(ids_words)
        self.data_num = len(ids_sentences)
        output_str = ''
        for i in range(count_data(sentences_path)):
            output_str += ' '.join(ids_sentences[i]) + '\n'
        with open(output_path, 'w') as f:
            f.write(output_str)

    def write_id2word(self, ids_path, output_path):
        sentences = []
        for line in open(ids_path):
            ids = line.strip().split()
            words_ids = [
                self.id2word.get(int(id), self.UNK_TOKEN) for id in ids
            ]
            sentences.append(words_ids)
        output_str = ''
        for i in range(count_data(ids_path)):
            output_str += ' '.join(sentences[i]) + '\n'
        with open(output_path, 'w') as f:
            f.write(output_str)


def load_data(file_path):
    data = []
    for line in open(file_path):
        words = line.strip().split()
        data.append(words)
    return data


def count_data(file_path):
    data_num = 0
    for line in open(file_path):
        data_num += 1
    return data_num


def padding(sentences, T, PAD=0):
    sentences += [PAD for i in range(T - len(sentences))]
    return sentences

def id_padding(sentences, T, PAD='0'):
    for i in range(T - len(sentences)):
        sentences.append(PAD)
    return sentences


def sentence_to_ids(vocab, sentence, UNK=3):
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    return ids