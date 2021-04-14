from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam


class Environment(object):
    def __init__(self, batch_size, vocab_size, emb_size, hidden_size,
                 T, dropout, lr):
        #色々パラメータ決めてる
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.T = T
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.discriminator = self._build_graph(
            self.vocab_size,
            self.emb_size,
            self.hidden_size,
            self.dropout
        )

    def _build_graph(self, vocab_size, emb_size, hidden_size, dropout):
        #入力層 shape:入力の次元 , dtype:入力から期待されるデータの型 , name:層の名前
        data_inp = Input(shape=(None, ), dtype='int32', name='input')

        #Embedding：正の整数を固定次元の密ベクトル（すべての要素の値を保持）に変換
        #単語などの構成要素に対して何らかのベクトル空間を与える
        #第一引数：語彙数
        #第二引数：密なembeddingsの次元数
        #第三引数：入力0をパディングのための特別値として使うか
        #第四引数：レイヤーの名前
        out = Embedding(
            vocab_size, emb_size, mask_zero=False, name='embedding'
        )(data_inp)

        #LSTM層：引数：正の整数値で出力の次元数
        out = LSTM(hidden_size)(out)
        
        #ドロップアウト
        #第一引数：rate:0と1の間の浮動小数点数．入力ユニットをドロップする割合．
        #第二引数：層の名前
        out = Dropout(dropout, name='dropout')(out)

        #全結合層
        #第一引数：出力空間の次元数
        #第二引数：使用する活性化関数
        #第三引数：層の名前
        out = Dense(1, activation='sigmoid', name='dense_sigmoid')(out)

        #data_inpを入力としてoutを計算するためにあらゆる層を含む
        discriminator = Model(data_inp, out)
        
        return discriminator

        #後ろについてる(data_inp)とか(out)って何？？？
        #こういう順番で層に入力取り込むってこと

    def pre_train(self, d_data, d_pre_episodes, d_pre_weight, d_pre_lr):
        d_optimizer = Adam(d_pre_lr)
        self.discriminator.compile(d_optimizer, 'binary_crossentropy') #二値分類
        self.discriminator.summary()
        self.discriminator.fit_generator(
            d_data,
            steps_per_epoch=None,
            epochs=d_pre_episodes
        )
        self.discriminator.save_weights(d_pre_weight)

    def initialize(self, d_pre_weight):
        self.discriminator.load_weights(d_pre_weight)
        d_optimizer = Adam(self.lr)
        self.discriminator.compile(d_optimizer, 'binary_crossentropy')
