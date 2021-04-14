## 概要
- tensorflowでtransformerを実装
- transformerを用いて逆GECモデルを作る取り組み

## プログラムの概要
- main_transformer.py と main_transformer_global.py以外は Tensorflow 公式チュートリアルのコピーである
  - 公式では colab や jupyter notebook での実行を想定しているのかセルごとの実装だったが，ここでは機能ごとにクラス化している
  - この辺りはブラックボックス化しても構わないが，原理を知りたければ，チュートリアルを見ることをおすすめする
- main_transformer.py と main_transformer_global.py は機能的には同じ
  - main_transformer_global.py はパラメータをグローバル変数としている．グローバル変数にしたほうが実装が容易だったからである．
  - main_transformer_global.py で学習をすることができる

### main_transformer_global.py について
- PIEファイル，m2ファイル，lang8ファイルどのデータで学習するかを決められる
  - m2ファイルはデータ量少ないため，評価で使うのがおすすめ
  - 学習にはlang8かPIEを使うべき
    - lang8 はデータ量がそこまで多くないので学習時間が短い．しかし，得られる結果は微妙．
    - PIE は5つのテキストファイルからなり，1つあたり1.2Gと大規模なテキストファイルである．学習に時間がかかり，メモリ溢れのため全てのデータを使って学習できたことはない
- コマンドライン引数 mode を train / generate / F05 にすることで機能が変わる
  - train：学習する．チェックポイントがある場合はそれを使う
  - generate：入力文が必要．その入力文を誤った文に変える．1回の推論を試すことができる
  - F05：F値を計算することができる

※ プログラムの多さから，ディレクトリを分けているが，実際に動かす際はGEC内のプログラムと同じ階層に全てのプログラムが存在していなければならない
