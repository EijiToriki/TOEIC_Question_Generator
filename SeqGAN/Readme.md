## 概要
- 文章生成器SeqGANのプログラムが保存されている
- SeqGAN ⇒ LeakGAN ⇒ GPT2 の変遷があり，最初に試した文章生成器
- 性能は悪く，意味も文法も成り立っていない文章が生成される
- プログラム自体はそこまで難しくないので，pythonに慣れたい，強化学習のプログラムを試してみたいという勉強には役立つかも
- これを使ってシステムを作るということはない

## 各ファイルの説明
### data ディレクトリ
- プログラムで使うデータを格納している
    - review.json → yelp から引っ張ってきたレビューのファイル
        - url : https://www.yelp.com/dataset/download 
    - review.txt → review.json をテキストファイルに直したやつ．レビューの一文一文が記載
    - articles1.csv → ニュース記事をCSV形式にしているファイル
        - url : https://www.kaggle.com/snapcrack/all-the-news
    - articles1.txt → articles1.csv から記事本文だけを抽出したファイル 1文1文 にしている
    - toeic_sentense_part5.txt → 公式問題集から手打ちした 150 行くらいのファイル
- データ量が多すぎるファイル(review.json , review.txt , articles1.csv , articles1.txt)についてはファイルサイズが大きすぎるためアップロードしない．

### プログラム
- 以下の4つで seqgan が構成されている
    - main.py → seqgan で文章生成をする際に実行するファイル
    　　　　　　 生成されるファイルに関しては「現場で使える!」のseqganのページを見て
    - dict.py → 読み込んだテキストファイルから辞書をつくる
    - agent.py → 生成器の役割
    - environment.py → 判別機の役割
    - テキストデータを事前にDBに格納しておく必要がある．(send_data.py)

- 以下のプログラムは前処理で使う
    - convert_reviewjson_to_dataset.py<br>
     → review.json から review.txt を生成する．具体的には読み込んだレビューを1文1文にしてテキストファイルに書く．
        - 第一引数： json ファイル名
        - 第二引数： txt ファイル名
        - 第三引数： しきい値(その値以下の単語数の文は書き込まない)
    - sentece_sprit.py<br>
     → nltk を使って review.json を1文1文に区切って review.txtを生成する．また辞書も生成する．
    - load_news.py<br>
     → articles1.csv を読み込んで，articles1.txt を生成する
        - 第一引数： 最小単語数
        - 第二引数： 最大単語数
    - send_data.py<br>
     → DBにIDと文章のペアを保存する．
