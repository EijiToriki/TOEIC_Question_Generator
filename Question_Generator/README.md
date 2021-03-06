## 概要
- 本研究の要となるプログラムが保存されている
  - 問題生成に関わるプログラム
- main.py を動かすことによって問題が生成される
  - 現状で生成される問題は，動詞の活用形問題と派生語問題の2つ

## プログラムについて
### Json2DB.py
- 後述する derivative_morpheme.py で生成された json ファイルを派生語DBに格納するプログラム
  - 既存単語の場合は，その単語の格納されていない派生語がないかを判定し，DBになければ格納する
  - 未知単語の場合は，その単語と派生語を格納する
※ ベースとなる単語を語基という(ex:effectiveの語基はeffect)

### Q_generate.py
- 問題を生成するプログラム
- 関数 make_Q では，事前に決められた問題パターンに即した問題を出力する
  - 問題パターンはTOEIC問題集 part5 の問題を360問分析した出題確率に即している
    - main.pyで予め問題のパターンを決定している
  - 動詞の活用形問題
    - 答えが1語の動詞問題 (現在形，過去形とか)
    - 答えが2語の動詞問題 (受動態，現在完了とか)
  - 派生語問題
    - 答えが名詞になる問題
    - 答えが動詞になる問題
    - 答えが形容詞になる問題
    - 答えが副詞になる問題
- コンストラクタには，問題文にする文章を渡す必要がある
  - 問題文はmain.pyで生成される文を渡す
- verb_form_q は，動詞の活用形問題と派生語問題の選択肢を作る関数
  - 難易度別で選択肢を作るようにしている
    - この難易度も確率的に出題するかしないかを決定する
    - 名詞問題の難問：派生語だが，品詞が名詞である単語が複数ある(ex：performance が答えの問題で選択肢に performer が混じっている)
    - 形容詞問題の難問：形容詞，現在分詞，過去分詞が選択肢に並ぶ(ex：conncting が答えの問題で選択肢に connected が混じっている)<br> ⇒ 文法だけではなく，意味も考慮する必要があるので2段階の思考が必要になる
- 関数 same_pos_q では，意味問題を実装している
  - 2020/7月くらいからアップデートしておらず，未完成
  - 問題になりえない超マイナー単語も選択肢に現れたりと，TOEICっぽさがない
  - 文章生成器も変な文を出すことが多いので，意味を汲み取るのは難しい
  - 文章生成器を改良することが先決

### ToDBCommand.py
- 派生語DBに関わるプログラム，derivative_morpheme.py と Json2DB.py を順番に手作業で実行するのは面倒
- 2つのプログラムを一度に実行し，テキストファイルも一気に処理しようというプログラム
- このプログラムを実行することによって派生語DBを構築できる
- ./data ディレクトリを用意し，その中にPIEデータである，a(number)_train_corr_sentences.txt を格納しておく．(number)は1〜5の整数．
  
### create_model.py
- 意味問題での使用を考えている word2vec のモデルを生成するクラス
  - word2vecでは単語間のコサイン類似度を測ることができ，似た単語が選択肢に含まれないようにすることで，答えが複数存在する悪問を出題しないようにする
- word2vec用のテキスト英文ファイルを事前に要しておく
  - ファイル名は for_word2vec.txt としておく．
  - data ディレクトリに保存しておく
  - ~~news文＋新しくGPT2で作った文のテキストファイルを入力することを想定している~~
  - ニュース記事のデータセットである程度のモデルを構成しておく．その後生成文を追加学習するようにしている(updateTrain関数)．

### derivative_morpheme.py
- テキストファイルから派生語辞書を獲得し，Jsonファイルに保存するプログラム
- SnowballStemmerによりstem（語基）を獲得し，stem（語基）が同じものを派生語とみなす

### main.py
- 問題を生成する際に実行するプログラム
  1. 問題パターンを決定する
  2. GPT2で文章を生成
  3. 生成された文章の中に，問題パターンに即した文が含まれているかをチェックする
    - 品詞や単語数の観点でチェック
    - 問題パターンに即した文がない場合は，文を作り直す
  4. 問題生成器のクラス Q_generate に文を渡し，問題生成を行う

### send_DB_pos_word.py
- GPT-2で生成したテキストファイルの英文に含まれる，単語と品詞のペアを ID 付きで DB に格納するクラス
  - 単語DBの単語を増やす役割
- 新規単語のみを登録できるようにしている

### to_base_form.py
- 動詞の原形を返すクラス
- 動詞の活用形を取得するために，原形が必要となるので作成した．
  - Webスクレイピングして，動詞の不規則変化表を取得．それに沿って，原形を取得できるようにした
  - 現在分詞や規則動詞に関しては，wordnetlemmatizer によって，ing や ed を取るようにした

### verb_pattern.py
- 二語以上の動詞を作成するプログラム
- 主語を見て，それに応じたhaveやbeの形を取得し，二語以上の動詞を作る
- 現在完了，進行形，受動態，現在完了受動態を作成可能
  
## プログラムの実行
- python main.py --mondaisu="問題数" --q_pattern="v_form" --format="file or print" により文章と問題どちらも生成される
  - file を指定することで，テキストファイルに問題を書き出す
  - print を指定することで，コマンドラインで問題を解ける
- dataディレクトリを事前に作り，GPT2で生成するのに使うTOEICデータを用意しておく
  - 今後意味問題を作成する上で，dataディレクトリ内に，word2vecのモデルも必要になる

