# TOEIC Part5 自動問題生成システム
## 概要
- 本システムは以下の2つで構成される
  - 文章生成部
    - TOEIC の問題に即して文を生成．GPT2のストーリー生成機能に公式問題集の文を入力することで，続きの文がTOEICのTOPICに沿った文になるようにしている．
    - GPT2 は，huggingfaceのTransformersを使用している．
    - 文法ミスや謎な単語が現れる点が現状(2021/4/6時点)の問題．
  - 問題生成部
    - 現状(2021/4/6時点)では，動詞の活用形問題と派生語問題を生成可能．
    - 動詞の活用形問題：現在形，過去形，未来形，進行形，受動態，現在完了などの正しい動詞の形を選ぶ問題．
    - 派生語問題：答えの単語の派生語が選択肢として与えられる問題．
- 文章生成部で自動生成した文に対して，問題生成部で穴を開ける．この流れでTOEIC Part5 を模倣した問題を生成することが研究目的である
- 文章生成器も問題生成器も Question_Generator の main.py で実装されている．

## 開発環境
- Python 3.6.10
- 必要ライブラリ
  - mlconjug3 3.7.6
  - tensorflow 2.4.0
  - torch 1.7.1
  - transformers 4.3.3

## 現状(2021/4/6)でのタスク
1. 文章判別器を用いた GAN の実装
  - 文章判別器を騙せるように，文章生成器を学習することで質の高い文を作れるのでは?という発想
  - 詳細と現状のプログラムは discriminater のフォルダを参照せよ
2. 意味問題の実装
  - 現状のシステムに問題パターンを更に追加したい
  - 意味問題は，part5で5割程度を占める頻出問題である
  - 詳細と現状のプログラムは Question_Generator のフォルダを参照せよ

## 各ディレクトリの概要
### GEC
- 逆GECを検討していた際に使用していたディレクトリ
- 上手く行かないことが分かっているので，使う予定はなし

## LeakGAN
- 文章生成器 LeakGAN を使おうとしていたときに，作ったプログラムを保存している
- 絶対に使うことはないので見る必要なし
- SeqGAN ⇒ LeakGAN ⇒ GPT2 という文章生成の変遷があったことを認識してもらうために残しておく

## Question_Generater
- 問題生成に関わるプログラムが全て格納されている
- ここをメインに修正，アップロードしていくことになる

## SeqGAN
- 文章生成器 SeqGAN を使おうとしていたときに，作ったプログラムを保存している
- LeakGAN と同様お話にならない性能のため使うことはない
- python に慣れるため，とりあえず文章生成や強化学習を試してみたいなら，実行してみても良いかも
- 研究を進める上では必要ない

## discriminater
- 「現状(2021/4/6)でのタスク」に記載した，文章判別器を用いた GAN の実装 に関するプログラムを保存している
- SimpleTransformers で実装した文章判別機と，huggingface の Transformers で実装した文章判別器が保存されている

## gpt-2
- 最初にGPT2を発見して，使おうとしていたときに作ったディレクトリ
- 今は Question_Generater に文章生成器もあるため，ここをいじることはない．
- とりあえずGPT2を使ってみたいときに，動かすのは有効

## 文章生成に関して
### SeqGAN
- 初めに試した文章生成器
- 「現場で使える!Python深層強化学習入門」のプログラムを基に文章生成
  - 本のプログラムではテキストファイルを読み込んでの学習だったが，メモリ量の関係でDBから文を読み込んで学習するように変更した
- 生成される文はただ単語を羅列しただけで，意味も文法も成り立っていない．問題文としては到底使えない

### LeakGAN
- SeqGAN の次に試した文章生成器
- 2017年くらいに論文が出ている
- 学習しても，文法メチャクチャな文しか出てこないので，問題文としては使えない
 
### GPT2
- SeqGANもLeakGANも文章の質が低い．改善しようと大量の文で学習しようとするとメモリ溢れでそもそも学習できない
- GPT2は簡単に文章生成ができ，なおかつ，ある程度文法ルールや意味を保った文が生成されている
- 現状(2021/4/6時点)で，問題とする文として最も適切なベースラインとなっている

### GPT2 の問題点と改良案
- SeqGAN や LeakGAN と比べると格段に意味のある文を作ってくれるのが GPT2
- しかし，文法ミスや謎な単語が現れる問題点も存在する
- 現状(2021/4/6時点) での案は，文章判別機の実装である
  - 文章判別器：入力された文が人の書いた文かGPT2が書いた文かを判別するモデル
  - 文章判別器を騙して人と判定されたものを，問題文にすればよいのでは?という発想
    - 2021年3月に SimpleTransformers(huggingfaceのTransformersをさらに簡単化したもの)を用いて，判別器の実装をしたが，精度が高すぎてGPTが判別器を騙せなかった．
  - 文章判別器を使って，文章生成器を学習させる GAN の判別器を学習させないバージョンを実装することを目標としている
    - 文章判別器の拡張性を持たせるために，huggingfaceのTransformersでの実装を行う．
    - pytorchで実装することによって，GANの実装も少しはやりやすくなるはず．
    
## 問題作りに関して
- 想定される問題は以下の5通り
  - 派生語問題
  - 関係代名詞や比較級などの文法問題
  - 動詞の活用形問題(現在・未来・過去・完了形・現在分詞・過去分詞)
  - 文章から正しい問題を選ぶ単語の意味問題
  - 文章から正しい問題を選ぶ連語の意味問題  
- 2021/4/6段階で動詞の活用形問題，派生語問題を出力できるようにしている

※ プログラムはQuestion_Generatorを随時更新していく

## GECについて
- GEC(Grammar Error Correction)は文法訂正のこと
- 文法訂正ではなく，正しい文から誤った文を出力する(逆GEC)ことによって問題生成に使えないか検討する
  - 例えば， He plays tennis every day. という文を入力した際に，He play tennis every day. や He were playing tennis every day. が出力されることを理想としていた．
  - 誤ったところを穴にすることで，人が間違えやすい = 問題にしやすい という考えで取り組んでいた
- しかし，逆GEC用のデータセットはなく，GEC用のデータセットで行っても上手くは行かず，2020/11月で諦めた
  - F値も計算したが，かなり低い値で 10^{-2} 以下のオーダーだった
