# 文章判別器
- 与えられた文が人の作った文かGPTの作った文かを判別する機械学習モデル
- GPT2の文が人が作った文だと判定されれば良い文とみなし，問題文にする

## プログラムについて
### 下準備
- GPT2と人の書いた文を別々のディレクトリに保存しておく必要がある
- main.pyおよびtransformers_main.pyをカレントディレクトリとして，./data/HUMANに人の書いた文を，./data/GPTにGPT2の文を保存することで，学習が可能になる

### main.py
- SimpleTransformers での実装
  - あまりにも精度が良すぎて，逆に使えない問題が発生している
  - 精度は，96%ほど．GPT2の文が人の書いた文と間違うことはほぼない
  - 学習が完了すると，cache_dir, outputs, runs 3つのディレクトリができている

### transformers_main.py
- Huggingface の Transformers での実装
  - SimpleTransformers で実装したときよりも性能が落ちている
    - 10epoch回してもlossが0.55くらい
- パラメータ mode を pred_glue にすると，run_glue.py(transformers の example にある文章分類学習プログラムの例) を動かした後に生成されたモデルで推論を行うことができる
  - モデルのディレクトリはoutputとしておく
  - このモデルでは，人の文を人と判定する正解率は98%, 一方でGPTの文をGPTと判定する正解率は34%
    - 3回に1回GPT2の文を人と判定するので，システムでも使えそうなモデルである

### パラメータ
- main.pyもtransformers_main.pyも同じである
  - path：テキストデータがどこにあるか．デフォルトは ./data/
  - mode：学習するか推論するか．デフォルトは推論を表す pred．学習をする際は train にする
  - overwrite：学習時に，既存モデルを上書きするか否か．デフォルトは上書きを許可するTrue
  - t_batch：学習時のバッチサイズ
  - e_batch：評価時のバッチサイズ
  - epochs：エポック数
  - pred_dir：推論時に呼び出すモデル．
    - main.py では，outputsを指定すればよい
    - transformers_main.py では，学習後にモデルを保存するパスとしても使用する
  - test_file：推論をする際にテストするテキストファイル

## 今後の課題
- 判別器を完成させる
  - 拡張性を考慮して，HuggingfaceのTransformersで実装できるようにする
  - pytorch ベースのほうが応用が効きやすい
- 判別器ができたら，GPT2をファインチューニング
  - 判別器を騙せるようにGPT2を学習させる
  - 判別器を学習させない GAN を作る
    - 判別器の精度が上がりすぎては，一生文が生成されないというSimpleTransformersと同じ問題が生じるため，判別器は学習しない
