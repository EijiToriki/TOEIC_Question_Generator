import pandas as pd
import argparse
import os
import pathlib
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel,ClassificationArgs
## RuntimeError('unable to open shared memory object </torch_14580_2987853038> in read-write mode',) を防ぐのに必要 ##
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
######################################################################################################################

## ソフトマックス関数を計算
def softmax(a):
    x = np.exp(a)
    u = np.sum(x)
    return x/u

## テキストにラベルを付与
## 学習データとテストデータを分割
def preprocess(path):
    df = pd.DataFrame(columns=["label","text"])
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]

    for dir in files_dir:   # dataディレクトリ(GPTとHUMANのループ:これをラベルとする)
        p_temp = pathlib.Path(path + dir + '/').glob('*.txt') # テキストファイルの取得
        for p in p_temp:
            file_name = path + dir + '/' + p.name    # ファイル名の取得
            with open(file_name,'r') as fr: 
                for line in fr.readlines(): # テキストファイルを1行1行読み取る
                    df = df.append({'label':dir,'text':line},ignore_index=True) # テキストファイルをラベル付きでデータフレームにアペンド

    # ラベルの数値化(0:GPT2, 1:Human)
    df['label'], uniques = pd.factorize(df['label'])

    # 学習データとテストデータに分割
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=0, stratify=df["label"]
    )
    train_df = pd.DataFrame([X_train_df,y_train_s]).T
    test_df = pd.DataFrame([X_test_df,y_test_s]).T

    # 型の変換
    train_df["label"] = train_df["label"].astype("int")
    test_df["label"] = test_df["label"].astype("int")

    return train_df, test_df

def train(args,train_df,test_df):
    # モデルパラメータを決める
    model_args = ClassificationArgs()
    model_args.overwrite_output_dir = args.overwrite    ## 上書きに関する
    model_args.num_train_epochs = args.epochs           ## エポック数
    model_args.train_batch_size = args.t_batch          ## 学習バッヂサイズ
    model_args.eval_batch_size = args.e_batch           ## 評価バッヂサイズ

    ## early stopping：過学習を防ぐ
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 1000
    ##

    # 学習と評価の実行
    model = ClassificationModel('roberta', 'roberta-base', num_labels=2, use_cuda=True, args=model_args)
    model.train_model(train_df,acc=sklearn.metrics.accuracy_score)
    result, model_outputs, wrong_predictions = model.eval_model(test_df,acc=sklearn.metrics.accuracy_score)
    print(result)

def pred_classification(model_path, file):     ## 学習済みモデルをロードして推論を実行する(文章生成器と合わせて使う予定)
    model = ClassificationModel("roberta",model_path)

    count_H = 0     # 人の文と予測されたものをカウント
    count_G = 0     # GPT2の文と予測されたものをカウント
    rare = 0

    with open(file,'r') as f:
        for line in f:
            predictions, raw_outputs = model.predict([line])

            print(raw_outputs[0][0],raw_outputs[0][1])
            print(softmax(raw_outputs[0]))

            if predictions[0]:
                print("HUMAN")
                count_H = count_H + 1
            else:
                print("GPT")
                count_G = count_G + 1
    
    print("人の文：" + str(count_H))
    print("GPT2の文：" + str(count_G))

def main(args):
    if args.mode == "train":    ## 学習_評価
        train_df, test_df = preprocess(args.path)
        train(args,train_df,test_df)
    elif args.mode == "pred":   ## 推論
        pred_classification(args.pred_dir, args.test_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## 前処理に関するパラメタ
    parser.add_argument("--path",type=str,default="./data/")        ## テキストデータのパス
    ## モード選択のパラメタ
    parser.add_argument("--mode",type=str,default="pred")          ## train or pred
    ## 学習時のパラメタ
    parser.add_argument("--overwrite",type=bool,default=True)      ## 学習の際，上書きをするか否か
    parser.add_argument("--epochs",type=int,default=1)              ## エポック数
    parser.add_argument("--t_batch",type=int,default=32)              ## 学習時のバッジサイズ
    parser.add_argument("--e_batch",type=int,default=16)              ## 評価時のバッジサイズ

    ## 推論に関するパラメタ
    parser.add_argument("--pred_dir",type=str,default="outputs")    ## 推論する際のモデルがあるディレクトリ
    parser.add_argument("--test_file",type=str,default="test.txt")    ## 推論する際のテスト文   
    args = parser.parse_args()
    main(args)