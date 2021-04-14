import pandas as pd
import argparse
import os
import pathlib
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
## RuntimeError('unable to open shared memory object </torch_14580_2987853038> in read-write mode',) を防ぐのに必要 ##
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
######################################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_data_size = len(train_df)
    train_batch_size = args.t_batch
    eval_data_size = len(test_df)
    eval_batch_size = len(test_df)

    best_loss = 100

    for e in range(args.epochs):
        print(str(e+1) + "エポック目")

        ## Training
        for i in range(-(-train_data_size//train_batch_size)):
            input_list = []
            label_list = []
            for j in range(train_batch_size):
                try :
                    input_list.append(train_df.iat[i*train_batch_size+j,0])
                    label_list.append(train_df.iat[i*train_batch_size+j,1])
                except IndexError:
                    None

            inputs = tokenizer(input_list, return_tensors="pt", padding=True, truncation=True, max_length=30)
            inputs = inputs.to(device)
            labels = torch.tensor([label_list]).unsqueeze(0)
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        train_loss = loss.item()     ## 学習時のloss

        ## Evaluation
        model.eval()

        ## 評価の場合は勾配を保存しないように設定
        with torch.no_grad():
            for i in range(-(-eval_data_size//eval_batch_size)):
                input_list = []
                label_list = []
                for j in range(eval_batch_size):
                    try :
                        input_list.append(test_df.iat[i*eval_batch_size+j,0])
                        label_list.append(test_df.iat[i*eval_batch_size+j,1])
                    except IndexError:
                        None

                inputs = tokenizer(input_list, return_tensors="pt", padding=True, truncation=True, max_length=30)
                inputs = inputs.to(device)
                labels = torch.tensor([label_list]).unsqueeze(0)
                labels = labels.to(device)
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
            
        valid_loss = loss.item()     ## 評価時のloss

        ## 評価時の loss が一番小さいモデルを保存する
        if valid_loss <= best_loss:
            torch.save(model.state_dict(),args.pred_dir)
            print("saved an updated model!!")
            best_loss = valid_loss
        
        ## 学習状況の確認用
        print(f'Loss Train={train_loss:.4f}, Test={valid_loss:.4f}, Best={best_loss:.4f}')

    
def pred_classification(model_path, file):     ## 学習済みモデルをロードして推論を実行する(文章生成器と合わせて使う予定)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    inputs = tokenizer("New patients should arrive fifteen minutes before their scheduled appointments . ", return_tensors="pt")
    outputs = model(**inputs)[0]

    labels = ['GPT','HUMAN']
    print(labels[torch.argmax(outputs)])

    print(outputs)

def pred_glue(file):
    model = RobertaForSequenceClassification.from_pretrained('output/')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    labels = ['GPT','HUMAN']
    model.eval()
        
    count_H = 0     # 人の文と予測されたものをカウント
    count_G = 0     # GPT2の文と予測されたものをカウント
    rare = 0

    with open(file,'r') as f:
        for line in f:
            tokenized_text = tokenizer.tokenize(line)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])

            ## 推論の実行    
            with torch.no_grad():
                outputs = model(tokens_tensor)[0]
                label = labels[torch.argmax(outputs)]
                print(label)
                if label == "HUMAN":
                    count_H = count_H + 1
                else:
                    count_G = count_G + 1
        
    print("人の文：" + str(count_H))
    print("GPT2の文：" + str(count_G))
    


def main(args):
    if args.mode == "train":    ## 学習_評価
        train_df, test_df = preprocess(args.path)
        train(args,train_df,test_df)
    elif args.mode == "pred":   ## 推論
        pred_classification(args.pred_dir, args.test_file)
    elif args.mode == "pred_glue":  ## run_glue.py によるモデルを推論
        pred_glue(args.test_file)

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
    parser.add_argument("--pred_dir",type=str,default="./transformers_model/model.pth")    ## 推論する際のモデルがあるディレクトリ
    parser.add_argument("--test_file",type=str,default="test.txt")    ## 推論する際のテスト文   
    args = parser.parse_args()
    main(args)