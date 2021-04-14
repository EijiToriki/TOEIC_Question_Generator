import psycopg2
import json
import os
import glob
import argparse
import re

users = 'torikieiji' # initial user
dbnames = 'derivative'
passwords = ''
conn = psycopg2.connect(" user=" + users +" dbname=" + dbnames +" password=" + passwords)
# excexute sql
cur = conn.cursor()

parser = argparse.ArgumentParser()
parser.add_argument("--json_path",type=str,default="./data/json/")
args = parser.parse_args()

for f in glob.glob(args.json_path + '*.json'):
    json_file = args.json_path + os.path.split(f)[1]
    print(json_file)        ## 進行状況の確認用
    OpenFile = open(json_file,'r')
    derivative_dict = json.load(OpenFile)

    ## DB挿入処理
    for k,v in derivative_dict.items():
        cur.execute("SELECT EXISTS(SELECT * FROM deri_test WHERE original = '" + k + "');")
        t_f = cur.fetchall()
        if t_f[0][0]:       ## 既存単語の場合
            cur.execute("SELECT derivative FROM deri_test WHERE original = '" + k + "';")
            deri = cur.fetchall()
            current_str = deri[0][0]
            current_list = deri[0][0].split(',')        ## 現在の派生語リストを取得
            insert_flag = 0                             ## 挿入するかしないか
            for word in v:
                if word not in current_list:
                    current_str = current_str + "," + word
                    insert_flag = 1
            if insert_flag == 1 :
                cur.execute("update deri_test set derivative='" + current_str + "' where original = '" + k + "';") ## 挿入処理
        else:               ## 未知単語の場合
            s = ""
            for word in v:
                s = s + word + ","
            s = s.rstrip(",")
            cur.execute("insert into deri_test(original,derivative)values('"+ k + "','"+ s + "');") ## 挿入処理

## 派生語4未満削除
cur.execute("SELECT * FROM deri_test;")
deri = cur.fetchall()
# deri[i][j]    i:行，j:0=original, 1=derivative
for i in range(len(deri)):
    cur.execute("SELECT derivative FROM deri_test WHERE original = '" + deri[i][0] + "';")
    result = cur.fetchall()
    deri_list = result[0][0].split(',')        ## 現在の派生語リストを取得
    if len(deri_list) < 4:  ## 派生語4つ以下の行
        cur.execute("delete from deri_test where original = '" + deri[i][0] + "';")

cur.execute("SELECT COUNT(*) FROM deri_test;")
result = cur.fetchall()
count = int(result[0][0])

print(str(count) + '行')

conn.commit()
cur.close()
conn.close()