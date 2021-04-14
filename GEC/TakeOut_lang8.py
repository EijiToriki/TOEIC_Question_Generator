import argparse

def main(args):
    err_sens = []       ## 英語学習者の文章
    cor_sens = []       ## 訂正文章    
    with open(args.file,'r') as fr:
        for line in fr:
            if line != "\n":
                line_list = line.split("\t")
                err_sens.append(line_list[4].replace('\n',''))
                cor_sens.append(line_list[-1].replace('\n',''))

    for err_sen,cor_sen in zip(err_sens,cor_sens):
        print(err_sen, cor_sen)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## ファイルを指定する
    parser.add_argument("--file",type=str,default="./dataset/lang-8-en-1.0/lang-8-en-1.0/mini_entries.test")
    args = parser.parse_args()
    main(args)