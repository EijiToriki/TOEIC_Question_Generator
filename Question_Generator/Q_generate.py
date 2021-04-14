import nltk
import mlconjug3
import random
import psycopg2
import stanza
import verb_pattern as vp
from nltk.stem.snowball import SnowballStemmer
from gensim.models import word2vec
from to_base_form import base_form

###################################################################################
## 問題生成に関わるクラス
###################################################################################
class Q_generate(object):
    def __init__(self,toeic_sen):
        self.word_dict = []      # 単語と品詞のペアのリスト
        self.sel_list = []       # 選択肢のリスト
        self.qpos_list = ['NN','NNS','JJ','RB','VB','VBD','VBG','VBN','VBP','VBZ']   # 問題にする品詞のリスト(意味問題)
        self.v_list = ['VB','VBD','VBG','VBN','VBP','VBZ']   # 問題にする品詞のリスト(動詞問題)
        self.n_list = ['NN','NNS']  # 名詞リスト
        self.adj_list = ['JJ','VBG','VBN']  # 形容詞の役割をするリスト
        self.adv_list = ['RB']
        self.q_pos_dict = {"v":self.v_list, "n":self.n_list, "adj":self.adj_list, "adv": self.adv_list}
        self.q_sen = toeic_sen
        self.stemmer = SnowballStemmer("english")

    # 問題を作成し,問題文と答えを返す
    def make_Q(self,big,small):
        ## 問題文を単語と品詞に分解 [][0]:単語 [][1]:品詞
        morph = nltk.word_tokenize(self.q_sen)
        pos = nltk.pos_tag(morph)

        ## 乱数のリスト(値の重複なし)
        word_num = random.sample(range(len(pos)),k=len(pos))

        double_flag = 0
        candidate = []
        hole_word = ""

        if big == "v_form":     ## 動詞活用形問題
            if small == "single":   ## 1語の動詞
                for i in word_num:
                    hole_word = pos[i][0]
                    hole_pos = pos[i][1]
                    if hole_pos in self.v_list:
                        break
            else:                   ## 2語以上の動詞
                double_flag,candidate = self.check_2more(pos)
                for w in candidate:
                    if w == candidate[-1]:
                        hole_word += w
                    else:
                        hole_word += w + " "
                can_morph = nltk.word_tokenize(hole_word)
                can_pos = nltk.pos_tag(can_morph)
                hole_pos = pos[-1][1]
        else:                   ## 派生語問題
            for i in word_num:
                hole_word = pos[i][0]
                hole_pos = pos[i][1]
                if hole_pos in self.q_pos_dict[small]:  ## 決定した問題パターンの品詞か否か
                    stem = self.stemmer.stem(hole_word)
                    stem = stem.replace("'",'"')
                    conn = self.connection_db_deri()
                    cur = conn.cursor()
                    cur.execute("SELECT EXISTS(SELECT * FROM deri_test WHERE original = '" + stem + "');")
                    t_f = cur.fetchall()
                    if t_f[0][0]:       ## 単語がDBに含まれているかのチェック
                        break

        q_sen_blank = self.make_qsen(double_flag,candidate,hole_word,morph,pos)     #穴をあけた q_sen を取得する
        self.verb_form_q(hole_pos,hole_word,big,small)     # 動詞の活用，派生語問題の選択肢を作成

        return hole_word,q_sen_blank     # 答えと問題文を返す


    def verb_form_q(self,hole_pos,hole_word,big,small):
        ######################################################################
        ## 選択肢の候補リスト作成
        ######################################################################

        init_flag = 0       # 先頭大文字：1, 小文字：0
        if hole_word[0].isupper():  # 答えの先頭文字が大文字なら
            init_flag = 1
            hole_word = hole_word.lower()

        cl = []                     ## 選択肢候補(cl=candidate list)を入れておくリスト        
        if big == 'v_form':         ## 動詞活用形問題
            ## 主語の抜き出し
            nlp = stanza.Pipeline('en') # initialize English neural pipeline
            doc = nlp(self.q_sen) # run annotation over a sentence
            sub_toks = {}
            for sent in doc.sentences:
                for word in sent.words:
                    if word.deprel == "nsubj":
                        sub_toks[sent.words[word.head-1].text if word.head > 0 else "root"] = [word.text]
            ## 動詞の原形を返すクラスの設定
            BS = base_form()

            if small == 'double':   ## 答えが2語以上の問題
                # 二語以上の動詞を分ける
                v_morph = nltk.word_tokenize(hole_word)
                v_pos = nltk.pos_tag(v_morph)
                bs = BS.to_base(v_pos[-1][0],v_pos[-1][1])     # 動詞の原形が返ってくる
                cl = self.get_verb_list(bs)                  # 動詞の活用リストが返ってくる
                cl.extend(vp.create_two_more_verb(v_pos[-1][0],v_pos[-1][1],sub_toks))
            else:
                bs = BS.to_base(hole_word,hole_pos)     # 動詞の原形が返ってくる
                cl = self.get_verb_list(bs)                  # 動詞の活用リストが返ってくる
                cl.extend(vp.create_two_more_verb(hole_word,hole_pos,sub_toks))
        elif big == 'deri':         ## 派生語問題
            stem = self.stemmer.stem(hole_word)
            stem = stem.replace("'",'"')
            conn = self.connection_db_deri()
            cur = conn.cursor()
            cur.execute("SELECT derivative FROM deri_test WHERE original = '" + stem + "';")
            deri = cur.fetchall()
            cl = deri[0][0].split(',')        ## 現在の派生語リストを取得
        cl = list(set(cl))

        print(cl)

        #################################################################################
        ## 選択肢を作る(難易度別で)
        #################################################################################
        cl_pos = nltk.pos_tag(cl)
        self.sel_list.clear()
        self.sel_list.append(hole_word)

        ## 難問 
        ### 名詞答えなら60%で，形容詞が答えなら70%で難問にする
        difficult_q = random.randint(0,100)
        if difficult_q <= 60:
            ### 答え：名詞編
            if hole_pos in self.n_list:     # 答えが名詞なら，他の名詞も選択肢に含める
                for w,p in cl_pos:
                    if p == hole_pos and w not in self.sel_list:
                        self.sel_list.append(w)     # 別の意味の名詞
                    if len(self.sel_list) == 4:
                        break

        if difficult_q <= 70:
            ### 答え：形容詞・分詞編
            if hole_pos in self.adj_list:   # 答えが形容詞とか分詞なら，形容詞・現在分詞・過去分詞を選択肢に含める
                for w,p in cl_pos:
                    if p in self.adj_list and w not in self.sel_list:
                        self.sel_list.append(w)     # 形容詞・現在分詞・過去分詞のダミー
                    if len(self.sel_list) == 4:
                        break

        while len(self.sel_list) < 4:
            sel_num = random.randint(0,len(cl)-1)
            if cl[sel_num] not in self.sel_list:
                self.sel_list.append(cl[sel_num])     
            
        if init_flag == 1:  # 答えの先頭文字が大文字なら
            self.init_upper()            # 他の選択肢も大文字にする

        random.shuffle(self.sel_list)


    # ## 同じ品詞違う意味問題の選択肢のリストを作る
    def same_pos_q(self,hole_pos,hole_word):
        ## 穴と同じ品詞の単語を取得
        conn = self.connection_db()
        cur = conn.cursor()
        cur.execute("select word from toeic_dict where pos = '" + hole_pos + "';")
        result = cur.fetchall()

        data_size = len(result)        

        list_pos = [] # pos = part of speech(品詞) 同じ品詞の単語リスト
        for word,word_pos in self.word_dict:
            if word_pos == hole_pos :
                list_pos.append(word)

        ## 同じ品詞から選択肢を作る(似た単語は選択肢に入れない)
        self.sel_list.clear()
        self.sel_list.append(hole_word)
        while True:
            sel_num = random.randint(0,data_size-1)
            if self.similar_2_words(hole_word,result[sel_num][0]) < 0.5:
                self.sel_list.append(result[sel_num][0])     #result[i][0] で i 番目の単語抜き取れる
            if len(self.sel_list) == 4:
                break

        if hole_word[0].isupper():  # 答えの先頭文字が大文字なら
            self.init_upper()            # 他の選択肢も大文字にする

        random.shuffle(self.sel_list)

        cur.close()
        conn.close()

    ## 二つの単語の類似度を出力
    def similar_2_words(self,word,word2):
        Model = word2vec.Word2Vec.load("./data/toeic.model")
        try:
            return Model.wv.similarity(word,word2)
        except KeyError:
            return 1.0

    def get_verb_list(self,verb):
        # You can now iterate over all conjugated forms of a verb by using the newly added Verb.iterate() method.
        default_conjugator = mlconjug3.Conjugator(language='en')
        test_verb = default_conjugator.conjugate(verb)
        all_conjugated_forms = test_verb.iterate()

        verb_list = []
        kind_list = []
        length = len(all_conjugated_forms) - 3

        for i in range(length):
            if i == 24:     # 不定詞の時
                verb_list.append(all_conjugated_forms[i][2])
                kind_list.append(all_conjugated_forms[i][1])
            elif all_conjugated_forms[i][3] not in verb_list or all_conjugated_forms[i][1] not in kind_list:
                verb_list.append(all_conjugated_forms[i][3])
                kind_list.append(all_conjugated_forms[i][1])

        return verb_list  # verb_list[] 0:原形，1:三人称単数現在，2:過去形，3:現在分詞，4:過去分詞，5:不定詞

    def check_2more(self,pos):      # 二語以上の動詞が文に含まれるかを検索する
        candidate = []              # 二語以上の動詞候補
        buf_i = -2
        double_flag = 0
        for i,(w,p) in enumerate(pos):   # 二語以上の動詞を検索
            if p in self.v_list or w == "will":     # will + 動原 も選択肢にしたい
                if len(candidate) == 0:
                    candidate.append(w)
                elif i-buf_i == 1:
                    candidate.append(w)
            else:
                if len(candidate) >= 2:
                    double_flag = 1
                    break
                else:
                    candidate.clear()
            buf_i = i

        return double_flag, candidate

    def make_qsen(self,double_flag,candidate,hole_word,morph,pos):
        q_sen_blank = ""
        flag = 0

        if double_flag == 1:
            candidate_counter = len(candidate)
            for i in range(len(morph)):
                if pos[i][0] == candidate[0]:
                    flag = 1
                    for j,w in enumerate(candidate):
                        if pos[i+j][0] != w:
                            flag = 0
                            break
                    if flag == 1:
                        q_sen_blank = q_sen_blank + "( "
                        candidate_counter = candidate_counter - 1
                elif flag == 1 and candidate_counter > 0:
                    if candidate_counter == 1:
                        q_sen_blank = q_sen_blank + " ) "
                        flag = 0
                    candidate_counter = candidate_counter - 1
                elif pos[i][0] == '.':
                    q_sen_blank = q_sen_blank + pos[i][0]
                else:
                    q_sen_blank = q_sen_blank + pos[i][0] + " "
        else:
            for i in range(len(morph)):
                if pos[i][0] == hole_word and flag==0 :     # 1文に同じ単語が二語含まれていたとき，文の最初に出てきた単語を穴にする
                    q_sen_blank = q_sen_blank + "(  ) "
                    flag = 1
                elif pos[i][0] == '.' :
                    q_sen_blank = q_sen_blank + pos[i][0]
                else:
                    q_sen_blank = q_sen_blank + pos[i][0] + " "
        return q_sen_blank

    def init_upper(self):
        for i,word in enumerate(self.sel_list):
            word = word[0].upper()+word[1:]
            self.sel_list[i] = word

    ## DB接続
    def connection_db(self):
        # connect postgreSQL
        users = 'torikieiji' # initial user
        dbnames = 'dict'
        passwords = ''
        conn = psycopg2.connect(" user=" + users +" dbname=" + dbnames +" password=" + passwords)

        return conn

    ## DB接続
    def connection_db_deri(self):
        # connect postgreSQL
        users = 'torikieiji' # initial user
        dbnames = 'derivative'
        passwords = ''
        conn = psycopg2.connect(" user=" + users +" dbname=" + dbnames +" password=" + passwords)

        return conn

    ## 問題と選択肢を表示
    def print_Q(self,i,hole_word,q_sen_blank,format,q_file,a_file):
        text_q = "問題{} : {}\n"
        text_a = "問題{}の答え：{} {}"
        if format == "print":
            print()
            print(text_q.format(i+1,q_sen_blank))
            print("選択肢")
            print("(A) %s" % self.sel_list[0])
            print("(B) %s" % self.sel_list[1])
            print("(C) %s" % self.sel_list[2])
            print("(D) %s" % self.sel_list[3])

            print("あなたの答えを入力(A,B,C,Dのいずれかで入力)")
            
            while True:
                ans = input()
                if ans == 'A' or ans == 'B' or ans == 'C' or ans == 'D':
                    break
                else:
                    print("入力エラー")
                    print("A,B,C,Dのいずれかで入力してください")
            
            self.print_ans(ans,hole_word)
            print("答えを確認したら何か入力してください")
            mozi = input()
        ## ファイルに書き込む処理
        elif format == "file":
            q_file.write(text_q.format(i+1,q_sen_blank))
            q_file.write("(A) %s \t" % self.sel_list[0])
            q_file.write("(B) %s \t" % self.sel_list[1])
            q_file.write("(C) %s \t" % self.sel_list[2])
            q_file.write("(D) %s \t" % self.sel_list[3])
            q_file.write("\n \n")

            for j in range(4):
                if hole_word == self.sel_list[j]:
                    if j == 0:
                        ans_fig = "(A)"
                    elif j == 1:
                        ans_fig = "(B)"
                    elif j == 2:
                        ans_fig = "(C)"
                    elif j == 3:
                        ans_fig = "(D)"             
                    a_file.write(text_a.format(i+1,ans_fig,hole_word))
                    a_file.write("\n")
                    break

    ## 正解かどうかを照合
    def print_ans(self,ans,hole_word):
        if ans == 'A':
            ans_num = 0
        elif ans == 'B':
            ans_num = 1
        elif ans == 'C':
            ans_num = 2
        else:
            ans_num = 3
        
        if self.sel_list[ans_num] == hole_word:
            print("正解")
        else:
            print("不正解")
            print("正解は",hole_word)
    ###################################################################################

    # 問題を回数分出力したりのループ
    def generate(self,i,format,big,small):    # i:問題番号,  big:大区分, small:小区分
        q_file = "./data/q_file.txt"
        ans_file = "./data/ans_file.txt"

        with open(q_file,'a') as qf, open(ans_file,'a') as af:
            hole_word,q_sen_blank = self.make_Q(big, small)
            self.print_Q(i,hole_word,q_sen_blank,format,qf,af)