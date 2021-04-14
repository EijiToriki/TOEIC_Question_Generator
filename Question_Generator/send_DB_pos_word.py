import psycopg2
import nltk

class send(object):
    def __init__(self,toeic_sen):
        #self.file_name = "./data/for_word2vec.txt" 
        self.toeic_sen = toeic_sen
        # connect postgreSQL
        self.users = '' # initial user
        self.dbnames = 'dict'
        self.passwords = ''
        self.conn = psycopg2.connect(" user=" + self.users +" dbname=" + self.dbnames +" password=" + self.passwords)
        # excexute sql
        self.cur = self.conn.cursor()
        self.cur.execute("SELECT COUNT(*) FROM toeic_dict")
        self.result = self.cur.fetchall()
        self.sentence = []       # 一文ずつ格納するリスト
        self.count = self.result[0][0] + 1   # id
        self.word_dict = []      # 単語と品詞のペアのリスト

        self.cur.execute("SELECT COUNT(*) FROM toeic_dict")
        self.result = self.cur.fetchall()


        for line in self.toeic_sen:
            line = line.replace("'","''")
            self.sentence.append(line)
            morph = nltk.word_tokenize(line)
            pos = nltk.pos_tag(morph)
            for word in pos:
                if word[1] != "NNP":
                    self.cur.execute("SELECT EXISTS(SELECT * FROM toeic_dict WHERE word = '"+ word[0] +"' AND pos = '"+ word[1] +"');")
                    t_f = self.cur.fetchall()
                    if not t_f[0][0]:      
                        self.cur.execute("insert into toeic_dict(id,pos,word)values(" + str(self.count) +",'"+ word[1].lower() + "','"+ word[0] + "');")
                        self.count = self.count + 1
                        self.word_dict.append(word)

        self.conn.commit()

        self.cur.close()
        self.conn.close()