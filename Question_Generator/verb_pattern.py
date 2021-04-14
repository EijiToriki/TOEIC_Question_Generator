import nltk
import mlconjug3
import random
import psycopg2
import stanfordnlp
import stanza
from gensim.models import word2vec
from to_base_form import base_form

def get_verb_list(verb):
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

###############################################################
## 助動詞絡み
###############################################################
def create_av(verb,pos):
    # av = auxiliary verb(助動詞)
    av = ["can","may","must","have to","should","will","shall","had better","ought to","could","would","might","can't","won't","couldn't","wouldn't"]

    av_index = random.randint(0,len(av)-1)
    BS = base_form()
    bs = BS.to_base(verb,pos)     # 動詞の原形が返ってくる
    form = av[av_index] + " " + bs

    return form

def create_apf(verb,pos):       ## 助動詞を含む進行形
    av = ["can","may","must","have to","should","will","shall","had better","ought to","could","would","might","can't","won't","couldn't","wouldn't"]

    av_index = random.randint(0,len(av)-1)
    BS = base_form()
    bs = BS.to_base(verb,pos)     # 動詞の原形が返ってくる
    ing = get_verb_list(bs)[-3]      # 過去分詞を取得
    form = av[av_index] + " " + "be" + " " + ing

    return form

def create_app(verb,pos):       ## 助動詞の過去形 + 現在完了
    av = ["may","must","should","could","can't"]

    av_index = random.randint(0,len(av)-1)
    BS = base_form()
    bs = BS.to_base(verb,pos)     # 動詞の原形が返ってくる
    pp = get_verb_list(bs)[-2]      # 過去分詞を取得
    form = av[av_index] + " " + "have" + " " + pp

    return form

def create_apv(verb,pos):       ## 助動詞を含む受動態
    av = ["can","may","must","have to","should","will","shall","had better","ought to","could","would","might","can't","won't","couldn't","wouldn't"]

    av_index = random.randint(0,len(av)-1)
    BS = base_form()
    bs = BS.to_base(verb,pos)     # 動詞の原形が返ってくる
    pp = get_verb_list(bs)[-2]      # 過去分詞を取得
    form = av[av_index] + " " + "be" + " " + pp

    return form

##################################################################################################
## 現在完了・受動態・進行形・未来形
##################################################################################################
def create_two_more_verb(verb,pos,sub_toks):

    have,be = get_HaveBe(verb,pos,sub_toks)

    BS = base_form()
    bs = BS.to_base(verb,pos)       # 動詞の原形が返ってくる
    pp = get_verb_list(bs)[-2]      # 過去分詞の取得
    ing = get_verb_list(bs)[-3]     # 現在分詞の取得

    form = []                       # 現在完了，受動態，進行形，現在完了受動態を格納するリスト
    form.append(have + " " + pp)    # 現在完了
    form.append(be + " " + pp)      # 受動態
    form.append(be + " " + ing)     # 進行形
    form.append("will "+ bs)        # 未来        
    form.append(have + " " + "been" + " " + pp)     # 現在完了受動態

    return form


def past_or_current(hole_pos):       ## 穴の動詞が現在形か過去形かを判別
    past_v = ['VBD','VBN']
    if hole_pos in past_v:   # 過去
        return "past"
    else:    # 現在
        return "current"

def one_or_twomore(s_pos):      # 主語が単数か複数かを判定
    if s_pos == "NNS" or s_pos == "NNPS":
        return "twomore"    #複数
    else:
        return "one"        #単数

def get_HaveBe(verb,pos,sub_toks):      # 主語や問題の動詞に対する適切なbe動詞，haveの形を取得
    try:
        n_pos = nltk.pos_tag(sub_toks[verb])    # 穴になる動詞の主語に相当する品詞情報を得る
        o_t = one_or_twomore(n_pos[0][1])
    except KeyError:
        o_t = "one"

    have = ""       # have の形を保存
    be = ""         # be 動詞の形を保存
    p_c = past_or_current(pos)
    if p_c == "past":   # 過去
        have = "had"
        if o_t == "one":
            be = "was"    #単数
        else:
            be = "were"   #複数
    else:               # 現在
        if o_t == "one":
            have = "has"    #単数
            be = "is"
        else:
            have = "have"   #複数
            be = "are"
    
    return have,be

if __name__ == "__main__":
    v_list = ['MD','VB','VBD','VBG','VBN','VBP','VBZ']   # 問題にする品詞のリスト(動詞問題)
    # MD:助動詞, VB:動詞の原形, VBD:動詞の過去形, VBG:現在分詞, VBN:過去分詞, VBP:動詞の原形, VBZ:三単元のsつき

    # line = "Two water-quality studies, which were released last week, reflect the hard work of the Water Resource Council to keep local water clean."
    # line = "New patients should arrive fifteen minutes before their scheduled appointments."
    # line = "The final version of the budget proposal must be submitted by Friday."
    # line = "Ms.Choi offers clients both tax preparation services and financial management consultations."
    # line = "Maya Byun was chosen by the excecutive team to head the new public relations department."
    line = "Belvin Theaters will soon allow customers to purchase tickets on its Web site."

    morph = nltk.word_tokenize(line)
    pos = nltk.pos_tag(morph)
    
    hole_word = 'allow'
    hole_pos = ''

    # for w,p in pos:
    #     if w == hole_word:
    #         hole_pos = p

    # print(hole_pos)

    # stanza.download('en') # download English model
    # nlp = stanza.Pipeline('en') # initialize English neural pipeline
    # doc = nlp(line) # run annotation over a sentence

    # sub_toks = {}

    # for sent in doc.sentences:
    #     for word in sent.words:
    #         if word.deprel == "nsubj":
    #             sub_toks[sent.words[word.head-1].text if word.head > 0 else "root"] = [word.text]

    # print(sub_toks)

    # print(create_two_more_verb(hole_word,hole_pos,sub_toks))

