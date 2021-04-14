import nltk
from m2_to_txt import m2_to_txt

def make_dict(correct,error):
    cor_words = nltk.word_tokenize(correct)
    err_words = nltk.word_tokenize(error)
    d = {}
    list_len = 0
    max_len = 0
    cc = -1
    ec = -1
    tmp_cc = 0
    insert_flag = 1 # 0:文字脱落，1:文字挿入

    if len(cor_words) > len(err_words):
        list_len = len(err_words)
    else:
        list_len = len(cor_words)

    for _ in range(list_len):
        ec += 1                         #ec = error_count
        cc += 1                         #cc = correct_count
        insert_flag = 1

        try:
            eq = cor_words[cc] != err_words[ec]
        except IndexError:
            return d
        
        if eq:
            if insert_flag == 1:
                ## 文字の挿入をチェック
                tmp_ec = ec
                err_word = err_words[ec]
                for j in range(ec,list_len):
                    if cor_words[cc] == err_words[ec]:
                        d[err_word] =   err_words[ec]
                        break
                    ec+=1
                    try:
                        err_word = err_word + " " + err_words[ec]
                    except IndexError:
                        None
                if ec >= list_len:
                    ec = tmp_ec 
                    insert_flag = 0
            if insert_flag == 0:
                ## 文字脱落をチェック
                tmp_cc = cc
                cor_word = cor_words[cc]
                for j in range(cc,list_len):
                    if cor_words[cc] == err_words[ec]:
                        d[cor_word] = err_words[ec]
                        break
                    cc+=1
                    try:
                        cor_word = cor_word + " " + cor_words[cc]
                    except IndexError:
                        None
                if cc >= list_len:
                    cc = tmp_cc
                    d[cor_words[cc]] = err_words[ec]

    return d


def g_and_e(g,e):
    ge = 0
    for k_g,v_g in g.items():
        for k_e,v_e in e.items():
            if k_g == k_e and v_g == v_e:
                ge += 1
    
    return ge

def R(g_list,e_list):
    bunshi, bunbo = 0, 0
    for g,e in zip(g_list,e_list):
        bunshi += g_and_e(g,e)
        bunbo += len(g)
    return bunshi/bunbo

def P(g_list,e_list):
    bunshi, bunbo = 0, 0
    for g,e in zip(g_list,e_list):
        bunshi += g_and_e(g,e)
        bunbo += len(e)

    return bunshi/bunbo

def calculate_f05(g_list,e_list):
    r = R(g_list,e_list)
    p = P(g_list,e_list)
    f05 = (1.25*r*p)/(r+0.5**2*p)
    return f05

if __name__ == "__main__":
    g_list, e_list = [], []

    # g の作成例
    m2 = m2_to_txt("./dataset/test2013.m2")
    m2.create_list_of_er_cor()

    for c_sentence,e_sentence in zip(m2.correct_sentence,m2.error_sentence):
        g = make_dict(c_sentence,e_sentence)
        print(g)

    # e の作成例
    # 逆GECができたら作る

    f05 = calculate_f05(g_list,e_list)
    print(f05)