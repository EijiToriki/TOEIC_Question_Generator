import psycopg2

#file_name = "./articles1.txt" 
file_name = "./toeic_sentense_part5.txt" 
# connect postgreSQL
users = 'postgres' # initial user
dbnames = 'news' #your own DB name
passwords = 'your own password'
conn = psycopg2.connect(" user=" + users +" dbname=" + dbnames +" password=" + passwords)
# excexute sql
cur = conn.cursor()

count = 1
with open(file_name,'r',encoding="cp932") as f:
    for line in f:
        line = line.replace("'","''")
        line = line.replace("\n","")
        cur.execute("insert into news(news_id,news_txt)values(" + str(count) +",'"+ line + "')")
        count = count + 1

conn.commit()

cur.close()
conn.close()
