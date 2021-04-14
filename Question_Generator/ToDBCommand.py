import subprocess
import time

step = 1000

for i in range(1,6):
    file_name = "./data/a" + str(i) + "_train_corr_sentences.txt"
    for j in range(1,1000000,step):
        write_file = "a" + str(step)
        cmd = "cat "+ file_name + " | head -n " + str(j) + " | tail -n " + str(1000) + " > ./data/" + write_file + ".txt"
        proc = subprocess.Popen(
            cmd,
            shell = True,
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE
        )
        proc2 = subprocess.Popen(['python','derivative_morpheme.py'])
        time.sleep(2)
        proc3 = subprocess.Popen(['python','Json2DB.py'])
        print("{}番目：{}".format(i,j))
