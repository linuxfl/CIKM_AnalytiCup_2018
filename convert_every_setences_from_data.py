#coding:utf-8
fp_w = open("corpus.txt", 'w')
with open("cikm_english_train_20180516.txt") as fin1, open("cikm_spanish_train_20180516.txt") as fin2, open("cikm_test_a_20180516.txt") as fin3, open("cikm_unlabel_spanish_train_20180516.txt") as fin4:
    for raw_line in fin1:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[1].strip("¿?.,!") + '\n' + line[3].strip("¿?.,!") + "\n")

    for raw_line in fin2:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[0].strip("¿?.,!") + '\n' + line[2].strip("¿?.,!") + "\n")

    for raw_line in fin3:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[0].strip("¿?.,!") + '\n' + line[1].strip("¿?.,!") + "\n")

    for i, raw_line in enumerate(fin4):
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[0].strip("¿?.,!") + "\n")

fp_w.close()
