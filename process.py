#coding:utf-8
fp_w = open("out.txt", 'w')
with open("cikm_english_train_20180516.txt") as fin1, open("cikm_spanish_train_20180516.txt") as fin2, open("cikm_test_a_20180516.txt") as fin3:
    for raw_line in fin1:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[-1] + "\t" + line[1].strip("¿?.,!¡").lower() + '\001' + line[3].strip("¿?.,!¡").lower() + "\n")

    for raw_line in fin2:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[-1] + "\t" + line[0].strip("¿?.,!¡").lower() + '\001' + line[2].strip("¿?.,!¡").lower() + "\n")

    for raw_line in fin3:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write("-1" + "\t" + line[0].strip("¿?.,!¡").lower() + '\001' + line[1].strip("¿?.,!¡").lower() + "\n")

fp_w.close()
