
fp_w = open("out.txt", 'w')
with open("cikm_english_train_20180516.txt") as fin1, open("cikm_spanish_train_20180516.txt") as fin2, open("cikm_test_a_20180516.txt") as fin3, open("cikm_unlabel_spanish_train_20180516.txt") as fin4:
    for raw_line in fin1:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[-1] + "\t" + line[1] + ' ' + line[3] + "\n")

    for raw_line in fin2:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[-1] + "\t" + line[0] + ' ' + line[2] + "\n")

    for raw_line in fin3:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write("-1" + "\t" + line[0] + ' ' + line[1] + "\n")

    for i, raw_line in enumerate(fin4):
        line = raw_line.strip("\n\r").split("\t")
        write_str = "-2\t"+line[0]+" "
        if i % 2 == 0:
            fp_w.write(write_str)
        else:
            fp_w.write(write_str.strip("-2\t") + "\n")

fp_w.close()
