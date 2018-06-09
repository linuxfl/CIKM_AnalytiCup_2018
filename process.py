#coding:utf-8

def filter_chars(words):
    #return words.strip("¿?.,!¡").lower()
    return words.lower().replace("¿", "").replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace("¡", "")

with open("cikm_english_train_20180516.txt") as fin1, open("cikm_spanish_train_20180516.txt") as fin2, open("cikm_test_a_20180516.txt") as fin3, open("out.txt", 'w') as fp_w:
    for raw_line in fin1:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[-1] + "\t")
        sentence = ""
        for word in line[1].split(" "):
            sentence += filter_chars(word) + " "
        fp_w.write(sentence.strip() + "\001")

        sentence = ""
        for word in line[3].split(" "):
            sentence += filter_chars(word) + " "
        fp_w.write(sentence.strip() + "\n")
    
    for raw_line in fin2:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write(line[-1] + "\t")
        sentence = ""
        for word in line[0].split(" "):
            sentence += filter_chars(word) + " "
        fp_w.write(sentence.strip() + "\001")

        sentence = ""
        for word in line[2].split(" "):
            sentence += filter_chars(word) + " "
        fp_w.write(sentence.strip() + "\n")

    for raw_line in fin3:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write("-1" + "\t")
        sentence = ""
        for word in line[0].split(" "):
            sentence += filter_chars(word) + " "
        fp_w.write(sentence.strip() + "\001")

        sentence = ""
        for word in line[1].split(" "):
            sentence += filter_chars(word) + " "
        fp_w.write(sentence.strip() + "\n")
