#coding:utf-8
import string
import re

filter_char = string.punctuation 
digit_char = "0123456789"
es_char = u"¿¡"
def filter_chars(word):
    word = word.lower().decode('utf-8')
    word = word.strip(filter_char)
    word = word.strip(digit_char)
    word = word.strip(es_char)
    return word.encode('utf-8')

with open("cikm_english_train_20180516.txt") as fin1, open("cikm_spanish_train_20180516.txt") as fin2, open("cikm_test_a_20180516.txt") as fin3, open("out.txt", 'w') as fp_w:
    for raw_line in fin1:
        line = raw_line.strip("\n\r").split("\t")
        sentence1 = ""
        for word in line[1].split(" "):
            word = filter_chars(word)
            if word != "":
                sentence1 += word + " "

        sentence2 = ""
        for word in line[3].split(" "):
            word = filter_chars(word)
            if word != "":
                sentence2 += word + " "
        
        if len(sentence1) == 0 or len(sentence2) == 0:
            continue
        
        fp_w.write(line[-1] + "\t" + sentence1.strip() + "\001" + sentence2.strip() + "\n")
    
    for raw_line in fin2:
        line = raw_line.strip("\n\r").split("\t")
        sentence1 = ""
        for word in line[0].split(" "):
            word = filter_chars(word)
            if word != "":
                sentence1 += word + " "

        sentence2 = ""
        for word in line[2].split(" "):
            word = filter_chars(word)
            if word != "":
                sentence2 += word + " "

        if len(sentence1) == 0 or len(sentence2) == 0:
            continue

        fp_w.write(line[-1] + "\t" + sentence1.strip() + "\001" + sentence2.strip() + "\n")


    for raw_line in fin3:
        line = raw_line.strip("\n\r").split("\t")
        fp_w.write("-1" + "\t")
        sentence1 = ""
        for word in line[0].split(" "):
            word = filter_chars(word)
            if word != "":
                sentence1 += word + " "

        sentence2 = ""
        for word in line[1].split(" "):
            word = filter_chars(word)
            if word != "":
                sentence2 += word + " "
        fp_w.write(sentence1.strip() + "\001" + sentence2.strip() + "\n")
