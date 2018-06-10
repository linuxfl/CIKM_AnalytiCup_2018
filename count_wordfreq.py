#coding:utf-8
import math
from tqdm import tqdm

words_dict = {}
sum = 0
with open("out.txt") as fin:
    for raw_line in fin:
        line = raw_line.strip("\n\r").split("\t")
        sentens = line[1].strip().split("\001")
        
        for senten in sentens:
            s = senten.split(" ")
            for w in s:
                w = w.strip("¿?,.\"")
                if w not in words_dict:
                    words_dict[w] = 0
                words_dict[w] += 1
            sum += 2
print sum

words = sorted(words_dict.items(), key=lambda x:x[1], reverse=True)
with open("words.txt", "w") as fout:
    for key, value in words:
        fout.write(key +  " " + str(math.log(sum / (float(value) + 1.0))) + "\n")

with open("wiki.es.vec") as fin, open("word2vec.dict", 'w') as fout:
    for raw_line in tqdm(fin):
        line = raw_line.strip("\r\n").split()
        if line[0] in words_dict:
            fout.write(raw_line)

