#coding:utf-8
words_dict = {}
sum = 0
with open("out.txt") as fin:
    for raw_line in fin:
        line = raw_line.strip("\n\r").split("\t")
        sentens = line[1].strip().split("\001")
        
        for senten in sentens:
            s = senten.split(" ")
            for w in s:
                w = w.strip("¿?,.")
                if w not in words_dict:
                    words_dict[w] = 0
                words_dict[w] += 1
            sum += 2
print sum

words = sorted(words_dict.items(), key=lambda x:x[1], reverse=True)
with open("words.txt", "w") as fout:
    for key, value in words:
        fout.write(key + " " + str(value) + " " + str(value * 1.0 / sum) + "\n")
