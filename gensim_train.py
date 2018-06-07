#coding:utf-8
import pandas as pd
import numpy as np
import jieba, gensim
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
doc2vec_size = 300

def read_data(infile):
    sentences = []
    for raw_line in open(infile):
        line = raw_line.strip("\n\r")
        sentences.append(line)
    return sentences

def read_corpus(data):
    for i, line in enumerate(data):
        # split with space to isolate each word
        # the words list are tagged with a label as its identity
        yield gensim.models.doc2vec.TaggedDocument(line.split(), ['%s' % i])

def doc2vec_train(data):
    train_corpus = list(read_corpus(data))
    
    model = gensim.models.doc2vec.Doc2Vec(vector_size=doc2vec_size, min_count=1, epochs=50, workers=7)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save("model.dat")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python generate_ins.py infile"
        exit()

    print "Read data and word segment ..."
    data = read_data(sys.argv[1])
    print "doc2vec train ..."
    doc2vec_train(data)
    print "doc2vec finished ..."
