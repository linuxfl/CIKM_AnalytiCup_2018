#coding:utf-8
import pandas as pd
import numpy as np
import jieba, gensim
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
doc2vec_size = 300

def read_data(infile):
    train_data = pd.read_csv(infile, names=['label', 'c1'], sep='\t')
    print train_data
    return train_data

def read_corpus(data):
    for i, line in enumerate(data['c1']):
        # split with space to isolate each word
        # the words list are tagged with a label as its identity
        yield gensim.models.doc2vec.TaggedDocument(line.split(), ['%s' % i])

def doc2vec_train(data):
    train_corpus = list(read_corpus(data))
    
    model = gensim.models.doc2vec.Doc2Vec(vector_size=doc2vec_size, min_count=1, epochs=50, workers=7)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    return model    

def write_train_ins(model, data, out_file):
    f_out = open(out_file, 'w')
    docvec = []
    
    ind = np.arange(data.shape[0])
    #np.random.shuffle(ind)
    for i in ind:
        row = []
        f_out.write(str(data['label'][i]) + " ")
        for idx in range(doc2vec_size):
            f_out.write(str(model[i][idx]) + " ")
        f_out.write("\n")
    f_out.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: python generate_ins.py infile train_ins"
        exit()
    print "Read data and word segment ..."
    data = read_data(sys.argv[1])
    print "doc2vec train ..."
    model = doc2vec_train(data)
    print "doc2vec finished ..."
    print "write train instance to file ..."
    write_train_ins(model, data, sys.argv[2])
