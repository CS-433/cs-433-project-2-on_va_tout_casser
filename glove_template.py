#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import glove
from gensim.models import word2vec
import embeddings
import fasttext
def main():
    print("loading cooccurrence matrix")
    with open('twitter-datasets\cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10
    last = 3
    print(cooc.row)
    print(cooc.col)
    print(cooc.data,len(list(zip(cooc.row, cooc.col, cooc.data))))
    return
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            #if( ix < last):
               # print(ix,jy,n)
            #last = ix
            print(ix,jy,n,len(cooc.row),len(cooc.col),len(cooc.data))
            # Left : is the line, i.e. which tweet it is
            # Middle : is the word number of this tweet
            # Right : is the number of occurences of this word over all tweets
			# fill in your SGD code here, 
			# for the update resulting from co-occurence (i,j)
		

    np.save('embeddings', xs)


if __name__ == '__main__':
    main()
