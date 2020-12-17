#TESTS MAX LENGTH
import numpy as np
path = "twitter-datasets\\"
neg = "PREGOOD_train_neg_full.txt"
pos = "PREGOOD_train_pos_full.txt"
a = list()
posC = 0
negC =  0
t = 0
useless = ['onomatopeia', 'abbreviation', 'laugh', 'repetition', 'misspelled']
def preprocess(tweet):
    tweet = tweet.replace('<user','')
    tweet = tweet.replace('<url','')
    for c in useless:
        tweet = tweet.replace(c,"")
    return tweet
with open(path + neg,encoding='utf-8',errors="namereplace") as f:
    for line in f:
        a.append(len(line.split(" ")))
        negC += preprocess(line).count("laugh")
with open(path + pos,encoding='utf-8',errors="namereplace") as f:
    for line in f:
        a.append(len(line.split(" ")))
        posC += preprocess(line).count("laugh")
with open(path+"PREGOOD_test_data.txt",encoding='utf-8',errors="namereplace") as f:
    for line in f:
        a.append(len(line.split(" ")))
        t += preprocess(line).count("laugh")

print(max(a),negC,posC,t)