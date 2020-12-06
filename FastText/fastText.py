import fasttext
import os.path
import numpy as np
from datetime import datetime
import string
def preprocess(tweet):
    tweet = tweet.replace('<user>','')
    tweet = tweet.replace('<url>','')
    tweet = "".join([char for char in tweet if char not in string.punctuation])
    return tweet



def addLabels(full=False):
    path = "twitter-datasets\\"
    if(full):
        neg = "train_neg_full.txt"
        pos = "train_pos_full.txt"
        res = "res_fasttext_full.txt"
    else:
        neg = "train_neg.txt"
        pos = "train_pos.txt"
        res = "res_fasttext.txt"
    
    if(os.path.isfile(path+res)):
        return
    else:
        try:
            fileRes = open(path+res,"w",errors="namereplace")
            with open(path+neg,encoding='utf-8',errors="namereplace") as f:
                for neg_line in f:
                    # try:
                    fileRes.write("__label__0 "+preprocess(neg_line))
                    # except UnicodeDecodeError:
                    #     print("Got an error",flush=True)
                    #     continue
            with open(path+pos,encoding="utf-8",errors="namereplace") as f:
                for pos_line in f:
                    fileRes.write("__label__1 "+preprocess(pos_line))
        except Exception :
            print("Error occured",flush=True)
            if(os.path.isfile(path+res)):
                fileRes.close()
                os.remove(path+res)
        finally:
            if(not fileRes.closed()):
                fileRes.close()

if __name__ == "__main__":
    full = True
    addLabels(full)
    
    path = "twitter-datasets\\"
    if(full):
        res = "res_fasttext_full.txt"
    else:
        res = "res_fasttext.txt"
    model = fasttext.train_supervised(input = path+res,epoch = 25)
    try:
        resFile = open("submission_"+str(full)+str(datetime.now()).replace(" ","__").replace(":","-")+".csv","w")
        resFile.write("Id,Prediction\n")
        with open(path+"test_data.txt") as f:
            for line in f:
                sep = line.find(",")
                id_ = line[0:sep]
                tweet = line[sep+1:]
                tweet = tweet.replace("\n","")
                tweet = preprocess(tweet)
                pred = model.predict(tweet)
                if(pred[0][0] == "__label__0"):
                    pred = -1
                elif(pred[0][0] == "__label__1"):
                    pred = 1
                else:
                    print("Not pred as predicted ! ",pred[0],pred[0][0])
                resFile.write(str(id_)+","+str(pred)+"\n")
    finally:
        resFile.close()