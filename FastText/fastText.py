import fasttext
import os.path
import numpy as np
def addLabels(full=False):
    path = "twitter-datasets\\"
    if(full):
        neg = "train_neg_full.txt"
        pos = "train_pos_full.txt"
        res = "res_fasttext_full.txt"
    else:
        neg = "train_neg.txt"
        pos = "train_pos_txt"
        res = "res_fasttext.txt"
    
    if(os.path.isfile(path+res)):
        return
    else:
        try:
            fileRes = open(path+res,"w")
            with open(path+pos) as f:
                for neg_line in f:
                    fileRes.write("__label__0 "+neg_line+"\n") # DO I NEED \n ???????????????
            with open(path+pos) as f:
                for pos_line in f:
                    fileRes.write("__label__1 "+pos_line+"\n") # DO I NEED \n ???????????????
        except :
            print("Error occured")
            if(os.path.isfile(path+res)):
                os.remove(path+res)
        finally:
            fileRes.close()

if __name__ == "__main__":
    full = True
    addLabels(full)
    path = "twitter-datasets\\"
    if(full):
        res = "res_fasttext_full.txt"
    else:
        res = "res_fasttext.txt"
    model = fasttext.train_supervised(path+res)
    




addLabels(True)