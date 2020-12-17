import fasttext
import os.path
import numpy as np
from datetime import datetime
import string


def addLabels(full=False):
    train_percentage = 0.9
    path = "twitter-datasets\\"
    if(full):
        neg = "neg_without_tf_idf_and_spellcheck_other_tok.txt"
        pos = "pos_without_tf_idf_and_spellcheck_other_tok.txt"
        res = "pos_without_tf_idf_and_spellcheck_other_tok_res_fasttext_full.txt"
        valid = "pos_without_tf_idf_and_spellcheck_other_tok_valid_fasttext_full.txt"
    else:
        neg = "train_neg.txt"
        pos = "train_pos.txt"
        res = "res_fasttext.txt"
    
    if(os.path.isfile(path+res) and os.path.isfile(path + valid)):
        return
    else:
        # try:
            num_pos = sum(1 for line in open(path+pos,encoding="utf-8",errors="namereplace"))
            num_pos = int(num_pos * train_percentage)
            num_neg = sum(1 for line in open(path+neg,encoding='utf-8',errors="namereplace"))
            num_neg = int(num_neg * train_percentage)
            fileRes = open(path+res,"w",errors="namereplace")
            fileValid = open(path+valid,"w",errors="namereplace")
            print("pos ",num_pos,"neg",num_neg)
            with open(path+neg,encoding='utf-8',errors="namereplace") as f:
                y = 0
                for neg_line in f:
                    # try:
                    #tweet = tweet.replace("\n","")
                    if(y <num_neg):
                        fileRes.write("__label__0 "+(neg_line))
                    else:
                        fileValid.write("__label__0 "+(neg_line))
                    # except UnicodeDecodeError:
                    #     print("Got an error",flush=True)
                    #     continue
                    y +=1
            with open(path+pos,encoding="utf-8",errors="namereplace") as f:
                #print("I pos line")
                i = 0 
                for pos_line in f:
                    #print("I pos line2 ",flush=True)
                   # print("writing ","__label__1 "+preprocess(tweet))
                    if(i < num_pos):
                        fileRes.write("__label__1 "+(pos_line))
                    else:
                        fileValid.write("__label__1 "+(pos_line))
                    i+=1
        # except Exception :
        #     print("Error occured",flush=True)
        #     if(os.path.isfile(path+res)):
        #         fileRes.close()
        #         os.remove(path+res)
        # finally:
        #     if(not fileRes.closed()):
        #         fileRes.close()

if __name__ == "__main__":
    full = True
    addLabels(full)
    
    path = "twitter-datasets\\"
    if(full):
        res = "pos_without_tf_idf_and_spellcheck_other_tok_res_fasttext_full.txt"
        valid = "pos_without_tf_idf_and_spellcheck_other_tok_valid_fasttext_full.txt"
    else:
        res = "fin_res_fasttext.txt"
        valid = "valid_fasttext_full.txt"
    print("begin",flush=True)
    model = fasttext.train_supervised(input = path+res,autotuneValidationFile = path + valid)
    print("finish",flush=True)
    print(model.test(path + valid)+ "####################################################################################################",flush=True)
    
    # try:
    #     resFile = open("ETI_submission_"+str(full)+str(datetime.now()).replace(" ","__").replace(":","-")+".csv","w")
    #     resFile.write("Id,Prediction\n")
    #     with open(path+"test_data.txt") as f:
    #         for line in f:
    #             sep = line.find(",")
    #             id_ = line[0:sep]
    #             tweet = line[sep+1:]
    #             tweet = tweet.replace("\n","")
    #             tweet = (tweet)
    #             pred = model.predict(tweet)
    #             if(pred[0][0] == "__label__0"):
    #                 pred = -1
    #             elif(pred[0][0]=="__label__1"):
    #                 pred = 1
    #             else:
    #                 print("Not pred as predicted ! ",pred[0],pred[0][0])
    #             resFile.write(str(id_)+","+str(pred)+"\n")
    # finally:
    #     resFile.close()