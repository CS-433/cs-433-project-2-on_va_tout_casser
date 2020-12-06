import os
import shutil
import numpy as np
import nltk 
nltk.download('wordnet')
import string
import gensim
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from tensorflow.keras import layers
import bert
import random
import math
from datetime import datetime


import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

path = 'dataset/'
test_path = path + "test_data.txt"
test_pickle_path = path + 'test_pickle_tweets.txt'


def line_cleaned(line, label):
    #test data
    if label == 5:
        sep = line.find(",")
        line = line[sep+1:]
    line = line.replace("\n","")
    line = line.replace('<user>','')
    line = line.replace('<url>','')
    line = "".join([char for char in line if char not in string.punctuation])
    
    return line

def set_data_path(full):
    #check if right place in terminal when launching
    if full:
        return path+'train_neg_full.txt', path+'train_pos_full.txt', path+'train_full_pickle_labels_bert.txt', path+'train_full_pickle_tweets_bert.txt'
    else:
        return path+'train_neg.txt',  path+'train_pos.txt', path+'train_pickle_labels_bert.txt',path+'train_pickle_tweets_bert.txt'


def tweets_to_clean_tweets(path, label):
    count = 0
    tweets_labels = []
    tweets_cleaned = []
    print("extracting {} dataset...".format(path))
    with open(path) as f:
        for line in f:
            tweets_cleaned.append(line_cleaned(line, label))
            tweets_labels.append(label)
            count += 1
            if(count % 10000 == 0):
                print(" {} tweets extracted".format(count))
    return tweets_labels, tweets_cleaned
        

def clean_tweets(full, use_pickle):
    neg_path, pos_path, pickle_labels_path, pickle_tweets_path = set_data_path(full)

    tweets_labels = []
    tweets_cleaned = []
    tweets_test_cleaned =[]
    if use_pickle:
        with open(pickle_labels_path, "rb") as fp1:
            tweets_labels = pickle.load(fp1)
        with open(pickle_tweets_path, "rb") as fp2:
            tweets_cleaned = pickle.load(fp2)
        with open(test_pickle_path, "rb") as fp3:
            tweets_test_cleaned = pickle.load(fp3)
        
        return tweets_labels, tweets_cleaned, tweets_test_cleaned
    
    print("extracting tweets_data...")
    tweets_labels_pos, tweets_tokenized_pos = tweets_to_clean_tweets(pos_path, 1)
    tweets_labels_neg, tweets_tokenized_neg = tweets_to_clean_tweets(neg_path, 0) #here = 0 !!!
    _, tweets_test_cleaned = tweets_to_clean_tweets(test_path, 5) 
    print("extracting tweets terminated")

    tweets_cleaned = tweets_tokenized_pos + tweets_tokenized_neg
    tweets_labels = tweets_labels_pos + tweets_labels_neg

    with open(pickle_labels_path, "wb") as fp1:
        pickle.dump(tweets_labels, fp1)
    with open(pickle_tweets_path, "wb") as fp2:
        pickle.dump(tweets_cleaned, fp2)
    with open(test_pickle_path, "wb") as fp3:
        pickle.dump(tweets_test_cleaned, fp3)

    return tweets_labels, tweets_cleaned, tweets_test_cleaned

def tokenize_tweet(tokenizer, tweets):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets))

def tokenize_tweets(tweets, tweets_test, use_pickle):
    if use_pickle:
        with open(path + "tokenized_full.txt", "rb") as fp1:
            tokenized = pickle.load(fp1)
        with open(path + "tokenized_test_full.txt", "rb") as fp2:
            tokenized_test = pickle.load(fp2)
        with open(path + "len_tokenized_full.txt", "rb") as fp3:
            length = pickle.load(fp3)
        
        return tokenized, tokenized_test, length


    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                  trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    tokenized = [tokenize_tweet(tokenizer, tweet) for tweet in tweets]
    tokenized_test = [tokenize_tweet(tokenizer, tweet) for tweet in tweets_test]

    with open(path + "tokenized_full.txt", "wb") as fp1:
        pickle.dump(tokenized, fp1)
    with open(path + "tokenized_test_full.txt", "wb") as fp2:
        pickle.dump(tokenized_test, fp2)
    with open(path + "len_tokenized_full.txt", "wb") as fp3:
        pickle.dump(len(tokenizer.vocab), fp3)

    return tokenized, tokenized_test, len(tokenizer.vocab)
    
    
class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3) 
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output


if __name__ == '__main__':
    full_data = True
    use_pickle = True
    tweets_labels, tweets_cleaned, tweets_test_cleaned = clean_tweets(full=full_data, use_pickle=use_pickle)
    

    # tweets = np.array(tweets_cleaned)
    y = np.array(tweets_labels)
   
    # tweets_test = np.array(tweets_test_cleaned)
    use_pickle = True
    token_tweets, token_tweets_test, len_tokenizer_vocab = tokenize_tweets(tweets_cleaned, tweets_test_cleaned, use_pickle)

    tweets_with_len = [[tweet, y[i], len(tweet)]for i, tweet in enumerate(token_tweets)]
    #sort by length
    print("tokenized terminated")
    tweets_with_len.sort(key=lambda x: x[2])

    #remove length attribute
    sorted_tweets_labels = [(tweet_lab[0], tweet_lab[1]) for tweet_lab in tweets_with_len]

    tweets_final, labels_tuple = zip(*sorted_tweets_labels)

    X = tf.keras.preprocessing.sequence.pad_sequences(tweets_final)
    y = np.asarray(labels_tuple)
    X_test_final = tf.keras.preprocessing.sequence.pad_sequences(token_tweets_test, maxlen=X.shape[1])

    index_list = list(range(y.shape[0]))
    random.seed(4)
    random.shuffle(index_list)
    y = y[index_list]
    X = X[index_list,:]

    validation_percentage = 0.85
    break_point = int(y.shape[0] * validation_percentage)
    X_train = X[0:break_point,:]
    X_validation = X[break_point:,:]
    y_train = y[0:break_point]
    y_validation = y[break_point:]
    
    
    
    batch_size = 64#64
    VOCAB_LENGTH = len_tokenizer_vocab
    EMB_DIM = 25#200
    CNN_FILTERS = 100
    DNN_UNITS = 256
    DROPOUT_RATE = 0.2
    NB_EPOCHS = 10

    text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=2,
                        dropout_rate=DROPOUT_RATE)


    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
    


    text_model.fit(X_train,y_train,  
                    validation_data=(X_validation, y_validation),
                    batch_size = batch_size,
                    epochs=NB_EPOCHS)
    text_model.save('classifiers/bert')

    pred = text_model.predict(X_test_final)
    pred_int = pred.round().astype("int")

    try:
        resFile = open("sub_BERT_"+"_batch_size_"+str(batch_size)+"_epochs_"+str(NB_EPOCHS)+"___"+str(datetime.now()).replace(" ","__").replace(":","-")+".csv","w")
        resFile.write("Id,Prediction\n")
        for i in range(len(pred_int)):
            predicted = pred_int[i]
            if(predicted == 0):
                predicted = -1
            elif(predicted != 1):
               print("Prediction type error on ",predicted)
            resFile.write(str(i + 1)+","+str(int(predicted))+"\n")
    except :
        print("Error encountered, try again")
    finally:
        resFile.close()

    
    

    # try:
    #     resFile = open("submission_BERT_"+"dim"+str(EMB_DIM)+"_"+".csv","w")
    #     resFile.write("Id,Prediction\n")
    #     for i in range(X_test_final.shape[0]):
    #         pred = text_model.predict(X_test_final[i,:])
    #         if(pred == 0):
    #             pred = -1
    #         resFile.write(str(i + 1) + "," + str(pred) + "\n")
    # finally:
    #     resFile.close()

  
    print("terminated")