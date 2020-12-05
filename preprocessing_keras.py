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


import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')



def preprocessing(train_path_pos,train_path_neg, test_path, batch_size, validation_split, seed):
    print()

def bert_encoder():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
    trainable=True)

    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]      # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

path = 'dataset/'
test_path = path + "test_data.txt"
test_pickle_path = path + 'test_pickle_tweets.txt'
google_vector_size = 300

def line_cleaned(line):
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
            tweets_cleaned.append(line_cleaned(line))
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
    _, tweets_test_cleaned = tweets_to_clean_tweets(test_path, 2) 
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

def tokenize_tweets(tweets, tweets_test):
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                  trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    tokenized = [tokenize_tweet(tokenizer, tweet) for tweet in tweets]
    tokenized_test = [tokenize_tweet(tokenizer, tweet) for tweet in tweets_test]
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

    token_tweets, token_tweets_test, len_tokenizer_vocab = tokenize_tweets(tweets_cleaned, tweets_test_cleaned)

    tweets_with_len = [[tweet, y[i], len(tweet)]for i, tweet in enumerate(token_tweets)]
    random.shuffle(tweets_with_len)
    #sort by length
    tweets_with_len.sort(key=lambda x: x[2])

    #remove length attribute
    sorted_tweets_labels = [(tweet_lab[0], tweet_lab[1]) for tweet_lab in tweets_with_len]

    #transform in tensor dataset
    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_tweets_labels, output_types=(tf.int32, tf.int32))

    batch_size = 32
    batched_dataset = processed_dataset.padded_batch(batch_size, padded_shapes=((None, ), ()))

    #split data to get 10% of test data
    total_batch = math.ceil(len(sorted_tweets_labels) / batch_size)
    test_batch = total_batch // 8
    batched_dataset.shuffle(total_batch)
    test_data = batched_dataset.take(test_batch)
    train_data = batched_dataset.skip(test_batch)

    VOCAB_LENGTH = len_tokenizer_vocab
    EMB_DIM = 200
    CNN_FILTERS = 100
    DNN_UNITS = 256
    OUTPUT_CLASSES = 2

    DROPOUT_RATE = 0.2

    NB_EPOCHS = 5

    text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=2,
                        dropout_rate=DROPOUT_RATE)


    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
    

    text_model.fit(train_data, epochs=NB_EPOCHS)

    with open("bert model", "wb") as fp5:
        pickle.dump(text_model, fp5)
    results = text_model.evaluate(test_data)
    print(results)
    # print("tokenizing...")
    # bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')


    # tweets_together = list(tweets)+ list(tweets_test)

    # tweets_preprocessed = bert_preprocess_model(tweets_together)

    # print(tweets_preprocessed.shape)
    print("terminated")