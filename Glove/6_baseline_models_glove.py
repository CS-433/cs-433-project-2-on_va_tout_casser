"""
Before using this, make sure to download from https://nlp.stanford.edu/projects/glove/ the glove.twitter.27B.zip 
and unzip it in the twitter-datasets folder !

PS: you need quite a lot of RAM :)

"""






from IPython.core.debugger import set_trace
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from keras import Constant,Embedding, LSTM, Dense, Dropout,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.initializers import Constant
import re
import gc
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from datetime import datetime
import os.path
def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)



def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]

    return " ".join(text)


# ### Embeddings

# #### Using GloVe

# https://nlp.stanford.edu/projects/glove/



def create_corpus_tk(train):
    corpus = []
    for text in train:
        words = [word.lower() for word in word_tokenize(text)]
        corpus.append(words)
    return corpus
def get_train_labels_with_test(train_percentage):
    path = "twitter-datasets\\"
    #train_sentences,train_labels,test_sentences,test_labels = list(),list(),list(),list()
    train = list()
    labels = list()
    with open(path + "train_pos_full.txt",encoding='utf-8',errors="namereplace") as f :
        for pos_line in f:
            train.append(pos_line.replace("\n"," "))
            labels.append(1)
    train_size_pos = int(len(labels)* train_percentage)
    train_sentences = train[:train_size_pos]
    train_labels = labels[:train_size_pos]
    test_sentences = train[train_size_pos:]
    test_labels = labels[train_size_pos:]
    trained_size = len(train)
    labels = list()
    with open(path + "train_neg_full.txt",encoding='utf-8',errors="namereplace") as f :
        for neg_line in f:
            train.append(neg_line.replace("\n"," "))
            labels.append(0)
    train_size_neg = int((len(labels))* train_percentage)
    
    train_sentences.extend(train[trained_size:train_size_neg+trained_size])
    train_labels.extend(labels[:train_size_neg])
    test_sentences.extend(train[train_size_neg+trained_size:])
    test_labels.extend(labels[train_size_neg:])
    return train,np.array(train_sentences),np.array(train_labels),np.array(test_sentences),np.array(test_labels)
# def get_train_label():
#     assert False #make sure you never use this method again
#     path = "twitter-datasets\\"
#     train = list()
#     labels = list()
#     with open(path + "train_pos_full.txt",encoding='utf-8',errors="namereplace") as f :
#         for pos_line in f:
#             train.append(pos_line)
#             labels.append(1)
#     with open(path + "train_neg_full.txt",encoding='utf-8',errors="namereplace") as f :
#         for neg_line in f:
#             train.append(neg_line)
#             labels.append(0)
#     return train,labels
def load_test_data():
    path = "twitter-datasets\\"
    test = list()
    indices = list()
    with open(path + "test_data.txt",encoding='utf-8',errors="namereplace") as f :
        for test_line in f:
            sep = test_line.find(",")
            id_ = test_line[0:sep]
            tweet = test_line[sep+1:]
            test.append(tweet)
            indices.append(id_)
    return test,indices

plt.style.use(style="seaborn")

path_to_npy = "twitter-datasets\\NUMPY_TEMPORARY_FILE.npy"
if(not os.path.isfile(path_to_npy)):
    total_train,train_sentences,train_labels,test_sentences,test_labels = get_train_labels_with_test(0.85)
    corpus = create_corpus_tk(total_train) # takes approx. 10 minutes in full or <1 min in not full
    total_train = list() # let the garbage collector free some RAM
    with open(path_to_npy,"wb") as f:
        np.save(f,train_sentences)
        np.save(f,train_labels)
        np.save(f,test_sentences)
        np.save(f,test_labels)
        np.save(f,np.array(corpus))
else:
    print("No need")
    with open(path_to_npy,"rb") as f:
        train_sentences = np.load(f)
        train_labels= np.load(f)
        test_sentences = np.load(f)
        test_labels = np.load(f)
        corpus = np.load(f,allow_pickle=True)
        corpus = corpus.tolist()


print("Number of total tweets ",len(train_sentences), len(test_sentences))
num_words = len(corpus)
print(num_words,flush = True)
max_len = 280 #max tweet length

# 157 is average #words per tweet + 2* its standard deviation


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_sentences)


train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(
    train_sequences, maxlen=max_len, truncating="post", padding="post"
)
# train_padded

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(
    test_sequences, maxlen=max_len, padding="post", truncating="post"
)
test_sequences,train_sequences = None,None
train_sentences,test_sentences = None,None # let the garbage collector free some RAM
gc.collect()
# test_padded
word_index = tokenizer.word_index
print("Number of unique words:", len(word_index))
##################################################################################################
# test_data,indices = load_test_data()
# sequences = tokenizer.texts_to_sequences(test_data)
# padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
# print("padded is",padded,len(padded[0]),flush=True)
# assert False
###################################################################################################
print("Begin to construct embedding_dict",flush=True)
dimension = 100
embedding_dict = {}
path = "twitter-datasets\\"
with open(path+"glove.twitter.27B\\glove.twitter.27B."+str(dimension)+"d.txt", "r",encoding="utf-8",errors="namereplace") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], "float32")
        embedding_dict[word] = vectors
f.close()
print("Finish to construct embedding_dict",flush=True)

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, dimension))

for word, i in word_index.items():
    if i < num_words:
        emb_vec = embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i] = emb_vec

# embedding_matrix
# word_index["reason"]
# embedding_dict.get("reason")
# (embedding_matrix[696] == embedding_dict.get("reason")).all()
model = Sequential()

model.add(
    Embedding(
        num_words,
        dimension,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_len,
        trainable=False,
    )
)
model.add(LSTM(dimension, dropout=0.1))
model.add(Dense(1,activation="relu")) # try with relu
#model.add(Dense(1, activation="sigmoid"))


optimizer = Adam()#learning_rate=3e-4)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(
    train_padded,
    train_labels,
    epochs=2, # TODO : INCREASE EPOCH, IT IS NOW LOW FOR TESTING PURPOSES - ~ 9 MIN / EPOCH IN NON FULL VS 4H IN FULL
    validation_data=(test_padded, test_labels),
    verbose=1,
    batch_size = 16,
     use_multiprocessing = False,
)


#sequences = tokenizer.texts_to_sequences(test.text)
train_labels,test_labels = None,None # let the garbage collector free some RAM
gc.collect()
test_data,indices = load_test_data()
sequences = tokenizer.texts_to_sequences(test_data)
padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
print("padded is",padded,flush=True)
pred = model.predict(padded)
pred_int = pred.round().astype("int")

print("prediction is",pred)
print("type of pred is ",type(pred))

print("assert equation ",indices[len(indices)-1],len(pred))
try:
    resFile = open("submission_GLOVE_FULL_"+"dim"+str(dimension)+"_"+str(datetime.now()).replace(" ","__").replace(":","-")+".csv","w")
    resFile.write("Id,Prediction\n")
    for i in range(len(pred_int)):
        predicted = pred_int[i]
        if(predicted == 0):
            predicted = -1
        elif(predicted != 1):
            print("Prediction type error on ",predicted)
        resFile.write(str(indices[i])+","+str(predicted)+"\n")
except :
    print("Error encountered, try again")
finally:
    resFile.close()

print("Rounded prediction is ",pred_int)


# padded[5]


# model.predict(padded[5].reshape(1, -1))


# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# def decode(text):
    # return " ".join([reverse_word_index.get(i, "?") for i in text])
# 

# decode(sequences[5])