import nltk 
nltk.download('wordnet')
import string
import gensim
import pickle
import gensim.models.doc2vec as qwe
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

path = 'dataset/'
path_results = 'doc2vec_results/'
no_label = 5


def clean_line(line):
    line = line.replace("\n","")
    line = line.replace('<user>','')
    line = line.replace('<url>','')
    line = "".join([char for char in line if char not in string.punctuation])
    words = nltk.tokenize.word_tokenize(line)
    return words


def data_path(full):
    if full:
        return path+'train_neg_full.txt', path+'train_pos_full.txt', path+'test_data.txt'
    else:
        return path+'train_neg.txt',  path+'train_pos.txt', path+'test_data.txt'



def raw_to_cleaned_tweets(path, label):
    count = 0
    tweets_labels = []
    tweets_cleaned = []
    print("extracting {} dataset...".format(path))
    with open(path) as f:
        for line in f:
            if label == no_label:
                sep = line.find(",")
                line = line[sep+1:]

            tweets_cleaned.append(clean_line(line))
            tweets_labels.append(label)
            count += 1
            if(count % 100000 == 0):
                print(" {} tweets extracted".format(count))
    return tweets_labels, tweets_cleaned



def model_to_X(model_dm, model_dbow, num_samples):
    X = np.zeros((num_samples, vector_size))
    for i in range(num_samples):
        X[i, :] = np.append(model_dm.docvecs[i], model_dbow.docvecs[i])
    return X

#TODO: explain, https://radimrehurek.com/gensim/models/doc2vec.html
# from the paper https://arxiv.org/pdf/1405.4053v2.pdf
def doc2vec(tweets_tokenized, tweets_test_tokenized, vector_size, window_size, epochs, seed):
    docs = [TaggedDocument(doc, [tag]) for tag, doc in enumerate(tweets_tokenized + tweets_test_tokenized)]
    print("training doc2vec PV_DM model...")

    model_dm =  Doc2Vec(docs,
                    dm=1, 
                    vector_size=vector_size, 
                    window=window_size, 
                    seed=seed,
                    epochs=epochs,
                    workers=multiprocessing.cpu_count())
    print("training doc2vec PV_DM model terminated")

    print("training doc2vec PV-DBOW model...")
    model_dbow =  Doc2Vec(docs,
                    dm=0, 
                    vector_size=vector_size, 
                    window=window_size, 
                    seed=seed,
                    epochs=epochs,
                    workers=multiprocessing.cpu_count())
    print("training doc2vec PV-DBOW model terminated")


    X_total = model_to_X(model_dm, model_dbow, len(tweets_tokenized) + len(tweets_test_tokenized))
    X = X_total[0: len(tweets_tokenized),:]


    X_test = X_total[len(tweets_tokenized):len(tweets_tokenized)+ len(tweets_test_tokenized),:]
    return X, X_test 


if __name__ == '__main__':
    #data processing
    full_data = False
    train_neg_path, train_pos_path, test_path = data_path(full_data)

    train_pos_label, train_tokenized_pos = raw_to_cleaned_tweets(train_pos_path, 1)    
    train_neg_label, train_tokenized_neg = raw_to_cleaned_tweets(train_neg_path, 0)
    _, test_tokenized = raw_to_cleaned_tweets(test_path, no_label)

    tweets_labels = train_pos_label + train_neg_label


    vector_size = 100
    window_size = 10
    seed = 1
    doc2vec_epochs = 20

    X, X_test = doc2vec(train_tokenized_pos + train_tokenized_neg, test_tokenized, vector_size, window_size, doc2vec_epochs, seed)

    y = np.array(tweets_labels)

    with open(path_results + 'x.npy', 'wb') as f1:
        np.save(f1, X)

    with open(path_results + 'y.npy', 'wb') as f2:
        np.save(f2, y)
    
    with open(path_results + 'x_test.npy.txt', 'wb') as f3:
        np.save(f3, X_test)

    print("Task Terminated")
