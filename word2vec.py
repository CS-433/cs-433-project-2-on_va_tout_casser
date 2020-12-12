import nltk 
nltk.download('wordnet')
import string
import gensim
import pickle
import gensim.models.doc2vec as qwe
import multiprocessing
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D
from keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding
import random
import tensorflow_hub as hub
import bert
import matplotlib.pyplot as plt
from datetime import datetime
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
from keras.models import load_model

path = 'dataset/'
test_path = path + "test_data.txt"
path_processed_dataset = "processed_dataset/" 
path_results = "results/"
path_model = "model/"
no_label = 5

    



# google_vector_size = 300
# def clean_tweets_tokenized(tweets_tokenized, intersection):
#     print("number of tweets :{}".format(len(tweets_tokenized)))
#     tweets_cleaned = []
#     count = 1
#     for tweet in tweets_tokenized:
#         if count % 10000 == 0:
#             print(" {} tweets cleaned".format(count))
#         tweet = [word for word in tweet if word in intersection]
#         tweets_cleaned.append(tweet)
#         count += 1
#     return tweets_cleaned


# def word2vec_google_model(tweets_tokenized, tweets_test_tokenized, vocab, google_internet, google_use_pickle, full):
#     #use per-trained word2vec model of google: 300 features per word: 3 millions words
#     print("Google model: Loading the model...")
#     if google_internet:
#         model = gensim.downloader.load("word2vec-google-news-300")
#     else: 
#         model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
#     print("Google model: Loading the model terminated")

#     tweets_cleaned = []
#     tweets_test_cleaned = []
#     if full: 
#         path_store = path + 'train_full_pickle_tweets_google.txt'
#         path_store_test = path + 'train_full_pickle_tweets_google_test.txt'
#     else:
#         path_store = path + 'train_pickle_tweets_google.txt'
#         path_store_test = path + 'train_full_pickle_tweets_google_test.txt'

#     if google_use_pickle:
#         with open(path_store, "rb") as fp1:
#             tweets_cleaned = pickle.load(fp1)
#         with open(path_store, "rb") as fp2:
#             tweets_test_cleaned = pickle.load(fp2)
#         return model, tweets_cleaned, tweets_test_cleaned, google_vector_size

#     print("Google model: removing unknown words ...")  
    
#     # TODO : change vocab to have the same voc after line_to_tokenized
    
#     google_voc = set(model.vocab.keys())
#     intersection = (vocab & google_voc)
#     tweets_cleaned = clean_tweets_tokenized(tweets_tokenized, intersection)
#     tweets_test_cleaned = clean_tweets_tokenized(tweets_test_tokenized, intersection)
    
#     print("Google model: removing unknown words terminated")

#     with open(path_store, "wb") as fp1:
#         pickle.dump(tweets_cleaned, fp1)
#     with open(path_store_test, "wb") as fp2:
#         pickle.dump(tweets_test_cleaned, fp2)

#     return model, tweets_cleaned,tweets_test_cleaned, google_vector_size






def clean_line(line):
    line = line.replace("\n","")
    line = line.replace('<user>','')
    line = line.replace('<url>','')
    line = "".join([char for char in line if char not in string.punctuation])
    return line



def word2vec_self_training_model(vocabulary, vector_size, window_size, epochs, seed=1, neg_sampling=5, alpha=0.065, min_word_count=2):
    print("training word2vec skipgram model...")
    model_skipgram = Word2Vec(sentences=vocabulary,
                        size=int(vector_size/2), 
                        window=window_size, 
                        iter=epochs,
                        seed=seed,
                        sg=1,
                        negative=neg_sampling,
                        alpha=alpha,
                        min_alpha=alpha,
                        min_count=min_word_count,
                        workers=multiprocessing.cpu_count())
    print("training word2vec skipgram model terminated")
    
    print("training word2vec cbow model...")
    model_cbow = Word2Vec(sentences=vocabulary,
                        size=int(vector_size/2), 
                        window=window_size, 
                        iter=epochs,
                        seed=seed,
                        sg=0,
                        negative=neg_sampling,
                        alpha=alpha,
                        min_alpha=alpha,
                        min_count=min_word_count,
                        workers=multiprocessing.cpu_count())
    print("training word2vec cbow model terminated")
    return model_skipgram, model_cbow

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
        

def list_tweets_to_list_words(list_tweets):
    list_words = []
    for tweet in list_tweets:
        list_words.extend(tweet.split())
    return list_words

def get_vocab(list_words):
    vocab = set(list_words)
    return list(vocab)

def build_embedding_index(model_skipgram, model_cbow):
    embeddings_index = {}
    for w in model_cbow.wv.vocab.keys():
        embeddings_index[w] = np.append(model_cbow.wv[w],model_skipgram.wv[w])
    return embeddings_index

def build_keras_tokenizer(tweets_cleaned, num_words):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(tweets_cleaned)
    return tokenizer

def get_max_length(tweets_tokenized):
    max = 0
    for tweet in tweets_tokenized:
        if len(tweet) > max:
            max = len(tweet)
    return max

def build_embedding_matrix(max_num_words, tokenizer, embedding_index, vector_size):
    embedding_matrix = np.zeros((max_num_words, vector_size))
    for word, i in tokenizer.word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def shuffle_data(X, y, seed=1):
    index_list = list(range(y.shape[0]))
    random.seed(seed)
    random.shuffle(index_list)
    y_shuffled = y[index_list]
    X_shuffled = X[index_list,:]
    return X_shuffled, y_shuffled

def store_processed_data(X, X_test, y, embedding_matrix):
    with open(path_processed_dataset+ "X"+ ".txt", "wb") as fp1:
        pickle.dump(X, fp1)
    with open(path_processed_dataset+ "X_test"+ ".txt", "wb") as fp2:
        pickle.dump(X_test, fp2)
    with open(path_processed_dataset+ "y"+ ".txt", "wb") as fp3:
        pickle.dump(y, fp3)
    with open(path_processed_dataset+ "embedding_matrix"+ ".txt", "wb") as fp4:
        pickle.dump(embedding_matrix, fp4)
    
def recover_processed_data():
    with open(path_processed_dataset+ "X"+ ".txt", "rb") as fp1:
        X = pickle.load(fp1)
    with open(path_processed_dataset+ "X_test"+ ".txt", "rb") as fp2:
        X_test = pickle.load(fp2)
    with open(path_processed_dataset+ "y"+ ".txt", "rb") as fp3:
        y = pickle.load(fp3)
    with open(path_processed_dataset+ "embedding_matrix"+ ".txt", "rb") as fp4:
        embedding_matrix = pickle.load(fp4)
    return X, X_test, y, embedding_matrix

def split_train_validation(X, y, train_percentage=0.85) :
    break_point = int(y.shape[0] * train_percentage)
    X_train = X[0:break_point,:]
    X_validation = X[break_point:,:]
    y_train = y[0:break_point]
    y_validation = y[break_point:]
    return X_train, y_train, X_validation, y_validation

def get_neural_network_model(embeding_matrix, max_num_words, vector_size, length_input, filter_number=100, dense_number=256, dropout=0.2,
                            loss ='binary_crossentropy', optimizer='adam',metrics=['binary_accuracy']):
    tweet_input = Input(shape=(length_input,), dtype='int32')
    tweet_encoder = Embedding(max_num_words, vector_size, weights=[embedding_matrix], input_length=length_input, trainable=True)(tweet_input)
    bigram_branch = Conv1D(filters=filter_number, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
    bigram_branch = GlobalMaxPooling1D()(bigram_branch)
    trigram_branch = Conv1D(filters=filter_number, kernel_size=3, padding='valid', activation='relu', strides=1)(tweet_encoder)
    trigram_branch = GlobalMaxPooling1D()(trigram_branch)
    fourgram_branch = Conv1D(filters=filter_number, kernel_size=4, padding='valid', activation='relu', strides=1)(tweet_encoder)
    fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
    merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

    merged = Dense(dense_number, activation='relu')(merged)
    merged = Dropout(dropout)(merged)

    merged = Dense(1)(merged)
    output = Activation('sigmoid')(merged)
    model = Model(inputs=[tweet_input], outputs=[output])
    model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
    model.summary()
    return model



def draw_graph_validation_epoch(history, datetime):
    history_dict = history.history

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss'+ "binary_crossentropy")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(path_results + 'graph_' + datetime + '.png')


def store_results(name_submission, pred_int):
    try:
        resFile = open(path_results + name_submission +  ".csv","w")
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


if __name__ == '__main__':
    full_data = False
    train_neg_path, train_pos_path, test_path = data_path(full_data)

    train_pos_label, train_pos = raw_to_cleaned_tweets(train_pos_path, 1)    
    train_neg_label, train_neg = raw_to_cleaned_tweets(train_neg_path, 0)
    _, test = raw_to_cleaned_tweets(test_path, no_label)

    all_tweets = train_pos + train_neg + test
    vocab = get_vocab(list_tweets_to_list_words(all_tweets))

    vector_size = 200 # must be odd : 1/2 for both models
    window_size = 6
    epochs_word2vec = 30
    neg_sampling = 5
    alpha = 0.065
    min_word_count =  2
    seed = 1
    
    model_skipgram, model_cbow =  word2vec_self_training_model(vocab, vector_size, window_size, epochs_word2vec, seed, neg_sampling, alpha, min_word_count) 

    embeddings_index = build_embedding_index(model_skipgram, model_cbow)

    max_num_words = 100000 # << len(vocab))
    #tokenize to ids
    print("tokenizing ...")
    tokenizer = build_keras_tokenizer(all_tweets,max_num_words)
    tweets_tokenized = tokenizer.texts_to_sequences(train_pos + train_neg)
    tweets_test_tokenized = tokenizer.texts_to_sequences(test)
    print("tokenizing terminated")
    max_length = get_max_length(tweets_tokenized + tweets_test_tokenized)


    #padding to get same length
    X = tf.keras.preprocessing.sequence.pad_sequences(tweets_tokenized, maxlen=max_length)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(tweets_test_tokenized, maxlen=max_length)
    y = np.array(train_pos_label + train_neg_label)

    print("building embedding matrix ...")
    
    embedding_matrix = build_embedding_matrix(max_num_words, tokenizer, embeddings_index, vector_size)
    print("building embedding matrix terminated")


    store_processed_data(X, X_test, y, embedding_matrix)
    recover_processed_data()
    seed_val = 1
    train_percentage_validation = 0.85
    X, y = shuffle_data(X, y, seed=seed_val)
    X_train, y_train, X_validation, y_validation = split_train_validation(X, y, train_percentage_validation)


    filter_number = 100
    dense_number = 256
    dropout = 0.2
    #modify the loss and the metric could break the graph
    loss ='binary_crossentropy'
    optimizer='adam'
    model = get_neural_network_model(embedding_matrix, embedding_matrix.shape[0], embedding_matrix.shape[1], max_length,
                                     filter_number=filter_number, dense_number=dense_number, dropout=dropout,
                                     loss =loss, optimizer=optimizer)

    filepath= path_model + "word2vec_CNN_best_weights"
    checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')

    print("training neural network...")
    epochs_nn = 2
    batch_size = 64
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        epochs=epochs_nn, batch_size=batch_size, verbose=1,
                        callbacks = [checkpoint])


    loaded_model = load_model(path_model + 'word2vec_CNN_best_weights')


    datetime = str(datetime.now()).replace(" ","__").replace(":","-")
    
    
    print("predict test set...")
    pred = loaded_model.predict(X_test)
    pred_int = pred.round().astype("int")

    
    name_submission = datetime
    store_results(name_submission, pred_int)

    print("drawing graph...")
    draw_graph_validation_epoch(history, datetime)

    print("the magic has terminated, what a great time we lived together")
    
 