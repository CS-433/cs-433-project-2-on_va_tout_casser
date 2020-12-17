from keras.layers import Dense, Dropout, Input, GlobalMaxPooling1D, Conv1D, concatenate, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding
from tensorflow.keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
from keras.models import load_model


def build_keras_tokenizer(tweets_cleaned, num_words):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(tweets_cleaned)
    return tokenizer


def tweets_tokenizer(tokenizer, tweets):
    return tokenizer.texts_to_sequences(tweets)


def  tweets_padding(tweets_tokenized, max_length):
    return pad_sequences(tweets_tokenized, maxlen=max_length)


def build_embedding_index(model_skipgram, model_cbow):
    embeddings_index = {}
    for w in model_cbow.wv.vocab.keys():
        embeddings_index[w] = np.append(model_cbow.wv[w],model_skipgram.wv[w])
    return embeddings_index


def build_embedding_matrix(max_num_words, tokenizer, embedding_index, vector_size):
    embedding_matrix = np.zeros((max_num_words, vector_size))
    for word, i in tokenizer.word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_neural_network_model(embedding_matrix, 
                            max_num_words, 
                            vector_size, 
                            length_input, 
                            filter_number=100, 
                            dense_number=256, 
                            dropout=0.1,
                            loss ='binary_crossentropy', 
                            optimizer='adam',
                            metrics=['binary_accuracy'], 
                            trainable=True,
                            activation = "relu"):
  
    tweet_input = Input(shape=(length_input,), dtype='int32')
    tweet_encoder = Embedding(max_num_words, vector_size,
                              embeddings_initializer = Constant(embedding_matrix),
                              input_length=length_input, trainable=trainable)(tweet_input)


    bigram_branch = Conv1D(filters=filter_number, kernel_size=2, padding='valid', 
                            activation=activation, strides=1)(tweet_encoder)
    bigram_branch = GlobalMaxPooling1D()(bigram_branch)

    trigram_branch = Conv1D(filters=filter_number, kernel_size=3, padding='valid', 
                            activation=activation, strides=1)(tweet_encoder)
    trigram_branch = GlobalMaxPooling1D()(trigram_branch)

    fourgram_branch = Conv1D(filters=filter_number, kernel_size=4, padding='valid', 
                             activation=activation, strides=1)(tweet_encoder)
    fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)

    merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

    merged = Dense(dense_number, activation=activation)(merged)
    merged = Dropout(dropout)(merged)
    merged = Dense(1)(merged)
    output = Activation('sigmoid')(merged)

    model = Model(inputs=[tweet_input], outputs=[output])
    model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
    model.summary()
    return model