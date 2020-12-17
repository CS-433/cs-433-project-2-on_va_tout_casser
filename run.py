from tweets_tools import *
from cnn import *
from utils import *
from word2vec_final import *
from datetime import datetime

#MAKE SURE TO BE ON RIGHT LOCATION WHEN LAUNCHING THE PROGRAM

#WINDOWS: replace '/' by '\\' : 
path = "datasets/"
path_processed_dataset = "processed_dataset/" 
path_results = "results/"
path_model = "model/"

pos_file = "train_pos_full.txt"
neg_file = "train_neg_full.txt"
test_file = "test_data.txt"

name_model = "Word2Vec_CNN"

seed = 42

#Word2vec Params
skip_vector_size = 50
cbow_vector_size = 10
total_vector_size = skip_vector_size + cbow_vector_size
window_size = 4
epochs_word2vec = 15
min_word_count =  2

#don't change the Word2Vec params when set to True, will alterate name_submission
use_pickle = False

#CNN Params
train_percentage_validation = 0.90
trainable = True
filter_number = 300
dense_number = 256
dropout = 0.2
epochs_nn = 5
batch_size = 50
activation_function = 'relu'
loss = 'binary_crossentropy'
optimizer = 'adam'




if __name__ == '__main__':
    if use_pickle:
        X, X_test, y, embedding_matrix, max_length = recover_processed_data(path_processed_dataset)
    
    else:

        print("Extracting Tweets...")
        train_neg_path, train_pos_path, test_path = data_path(path, neg_file, pos_file, test_file)

        train_pos_label, train_pos = raw_to_cleaned_tweets(train_pos_path, 1)    
        train_neg_label, train_neg = raw_to_cleaned_tweets(train_neg_path, 0)
        _, test = raw_to_cleaned_tweets(test_path, 'no_label')
        print("Extracting Tweets terminated")


        print("formating tweets for word2vec...")
        all_tweets = train_pos + train_neg + test
        all_tweets_sentences = tweets_splitted_in_words(all_tweets)
        print("formating tweets for word2vec terminated")

        model_skipgram, model_cbow =  word2vec_self_training_model( all_tweets_sentences,
                                                                   skip_vector_size,
                                                                   cbow_vector_size, 
                                                                   window_size,
                                                                   epochs_word2vec,
                                                                   seed, 
                                                                   min_word_count) 

    

        #tokenize to ids
        print("tokenizing ...")
        max_num_words = len(list(model_skipgram.wv.vocab))
        tokenizer = build_keras_tokenizer(all_tweets,max_num_words)
        tweets_tokenized = tweets_tokenizer(tokenizer, train_pos + train_neg)
        tweets_test_tokenized = tweets_tokenizer(tokenizer, test)
        print("tokenizing terminated")
    

        print("Padding ...")
        max_length = get_max_length(tweets_tokenized + tweets_test_tokenized)
        X = tweets_padding(tweets_tokenized, max_length)
        X_test = tweets_padding(tweets_test_tokenized, max_length)
        y = np.array(train_pos_label + train_neg_label)
        print("Padding  terminated")


        print("building embedding matrix ...")
        embeddings_index = build_embedding_index(model_skipgram, model_cbow)
        embedding_matrix = build_embedding_matrix(max_num_words, tokenizer, embeddings_index, total_vector_size)
        print("building embedding matrix terminated")

        store_processed_data(X, X_test, y, embedding_matrix, max_length, path_processed_dataset)


    print("Preparing the neural network...")
    X, y = shuffle_data(X, y, seed=seed)
    X_train, y_train, X_validation, y_validation = split_train_validation(X, y, train_percentage_validation)

    model = get_neural_network_model(embedding_matrix, embedding_matrix.shape[0], embedding_matrix.shape[1],
                                     max_length,
                                     filter_number = filter_number,
                                     dense_number = dense_number,
                                     dropout = dropout,
                                     loss = loss, 
                                     optimizer= optimizer, 
                                     trainable = trainable,
                                     activation = activation_function)

    filepath= path_model + "word2vec_CNN_best_weights"
    checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy',
                                verbose=1, save_best_only=True, mode='max')

    print("Preparing the neural network terminated")


    print("training neural network...")
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        epochs=epochs_nn, batch_size=batch_size, verbose=1,
                        callbacks = [checkpoint])
    loaded_model = load_model(filepath)
    print("training neural network terminated")
    

    print("predict test set...")
    pred = loaded_model.predict(X_test)
    pred_int = pred.round().astype("int")
    _, accuracy = loaded_model.evaluate(X_validation, y_validation, batch_size=128) 
    print("predict test set terminated")


    print("Storing results...")
    name_list = ["test_file", "seed",
                "skip_vector_size", "cbow_vector_size",
                "window", "word2vec_epochs", "min_word_count", 
                "val_percentage", "trainable", 
                "filters", "dense", "dropout",  
                "epochs_nn","batch_size",
                "activation", "loss", "optimizer", "result_accuracy"]
    value_list = [str(test_file),str(seed), 
                str(skip_vector_size), str(cbow_vector_size),
                str(window_size), str(epochs_word2vec), str(min_word_count),
                str(train_percentage_validation), str(trainable),  
                str(filter_number), str(dense_number), str(dropout),
                str(epochs_nn), str(batch_size),
                activation_function, loss, optimizer, 
                str(accuracy)]

    date = datetime.now()
    date_time = date.strftime('%x').replace("/","-") + '__' +date.strftime('%X').replace(":","-")
    name_submission =  name_model + "__" + '{:06.4f}'.format(accuracy) + "__" + date_time

    store_submission(name_submission, pred_int, path_results)
    store_results(name_list, value_list, path_results, date_time, name_model)
    draw_graph_validation_epoch(history, name_submission, path_results, loss)
    print("Storing results terminated")

    print("the magic has terminated, what a great time we lived together")
