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
# from gensim.models.deprecated.doc2vec import FAST_VERSION

path = 'dataset/'
test_path = path + "test_data.txt"
test_pickle_path = path + 'test_pickle_tweets.txt'
google_vector_size = 300

def line_to_tokenized(line):
    line = line.replace('<user>','')
    line = line.replace('<url>','')
    line = "".join([char for char in line if char not in string.punctuation])
    #Tokenize: transform string to list of words
    words = nltk.tokenize.word_tokenize(line)
    #lemmatize : replace word by its root //optional
    words = [nltk.WordNetLemmatizer().lemmatize(word) for word in words]
    #remove stopwords //optional, TODO takes extremly long: see how to accelerate (use a voc appearing in stopword.english and in our data)
    # words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    return words

def set_data_path(full):
    #check if right place in terminal when launching
    if full:
        return path+'train_neg_full.txt', path+'train_pos_full.txt', path+'train_full_pickle_labels.txt', path+'train_full_pickle_tweets.txt'
    else:
        return path+'train_neg.txt',  path+'train_pos.txt', path+'train_pickle_labels.txt',path+'train_pickle_tweets.txt'

def tweets_to_tokens(path, label):
    count = 0
    tweets_labels = []
    tweets_tokenized = []
    print("extracting {} dataset...".format(path))
    with open(path) as f:
        for line in f:
            words = line_to_tokenized(line)
            tweets_tokenized.append(words)
            tweets_labels.append(label)
            count += 1
            if(count % 10000 == 0):
                print(" {} tweets extracted".format(count))
    return tweets_labels, tweets_tokenized
        

def extract_tweets_data(full, use_pickle):
    neg_path, pos_path, pickle_labels_path, pickle_tweets_path = set_data_path(full)

    tweets_labels = []
    tweets_tokenized = []
    tweets_test_tokenized =[]
    if use_pickle:
        with open(pickle_labels_path, "rb") as fp1:
            tweets_labels = pickle.load(fp1)
        with open(pickle_tweets_path, "rb") as fp2:
            tweets_tokenized = pickle.load(fp2)
        with open(test_pickle_path, "rb") as fp3:
            tweets_test_tokenized = pickle.load(fp3)
        
        return tweets_labels, tweets_tokenized, tweets_test_tokenized
    
    print("extracting tweets_data...")
    tweets_labels_pos, tweets_tokenized_pos = tweets_to_tokens(pos_path, 1)
    tweets_labels_neg, tweets_tokenized_neg = tweets_to_tokens(neg_path, -1)
    _, tweets_test_tokenized = tweets_to_tokens(test_path, 0) 
    print("extracting tweets terminated")

    tweets_tokenized = tweets_tokenized_pos + tweets_tokenized_neg
    tweets_labels = tweets_labels_pos + tweets_labels_neg

    with open(pickle_labels_path, "wb") as fp1:
        pickle.dump(tweets_labels, fp1)
    with open(pickle_tweets_path, "wb") as fp2:
        pickle.dump(tweets_tokenized, fp2)
    with open(test_pickle_path, "wb") as fp3:
        pickle.dump(tweets_test_tokenized, fp3)

    return tweets_labels, tweets_tokenized, tweets_test_tokenized
    

def clean_tweets_tokenized(tweets_tokenized, intersection):
    print("number of tweets :{}".format(len(tweets_tokenized)))
    tweets_cleaned = []
    count = 1
    for tweet in tweets_tokenized:
        if count % 10000 == 0:
            print(" {} tweets cleaned".format(count))
        tweet = [word for word in tweet if word in intersection]
        tweets_cleaned.append(tweet)
        count += 1
    return tweets_cleaned
    
def word2vec_google_model(tweets_tokenized, tweets_test_tokenized, vocab, google_internet, google_use_pickle, full):
    #use per-trained word2vec model of google: 300 features per word: 3 millions words
    print("Google model: Loading the model...")
    if google_internet:
        model = gensim.downloader.load("word2vec-google-news-300")
    else: 
        model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
    print("Google model: Loading the model terminated")

    tweets_cleaned = []
    tweets_test_cleaned = []
    if full: 
        path_store = path + 'train_full_pickle_tweets_google.txt'
        path_store_test = path + 'train_full_pickle_tweets_google_test.txt'
    else:
        path_store = path + 'train_pickle_tweets_google.txt'
        path_store_test = path + 'train_full_pickle_tweets_google_test.txt'

    if google_use_pickle:
        with open(path_store, "rb") as fp1:
            tweets_cleaned = pickle.load(fp1)
        with open(path_store, "rb") as fp2:
            tweets_test_cleaned = pickle.load(fp2)
        return model, tweets_cleaned, tweets_test_cleaned, google_vector_size

    print("Google model: removing unknown words ...")  
    
    # TODO : change vocab to have the same voc after line_to_tokenized
    
    google_voc = set(model.vocab.keys())
    intersection = (vocab & google_voc)
    tweets_cleaned = clean_tweets_tokenized(tweets_tokenized, intersection)
    tweets_test_cleaned = clean_tweets_tokenized(tweets_test_tokenized, intersection)
    
    print("Google model: removing unknown words terminated")

    with open(path_store, "wb") as fp1:
        pickle.dump(tweets_cleaned, fp1)
    with open(path_store_test, "wb") as fp2:
        pickle.dump(tweets_test_cleaned, fp2)

    return model, tweets_cleaned,tweets_test_cleaned, google_vector_size



# can train more sentences : model.train(more_sentences)
def word2vec_self_training_model(tweets_tokenized, vector_size, window_size, epochs, seed, stored_model):
    if stored_model:
        model = Word2Vec.load('model/model_word2vec')
    else:
        print("training word2vec model...")
        model = Word2Vec(sentences=tweets_tokenized,
                        size=vector_size, 
                        window=window_size, 
                        iter=epochs,
                        seed=seed,
                        workers=multiprocessing.cpu_count())
        print("training word2vec model terminated")

        model.save('model/model_word2vec')

    #fixing model to save space
    vectors = model.wv
    del model
    return vectors



def model_to_X(model, num_samples):
    X = np.zeros((num_samples, vector_size))
    for i in range(num_samples):
        X[i, :] = model.docvecs[i]
    return X

#TODO: explain, https://radimrehurek.com/gensim/models/doc2vec.html
# from the paper https://arxiv.org/pdf/1405.4053v2.pdf
def doc2vec(tweets_tokenized, tweets_test_tokenized, vector_size, window_size, epochs, seed, get_stored_model, PV_DM = 1):
    if get_stored_model:
        model = Doc2Vec.load('model/model_doc2vec')
    else:
        docs = [TaggedDocument(doc, [tag]) for tag, doc in enumerate(tweets_tokenized + tweets_test_tokenized)]
        print("training doc2vec model...")

        model =  Doc2Vec(docs,
                        dm=PV_DM, 
                        vector_size=vector_size, 
                        window=window_size, 
                        seed=seed,
                        epochs=epochs,
                        workers=multiprocessing.cpu_count())
        print("training doc2vec model terminated")
        model.save('model/model_doc2vec')

    X_total = model_to_X(model, len(tweets_tokenized) + len(tweets_test_tokenized))
    X = X_total[0: len(tweets_tokenized),:]
    X_test = X_total[len(tweets_tokenized):len(tweets_tokenized)+ len(tweets_test_tokenized),:]
    return X, X_test 

def get_vocab_and_max_length(tweets_tokenized, vocab_stored):
    vocab = set()
    if vocab_stored:
        with open('dataset/vocab.pkl', 'rb') as f:
            vocab = set(pickle.load(f))
    else: 
        flattened = [item for sublist in tweets_tokenized for item in sublist]
        print(type(flattened[0]))
        vocab = set(flattened)
        flattened = []
    max_len = 0
    for tweet in tweets_tokenized:
        if len(tweet) > max_len:
            max_len = len(tweet)
    with open('dataset/vocab.pkl', "wb") as fp2:
        pickle.dump(vocab, fp2)
    return vocab, max_len

if __name__ == '__main__':
    #data processing
    full_data = False
    #if want to extract tweets tokenized directly from pickle file
    use_pickle = True
    tweets_labels, tweets_tokenized, tweets_test_tokenized = extract_tweets_data(full=full_data, use_pickle=use_pickle)


    vocab_from_storage = False
    vocab, max_len = get_vocab_and_max_length((tweets_tokenized + tweets_test_tokenized), vocab_from_storage)
    print(len(vocab))
    #!!! use either model_word2vec or model_google, not both !!!
    #Word2vec
    get_stored_model = False
    #word2vec parameters
    vector_size = 50
    window_size = 10
    epochs = 2
    seed = 1
    # model_word2vec = word2vec_self_training_model(tweets_tokenized + tweets_test_tokenized, 
    #                                               vector_size, window_size, epochs, seed,
    #                                               stored_model = get_stored_model)

    # loading the google model from internet // using google pretrained model, the vector size is fixed at 300
    google_internet = False
    google_use_pickle = False
    model_google, tweets_cleaned, tweets_test_cleaned, vector_size = word2vec_google_model(tweets_tokenized, tweets_test_tokenized, vocab,
                                                                                         google_internet, google_use_pickle, full_data)


    #Tweets to vectors
    #1. using doc2vec !!! if so, comment word2vec model computation (line above)
    doc2vec_epochs = 10
    #use distributed memory = 1, bag-of-words = 0 see paper https://arxiv.org/pdf/1405.4053v2.pdf
    PV_DM = 1
    get_stored_model_doc2vec = False
    # X, X_test = doc2vec(tweets_tokenized, tweets_test_tokenized, vector_size, window_size, doc2vec_epochs, seed, 
    #             get_stored_model = get_stored_model_doc2vec,PV_DM = PV_DM)

    y = np.array(tweets_labels)

    with open('data_vectors/x.npy', 'wb') as f1:
        np.save(f1, X)

    with open('data_vectors/y.npy', 'wb') as f2:
        np.save(f2, y)
    
    with open('dataset/x_test.npy.txt', 'wb') as f3:
        np.save(f3, X_test)

    print("Task Terminated")

    
    