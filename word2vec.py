# import word2vec
import nltk 
nltk.download('wordnet')
import string
import gensim
import pickle
import multiprocessing
from gensim.models.word2vec import Word2Vec

path = 'dataset/'
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

def extract_tweets_data(full, use_pickle):
    neg_path, pos_path, pickle_labels_path, pickle_tweets_path = set_data_path(full)

    tweets_labels = []
    tweets_tokenized = []
    if use_pickle:
        with open(pickle_labels_path, "rb") as fp1:
            tweets_labels = pickle.load(fp1)
        with open(pickle_tweets_path, "rb") as fp2:
            tweets_tokenized = pickle.load(fp2)
        return tweets_labels, tweets_tokenized

    print("extracting tweets_data...")
    
    for label, fn in enumerate([neg_path, pos_path]):
        count = 0
        print("extracting {} dataset...".format(label))
        with open(fn) as f:
            for line in f:
                words = line_to_tokenized(line)
                tweets_tokenized.append(words)
                tweets_labels.append(label)

                count += 1
                if(count % 2000 == 0):
                    print(" {} tweets extracted".format(count))
    print("extracting tweets terminated")

    with open(pickle_labels_path, "wb") as fp1:
        pickle.dump(tweets_labels, fp1)
    with open(pickle_tweets_path, "wb") as fp2:
        pickle.dump(tweets_tokenized, fp2)

    return tweets_labels, tweets_tokenized 
    

def word2vec_model(tweets_tokenized, vector_size, window_size, epochs, seed, use_google, google_internet,google_use_pickle,full):
    if use_google:
        return word2vec_google_model(tweets_tokenized, google_internet, google_use_pickle,full)
    else :
        return word2vec_self_training_model(tweets_tokenized, vector_size, window_size, epochs, seed)


def word2vec_google_model(tweets_tokenized, google_internet, google_use_pickle, full):
    #use per-trained word2vec model of google: 300 features per word: 3 millions words
    print("Google model: Loading the model...")
    if google_internet:
        model = gensim.downloader.load("word2vec-google-news-300")
    else: 
        model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
    print("Google model: Loading the model terminated")

    


    tweets_cleaned = []
    if full: 
        path_store = path + 'train_full_pickle_tweets_google.txt'
    else:
        path_store = path + 'train_pickle_tweets_google.txt'

    if google_use_pickle:
        with open(path_store, "rb") as fp1:
            tweets_cleaned = pickle.load(fp1)
        return model, tweets_cleaned, google_vector_size


    print("Google model: removing unknown words ...")  
    
    # TODO : change vocab to have the same voc after line_to_tokenized
    with open('vocab.pkl', 'rb') as f:
        voc_tweets = set(pickle.load(f))
    
    google_voc = set(model.vocab.keys())
    intersection = (voc_tweets & google_voc)
    
    count = 0
    for tweet in tweets_tokenized:
        if count % 2000 == 0:
            print(" {} tweets cleaned".format(count))
        #to remove already known unknown words
        tweet = [word for word in tweet if word in intersection]
        
        tweets_cleaned.append(tweet)
        count += 1
    print("Google model: removing unknown words terminated")

    with open(path_store, "wb") as fp1:
        pickle.dump(tweets_cleaned, fp1)

    return model, tweets_cleaned, google_vector_size



def word2vec_self_training_model(tweets_tokenized, vector_size, window_size, epochs, seed):
    print("training word2vec model...")
    model = Word2Vec(sentences=tweets_tokenized,
                    size=vector_size, 
                    window=window_size, 
                    iter=epochs,
                    seed=seed,
                    workers=multiprocessing.cpu_count())
    print("training word2vec model terminated")
    #save model
    print("saving model")
    model.save('model/model_word2vec')
    print("saving model terminated")
    #load model to continue to train
    # new_model = Word2Vec.load('model/model_word2vec')
    # model.train(more_sentences)

    #fixing model to save space
    vectors = model.wv
    del model
    return vectors, tweets_tokenized, vector_size
            



if __name__ == '__main__':
    #data processing
    full_data = False
    #if want to extract tweets tokenized directly from pickle file
    use_pickle = True
    tweets_labels, tweets_tokenized = extract_tweets_data(full=full_data, use_pickle=use_pickle)

    #word2vec parameters
    vector_size = 5       # ! using google pretrained model, the vector size is fixed at 300
    window_size = 10
    epochs = 2
    seed = 1

    use_google = False
    #loading the google model from internet
    google_internet = False
    google_use_pickle = False
    model, tweets_cleaned, vector_size = word2vec_model(tweets_tokenized, 
                                                        vector_size, window_size, epochs, seed,
                                                        use_google, google_internet, google_use_pickle, full_data)
    
    