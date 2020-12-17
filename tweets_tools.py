import string
import random



def clean_line(line):
    line = line.replace("\n","")
    line = line.replace('<user>','')
    line = line.replace('<url>','')
    line = "".join([char for char in line if char not in string.punctuation])
    return line

def raw_to_cleaned_tweets(path, label):
    count = 0
    tweets_labels = []
    tweets_cleaned = []
    print("extracting {} dataset...".format(path))
    with open(path) as f:
        for line in f:
            if label == 'no_label':
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

def tweets_splitted_in_words(tweets):
  result = []
  for tweet in tweets:
    result.append(tweet.split())
  return result

def get_vocab(list_words):
    vocab = set(list_words)
    return list(vocab)

def get_max_length(tweets_tokenized):
    max = 0
    for tweet in tweets_tokenized:
        if len(tweet) > max:
            max = len(tweet)
    return max


def shuffle_data(X, y, seed=1):
    index_list = list(range(y.shape[0]))
    random.seed(seed)
    random.shuffle(index_list)
    y_shuffled = y[index_list]
    X_shuffled = X[index_list,:]
    return X_shuffled, y_shuffled

def split_train_validation(X, y, train_percentage=0.90) :
    break_point = int(y.shape[0] * train_percentage)
    X_train = X[0:break_point,:]
    X_validation = X[break_point:,:]
    y_train = y[0:break_point]
    y_validation = y[break_point:]
    return X_train, y_train, X_validation, y_validation