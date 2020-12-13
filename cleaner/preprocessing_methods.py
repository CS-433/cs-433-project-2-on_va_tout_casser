import numpy as np
import nltk
import pandas as pd 
import string
import re
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
import time


################################################## Lists of useful informations ######################################################
######################################################################################################################################
######################################################################################################################################


twitter_abbreviations = [('ofc', 'of course'), ('lmao', 'laughing my ass off', 'haha'), ('lmfao', 'laughing my fucking ass off', 'haha'),
       ('lyvm', 'love you very much'), ('afk', 'away from keyboard'), 
      ('aight', 'alright'), ('aka', 'also known as'), ('lol', 'laughing out loud', 'haha'), 
      ('aisi', 'as i see it'), ('alcon', 'all concerned'), ('asap', 'as soon as possible'), 
      ('atm', 'at the moment'), ('aweso', 'awesome'), ('b', 'be'), ('bf', 'boyfriend'), ('bae', 'baby'), ('bb', 'baby'),
        ('babe', 'baby'), ('bbl', 'be back later'), ('bc', 'because'), ('cuz', 'because'), ('bday', 'birthday'), 
            ('bff', 'best friend forever'), ('bfn', 'bye for now'), ('bg', 'big grin', 'haha'), ('bih', 'burn in hell'),
        ('bl', 'belly laugh', 'haha'), ('bloke', 'man'), ('bn', 'bad news'), ('bol', 'best of luck'), ('boyf', 'boyfriend'),
        ('brb', 'be right back'), ('bs', 'bullshit'), ('brd', 'bored'), ('btw', 'by the way'),
         ('clab', 'crying like a baby'), ('ciao', 'goodbye'), ('y', 'why'), ('cto', 'check this out'), ('cos', 'because'), 
            ('cmon', 'come on'), ('c', 'see'), ('cu', 'see you'), ('cya', 'see you'), ('dafuq', 'what the fuck'), ('wtf', 'what the fuck'),
            ('dc', 'disconnect'), ('dd', 'dear'), ('derp', 'silly'), ('dgaf', 'do not give a fuck'),
            ('dh', 'dear'), ('dhu', 'dinosaur hugs'), ('diy', 'do it yourself'),('dm', 'direct message'), ('dmed', 'direct message'),
            ('dnt', 'do not'), ('dw', 'do not worry'), ('ez', 'easy'), ('fab', 'fabulous'), ('fam','family'),
                ('fb', 'facebook'), ('ffs', 'for fuck sake'), ('fml', 'fuck my life'), ('ftw', 'for the win'),
                ('fyi', 'for your information'), ('gf', 'girlfriend'), ('gb', 'goodbye'), ('gd', 'good'), ('gl', 'good luck'),
            ('gn', 'goodnight'), ('gnight', 'goodnight'), ('gnite', 'goodnight'), ('gtg', 'got to go'), ('hmu', 'hit me up'),
            ('hw', 'homework'), ('idc', 'i do not care'), ('idk', 'i do not know'), ('idgaf', 'i do not give a fuck'),
            ('ik', 'i know'), ('ikr', 'i know right'), ('ily', 'i love you'), ('imo', 'in my opinion'),
            ('imu', 'i miss you'), ('irl', 'in real life'), ('k', 'okay'),  ('lmk', 'let me know'),
            ('lil', 'little'), ('meh', 'shrug'), ('msg', 'message'), ('ngl', 'not going to lie'),('nvm', 'never mind'),
            ('np', 'no problem'), ('nvr', 'never'), ('omg', 'oh my god'), ('omfg', 'oh my fucking god'),
            ('omw', 'on my way'), ('otp', 'one true pairing'), ('otw', 'off to work'), ('peeps', 'people'),
            ('pls', 'please'), ('plz', 'please'), ('ppl', 'people'),('probs', 'probably'),('prolly', 'probably'),
            ('qt', 'cute'), ('rofl', 'rolling on the floor laughing', 'haha'), ('rip','rest in peace'), ('rn','right now'),
            ('rly', 'really'), ('sd', 'sweet dreams'), ('smh', 'shake my head'), ('srsly', 'seriously'),
            ('sry', 'sorry'), ('sup', 'what is up'), ('hbu', 'how about you'), ('stfu', 'shut the fuck up'),
            ('ty', 'thank you'), ('tyvm', 'thank you very much'), ('thx', 'thank you'), ('tbh', 'to be honest'),
            ('thot', 'that whore over there'), ('totes', 'totally'), ('wbu', 'what about you'), ('wtf','what the fuck'),
            ('x', 'kiss'), ('xd', 'smile', 'haha'), ('<3', 'heart'), ('pic', 'picture'), ('2', 'to'), ('b4', 'before'),
            ('xoxo', 'kiss'), (':)', 'happy'), (':(', 'sad'), ('4', 'for')]


twitter_obvious_mistakes = [('dat', 'that'), ('dis', 'this'), ('wud', 'would'), ('da', 'the'), ('enuf', 'enough'),
                           ('gud', 'good'), ('dunno', 'do not know'), ('im', 'i am'), ('ok', 'okay'), ('r', 'are'), ('u', 'you'),
                           ('luv', 'love'), ('ur', 'your'), ('wat', 'what'), ('wut', 'what'), ('y', 'why'), ('ya', 'you'), ('yea', 'yes'), 
                           ('dont', 'do not'), ('cant', 'cannot'), ('wont', 'will not'), ('aint', 'am not'), ('isnt', 'is not'),
                            ('doesnt', 'does not'), ('hasnt', 'has not'), ('havent', 'have not'), ('id', 'i would'),
                            ('theres', 'there is'), ('thats', 'that is'), ('wheres', 'where is'), ('werent', 'were not'), ('tho', 'though'),
                           ('doe', 'though'), ('altho', 'though'), ('moar', 'more'), ('fren', 'friend'), ('dem', 'them')]


contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he shall",
"he'll've": "he shall have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'l": "i will",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'al": "you all",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


onomatopees = [('pf', 'sigh'), ('mwah', 'kiss'), ('muah', 'kiss'), ('uh', 'uh'), ('tsk', 'tsk'), ('hm', 'hm'), ('mhm', 'mhm'), ('wow', 'wow')]


laughs = ['hah', 'heh', 'hih', 'huh', 'hoh', 'jaj']


stopword_eg = nltk.corpus.stopwords.words('english')


twitter_stopwords = ['user', 'rrrpp', 'mspld', 'exclmt', 'abbrvt', 'hashtagg']


twitter_stopwords.extend(stopword_eg)


list_of_tokens = ['ooo', 'abbrvt', 'hashtagg', 'exclmt', 'haha', 'rrrpp', 'mspld']

########################################################## Preprocessing helpers ###########################################################
###########################################################################################################################################
###########################################################################################################################################


def split_list_of_tuples(l):
    """ Retrieve a list of the first elements in each tuples"""
    el1 = []
    for t in l:
        el1.append(t[0])
    return el1


def repeat_exclamation(word):
    res = False
    i = 0
    while i < len(word) - 1:
        if word[i] == '!' and word[i+1] == '!':
            res = True
        i += 1
    return res


def remove_punctuation(text, excepts='!#<\'():', numbers=True):
    """ Removes the punctuation from a text, except the ones in except (a string)."""
    text = text.lower()  # Remove caps
    punct = string.punctuation
    
    for symb in excepts:
        punct = punct.replace(symb, '')
        
    text = "".join([char for char in text if char not in punct])
    #text = re.sub('[0-1]+', '', text)
    #text = re.sub('[5-9]+', '', text)
    if not numbers :
        raise NotImplementedError
    return text


def turn_corrections_and_tokens_into_list(data):
    """ Method returns a list of the corrections and the tokens."""
    l = []
    for i, corr in enumerate(data.correction):
        l += corr
        l += data.iat[i, 3]  # Add the tokens
    return l


def remove_stopwords(correction, stopword):
    """ function reads the correction part of data and adds a new column of words without any stopword.
    """
    
    correction = [word for word in correction if word not in stopword]
    return correction


ps = nltk.PorterStemmer()
def stem(correction):
    
    correction = [ps.stem(word) for word in correction]
    return correction

wn = nltk.WordNetLemmatizer()

def lemmatize(corr):
    corr = [wn.lemmatize(word) for word in corr]
    return corr


def still_in_same_tweet(data, j):
    """ Method checks if the next word is in the same tweet"""
    if j + 1 == len(data.filewords):
        return True
    if j == len(data.filewords):
        return False
    pos1 = data.iat[j, 1]
    pos2 = data.iat[j+1, 1]
     
    return pos1[0] == pos2[0]


# Methods for the repeat part of the preprocessing.


def repeat_identifier(word, spell):
    """ Function takes a word and will answer whether there are repeated characters in order to correct them
        if they are misspelled.
    """
    count = 0
    # if word == spellchecker.correction(word):
    while count < len(word):
        if count + 1 < len(word):
            if word[count] == word[count + 1] and len(spell.unknown([word])) == 1:
                return True
        count += 1
    return False


def find_repeat_indices(data):
    """ Function that returns a list of indices for each word that has repeating letters (in data)."""
    rep = []
    spell = SpellChecker()
    for i, w in enumerate(data.filewords):
        if repeat_identifier(w, spell) and repeat_identifier(data.iat[i, 2], spell):
            rep.append(i)
    return rep


def find_repeat_markers(word):
    """ Given a word with repeating characters, this method will compute
        two lists :
        repeat_markers : a list of indices where a repetion will occur
        repeat_numbers : a list of such number of repetitions
        
        Example : the word 'heeloo' will return [1, 4] as repeat markers, and
                  [1, 1] as repeat numbers. 
                  """
    count = 0
    # initialise the indicators of repeatedness in the word
    # the repeat_markers will be a list of the first indices where a character will repeat.
    # The repeat_numbers will denote the amount of repetitions.
    repeat_markers = []
    repeat_numbers = []
    
    # Read the word and identify the repeating characters.
    while count < len(word):
        if count + 1 < len(word):
            if word[count] == word[count + 1]:               
                repeater = count
                while repeater + 1 < len(word) and word[repeater] == word[repeater+1]:
                    repeater += 1
                repeat_numbers.append(repeater - count)
                repeat_markers.append(count)
                count = repeater
        count += 1
    
    return repeat_markers, repeat_numbers


def construct_candidates_for_repeat_word(word):
    """ Method takes a word as a string in input, this word must have at least one repeating character
        otherwise an error will occur.
        This method will return a list of words that have less repeating charcters in them (in decreasing order).
        """
    
    repm, repn = find_repeat_markers(word)
    # We take repn, and return an array of all the possibilities we can have for repeated letters :
    # For instance in heeloo, we have repn = [1, 1] therefore all_possibilities will be 
    # np.array([[0,0], [1, 0], [0, 1], [1, 1]])
    all_possibilities = np.array(np.meshgrid(*[range(x+1) for x in repn])).T.reshape(-1, len(repn))
    
    # We sort this array so that it goes from the least changes to be made, to the more changes to be made:
    # So we sort w.r.t the sum of elements in each row
    
    idx = np.argsort(np.sum(all_possibilities, 1))
    sorted_possibilities = np.take(all_possibilities, idx, axis=0)[::-1]  # In decreasing order
    
    candidates = []
    # Create the list of repeated characters
    repeated_chars = []
    for i in repm:
        repeated_chars.append(word[i])
    
    for ind in range(len(sorted_possibilities)):
        case = sorted_possibilities[ind]
        # Create the instance of repeated characters
        rep_chars_in_candidate = []
        for i, s in enumerate(repeated_chars):
            rep_chars_in_candidate.append(s * (case[i] + 1))
        # Now create the candidate
        candidate = word[:repm[0]]
        for index in range(len(repeated_chars)):
            candidate += rep_chars_in_candidate[index]
            if index + 1 != len(rep_chars_in_candidate):
                candidate += word[repm[index] + repn[index] + 1: repm[index + 1]]  # Take the part of the word inbetween repetitions
            else: continue
        
        candidate += word[repm[-1] + repn[-1] + 1:]
        candidates.append(candidate)
        
    return candidates


def do_correction(data, i):
    """ Function takes the data frame and the index and answers, with a bool, if 
        we have to try to correct the word there. We don't want to correct words where there are certain tokens :
        as they already have been resolved.
        The only token we will attempt the correction on is the rrrpp token.
        """
    correct = False
    tokens = data.iat[i, 3]
    if tokens == ['rrrpp']:
        correct = True  # By correct, we mean, we have to correct
    if tokens == []:
        correct = True
    return correct


def make_list_of_corrections(data):
    ls = []
    for i, corr in enumerate(data.correction):
        if not do_correction(data, i):
            continue
        elif " " in corr:
            for word in corr.split():
                ls.append(word)
        elif isinstance(corr, list):
            ls.extend(corr)
            
        elif not corr:  # Check if the list is empty. 
            continue
        else: 
            ls.append(corr)
    return ls


######################################################### Preprocessing Methods ##########################################################
#########################################################################################################################################
########################################################################################################################################


def create_reference_dictionary(filename, learning=True, limit=1000000000000000):
    """ This function creates a dictionary of references :
        a pandas dataframe with 5 columns : the word itself, the position of the word in the file as a tuple, 
        the correction of the words (initially the word itself), the tokens and an alternative correction column.
        learning separates the case where we take the words from a learning file or a testing file (they are written differently).
        """
    dat = {"filewords": [], "position": [], "correction" : [], "tokens": [], "alternative correction": []}
    with open(filename, 'r', encoding="utf8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        if i == limit: break
        else:
            s_line = line.strip()
            if not learning:
                # In this case each line starts by its number followed by a comma
                k = 0
                while s_line[k] != ',':
                    k += 1
                # Remove everything up to the comma (included)
                s_line = s_line[k + 1:]
                
            
            # Remove punctuation except #, !, <, >
            s_line = remove_punctuation(s_line)
            tweet_words = s_line.split()
            for j, w in enumerate(tweet_words):
                dat["filewords"].append(w)
                dat["position"].append((i, j))
                dat["correction"].append(w)
                dat["tokens"].append([])
                dat["alternative correction"].append(w)
    
    return pd.DataFrame(dat)

### Resolve methods for Preprocessing


def resolve_full_numbers(data):
    """ Function removes the fullmatched numbers using a regex, note that sometimes people write O for 0."""
    
    # Find the number indices throug a regular expression.
    number_indices = data.index[data.filewords.str.fullmatch(r"[0-9Oo]+")].tolist()
    
    # We have to exclude the cases with only Oo indices.
    only_Oo_indices = data.index[data.filewords.str.fullmatch(r"[Oo]+")].tolist()
    
    # We will interpret these numbers differently
    two_four_indices = data.index[data.filewords.str.fullmatch(r"2|4")].tolist()
    
    # Successively remove elements from each of the two lists from the first list
    temp1 = [item for item in number_indices if item not in only_Oo_indices]
    indices = [item for item in temp1 if item not in two_four_indices]
    
    # Correct the data frame :
    for ind in indices:
        data.iat[ind, 2] = ""
        tok = data.iat[ind, 3]
        tok.append('number')
        data.iat[ind, 3] = tok

        
        
def resolve_remaining_numbers(data):
    """ Function replaces any remaining punctuations by blanks or equivalent in characters e.g 4 -> for."""
    
    # Other remaining characters are ":","()".
    # Note that we are going through the correction and this function will be called after resolve_full_numbers
    two_for_ind = data.index[data.correction.str.contains("2|4")].tolist()
    
    # get any other number :
    path_ind = data.index[data.correction.str.contains("[\(\):1356789<>]")].tolist()
    # Apply corrections
    for i in two_for_ind:
        word = data.iat[i, 2]
        corrected_word = re.sub('2', 'to', word)
        corrected_word = re.sub('4', 'for', corrected_word)
        # In this precise case we'll also change the fileword column.
        
        data.iat[i, 0] = corrected_word
        data.iat[i, 2] = corrected_word
        
    for i in path_ind:
        # Remove all the other punctuation.
        word = data.iat[i, 2]
        corrected_word = re.sub('[\(\):1356789<>]+', '', word)
        # In this precise case we'll also change the fileword column.
        
        if 'abbrvt' in data.iat[i, 3]:
            continue
        else :
            data.iat[i, 0] = corrected_word
            data.iat[i, 2] = corrected_word

        
def resolve_laugh(data, haha):
    """ Function reads the data and replaces any word containing a term in haha by a token that expresses
        laughter. """
    for laugh in haha:
        # Find the laugh indices
        ha_ind = data.index[data.filewords.str.contains(laugh)].tolist()
        
        # Interpret these words as laughter :
        for ha_err in ha_ind:
            data.iat[ha_err, 2] = ""
            tokens = data.iat[ha_err, 3]
            tokens.append('haha')
            data.iat[ha_err, 3] = tokens
            
    #return data


def resolve_correction_into_words(data):
    """ Method takes the dataframe, and the index, and will separate the correction into
        multiple words (if it needs it). It will add rows for the new words and update the position of the words
        in the same line that are not affected by the correction."""
    
    for i, corr in enumerate(data.correction):
        data.iat[i, 2] = corr.split() 
    #return data
    
    
def resolve_onomatopee(data, instances):
    
    onos = split_list_of_tuples(instances)
    for i, ono in enumerate(onos):
        # Find the indices where there's this mistake.
        ono_ind = data.index[data.filewords == ono].tolist()

        # Correct the abbreviation and keep a token.
        for ono_err in ono_ind:
            data.iat[ono_err, 2] = instances[i][1]
            tokens = data.iat[ono_err, 3]
            tokens.append('ooo')
            # If there's additional information about the abbreviation (e.g. lol)
            if len(instances[i]) == 3:
                tokens.append(instances[i][2])
            data.iat[ono_err, 3] = tokens
    
    #return data
    
    
def resolve_obvious_mistakes(data, om):

    mistakes = split_list_of_tuples(om)
    for i, mis in enumerate(mistakes):
        # Find the indices where there's this mistake.
        mis_ind = data.index[data.filewords == mis].tolist()

        # Correct the abbreviation and keep a token.
        for mis_err in mis_ind:
            data.iat[mis_err, 2] = om[i][1]
            tokens = data.iat[mis_err, 3]
            tokens.append('mspld')
            # If there's additional information about the abbreviation (e.g. lol)
            if len(om[i]) == 3:
                tokens.append(om[i][2])
            data.iat[mis_err, 3] = tokens
    #return data
    
    
def resolve_abbreviations(data, abbr, keep_abbreviations):
    """ Function takes a list of abbreviation and their correction (a list of tuples), and checks the dataset to correct these mistakes
    """
    abbs = split_list_of_tuples(abbr)
    for i, abb in enumerate(abbs):
        # Find the indices where there's this abbreviation.
        abb_ind = data.index[data.filewords == abb].tolist()
        
        # Correct the abbreviation and keep a token.
        for abb_err in abb_ind:
            # If we want to keep the abbreviations without changing them in words.
            if not keep_abbreviations:
                data.iat[abb_err, 2] = abbr[i][1] 
            tokens = data.iat[abb_err, 3]
            tokens.append('abbrvt')
            # If there's additional information about the abbreviation (e.g. lol)
            if len(abbr[i]) == 3:
                tokens.append(abbr[i][2])
            data.iat[abb_err, 3] = tokens
    #return data
    
    
def resolve_punctuation(data, apostrophe_res):
    """ Function reads a pandas dataframe and removes the remaining punctuation marks.
        namely the apostrophes the exclamation marks and the hashtags. """
    
    ex_indices = data.index[data.filewords.str.contains('!')].tolist()
    hash_indices = data.index[data.filewords.str.contains('#')].tolist()
    ap_indices = data.index[data.filewords.str.contains('\'')].tolist()
    
    for exi in ex_indices:
        # Take the raw word
        w = data.iat[exi, 0]
        tokens = data.iat[exi, 3]
        tokens.append('exclmt')
        if repeat_exclamation(w):
            tokens.append('rrrpp')
        data.iat[exi, 3] = tokens
        w  = "".join([char for char in w if char != '!'])
        data.iat[exi, 2] = w
        
    for hi in hash_indices:
        w = data.iat[hi, 0]
        tokens = data.iat[hi, 3]
        tokens.append('hashtagg')
        data.iat[hi, 3] = tokens
        # Create the word without the "#", and put it in the correction column
        w = w.replace('#', '')
        #print(w)
        #w  = "".join([char for char in w if char != '#'])
        data.iat[hi, 2] = w
        
    for ai in ap_indices:
        w = data.iat[ai, 0]
        if w in apostrophe_res.keys():
            w = apostrophe_res[w]
        else:
            w  = "".join([char for char in w if char != '\''])
        data.iat[ai, 2] = w
        
    #return data         
    

def remove_stopwords_from_dataframe(data, stopword):
    
    for i, corr in enumerate(data.correction):
        data.iat[i, 2] = remove_stopwords(corr, stopword)
        data.iat[i, 3] = remove_stopwords(data.iat[i, 3], stopword)
    #return data

    
def stem_and_lemmatize_data(data):
    
    for i, corr in enumerate(data.correction):
        stemmed_corr = stem(corr)
        data.iat[i, 2] = lemmatize(corr)
    #return data
    
    
def correct_repeat_word(data, indices, onomatopee, abbreviations, keep_abbreviations):
    """ Function will correct the words, making sure to check among abbreviations and 
        onomatopees first before using the spell checker."""
    
    abbs = split_list_of_tuples(abbreviations)
    onos = split_list_of_tuples(onomatopee)
    spell = SpellChecker()
    for ind in indices:
        # Take the words among the corrections (because we could have hastags with repetitions etc)
        w = data.iat[ind, 2]
        cand_rep = construct_candidates_for_repeat_word(w)
        found = False
        for candidate in cand_rep:
            if candidate in abbs:
                # Check if the candidate is an abbreviation.
                found = True
                abb_index = abbs.index(candidate)
                if not keep_abbreviations:
                    data.iat[ind, 2] = abbreviations[abb_index][1]
                else: 
                    data.iat[ind, 2] = candidate
                tokens = data.iat[ind, 3]
                tokens.append('rrrpp')
                tokens.append('abbrvt')
                
                if len(abbreviations[abb_index]) == 3:
                    tokens.append(abbreviations[abb_index][2])
                break
                    
            elif candidate in onos:
                # Check if the canditate is an onomatopee.
                found = True
                ono_index = onos.index(candidate)
                data.iat[ind, 2] = onomatopee[ono_index][1]
                
                tokens = data.iat[ind, 3]
                tokens.append('rrrpp')
                tokens.append('ooo')
                
                if len(abbreviations[ono_index]) == 3:
                    tokens.append(abbreviations[ono_index][2])
                break
            
            elif len(spell.unknown([candidate])) != 1:  # Checks if the spelling is correct.
                found = True
                
                data.iat[ind, 2] = candidate
                
                tokens = data.iat[ind, 3]
                tokens.append('rrrpp')
                break
        if not found:
            # Take the candidate with the least amount of repetitions in it.
            data.iat[ind, 2] = cand_rep[-1]
            tokens = data.iat[ind, 3]
            tokens.append('rrrpp')
               
    #return data       

    
def correct_remainder(data, candidates = 3):
    
    spell = SpellChecker()
    corrections = make_list_of_corrections(data)
    
    mistakes = spell.unknown(corrections)
    
    # Compute checks for the progression of the function.
    N = len(mistakes)
    print("number of corrections {}".format(N))
    checkmark = N / 20
    
    for i, mistake in enumerate(mistakes):
        # Prints for where we are in the correction phase :
        if i % checkmark == 0:
            print("In correct_remainder : we are at {} completion".format(i / ( N / 20)))
        if len(spell.candidates(mistake)) <= candidates:
            correct = spell.correction(mistake)
            #mask = data.correction.apply(lambda x : mistake in x)
            #data[mask] = [correct]
            data.correction.replace([mistake], [correct])
            # Note that if the correction has several words, by construction they are written correctly.
            #for ind in mis_ind:
            #    data.iat[ind, 2] = [correct] # Correction is a list of words at this point
        else: continue
    

    
    
##################################################### TF-IDF ANALYSIS ############################################################# 
###################################################################################################################################
###################################################################################################################################
    

def compute_tf_idf_analysis(df1, df2=None):
    """ Method takes one or two dataframes and computes a tf idf analysis on them
        if there are two dataframes given, it will return three dataframes, (the first with the
        tf-idf analysis on df1, the second with the analysis on df2, the third with the analysis on both)
        """
    # Transform the corrections into a long string :
    list_of_corrections1 = turn_corrections_and_tokens_into_list(df1)
    corr_string1 = [" ".join(c for c in list_of_corrections1)]
    # Initialize the tf-idf vectorizer.
    vectorizer = TfidfVectorizer()
    
    matrix1 = vectorizer.fit_transform(corr_string1).todense()
    
    # Transform the results into pandas dataframes
    matrix1 = pd.DataFrame(matrix1, columns=vectorizer.get_feature_names())
    
    top_words1 = matrix1.sum(axis=0).sort_values(ascending=False)
    
    frames = []
    frames.append(top_words1)
    
    if df2 is None:
        return frames
    else:
        # Repeat the same methods as above for the second dataframe
        list_of_corrections2 = turn_corrections_and_tokens_into_list(df2)
        list_of_corrections1.extend(list_of_corrections2)
        corr_string2 = [" ".join(c for c in list_of_corrections2)]
        #corr_string12 = [" ".join(c for c in corrss)]
        #print(type(corr_string12[0]))
        
        corr_string12 = [" ".join(c for c in list_of_corrections1)] # For the analysis on both.
        matrix2 = vectorizer.fit_transform(corr_string2).todense()
    
        # Transform the results into pandas dataframes
        matrix2 = pd.DataFrame(matrix2, columns=vectorizer.get_feature_names())
        
        matrix12 = vectorizer.fit_transform(corr_string12).todense()
    
        # Transform the results into pandas dataframes
        matrix12 = pd.DataFrame(matrix12, columns=vectorizer.get_feature_names())
        
        top_words2 = matrix2.sum(axis=0).sort_values(ascending=False)
        top_words12 = matrix12.sum(axis=0).sort_values(ascending=False)
        frames.append(top_words2)
        frames.append(top_words12)
        return frames
    
    
####################################################### PREPROCESSING ###############################################################
#####################################################################################################################################
#####################################################################################################################################


def preprocess(filename, learning=True, orthograph=False, keep_abbreviations=False):
    """ This method takes a filename of tweets and returns a pandas data frame on which the preprocessing has been done.
        orthograph = False asks whether we want to try and run the spellchecker on the remaining words (it can be very long).
        learning = True asks whether we are in a learning phase or testing phase.
        """
    # Create the pandas dataframe (it strips the line number if we are in the testing phase)
    tic = time.perf_counter()
    df = create_reference_dictionary(filename, learning)
    toc = time.perf_counter()
    print("Created dictionnary : {} time passed".format(toc - tic))
    
    # Resolve successively the different preprocessing phases on df.
    # First check the punctuation :
    resolve_full_numbers(df)
    toc = time.perf_counter()
    print("First number resolution : {} time passed".format(toc - tic))
    
    # Then remove the abbreviations (some could feature numbers)
    resolve_abbreviations(df, twitter_abbreviations, keep_abbreviations)
    toc = time.perf_counter()
    print("abbreviation resolution : {} time passed".format(toc - tic))
    
    # Remove the remaining numbers    
    resolve_remaining_numbers(df)
    toc = time.perf_counter()
    print("Second number resolution : {} time passed".format(toc - tic))
    
    # Resolve the twitter obvious mistakes (such as dat for that etc)
    resolve_obvious_mistakes(df, twitter_obvious_mistakes)
    toc = time.perf_counter()
    print("Mistakes correction : {} time passed".format(toc - tic))
    
    # Take care of the onomatopeia
    resolve_onomatopee(df, onomatopees)
    toc = time.perf_counter()
    print("Onomatopeia resolution : {} time passed".format(toc - tic))
    
    # Take care of the laughs
    resolve_laugh(df, laughs)
    toc = time.perf_counter()
    print("laughter resolved : {} time passed".format(toc - tic))
    
    # Take care of the remaining punctuation
    resolve_punctuation(df, contractions)
    toc = time.perf_counter()
    print("Remaining punctuation taken care of : {} time passed".format(toc - tic))
    
    # Take care of words with repetitions
    # Find the indices where a repetion occurs
    rep_df = find_repeat_indices(df)
    correct_repeat_word(df, rep_df, onomatopees, twitter_abbreviations, keep_abbreviations)
    toc = time.perf_counter()
    print("Repeat words taken care of : {} time passed".format(toc - tic))
    
    # Up until now each correction was a string, we change them into a list of words
    resolve_correction_into_words(df)
    toc = time.perf_counter()
    print("Splitted the corrections : {} time passed".format(toc - tic))
    
    # We can now remove the stop words from the corrections don't remove the tokens as they are used in the last
    # correction method.
    remove_stopwords_from_dataframe(df, stopword_eg)
    toc = time.perf_counter()
    print("Stopwords removed : {} time passed".format(toc - tic))
    
    # Lastly, we can stem and lemmatize the corrections
    stem_and_lemmatize_data(df)
    toc = time.perf_counter()
    print("Stemmed and Lemmatized words : {} time passed".format(toc - tic))
    
    # Try to correct some spelling mistakes
    if orthograph:
        correct_remainder(df)
        toc = time.perf_counter()
        print("Corrected words : {} time passed".format(toc - tic))
        
    # Perhaps some words were corrected into stopwords.
    #remove_stopwords_from_dataframe(df, twitter_stopwords)
    toc = time.perf_counter()
    print("Stopwords removed : {} time passed".format(toc - tic))
        
    return df


########################################################### File Creation Method #########################################################
##########################################################################################################################################
##########################################################################################################################################

    
def write_new_test_file_from_df(data, filename):
    
    with open(filename, 'w', encoding="utf8") as fh:
        i = 0
        while i < len(data.position):
            pos = data.iat[i, 1]
            line_number = pos[0]
            # write the line number at the start of each line.
            fh.write("{},".format(line_number))
            j = i
            while still_in_same_tweet(data, j):
                for word in data.iat[j, 2]:  # write each word in correction.
                    fh.write("{} ".format(word))
                for token in data.iat[j, 3]: # write each token
                    fh.write("{} ".format(token))
                j += 1
            if j >= len(data.filewords):  # Check if we arrived at the end of the file.
                break
            # Still need to write the last word (because we basically check if the next word is in the tweet, and write
            # it if it has a successor in the same tweet)
            
            for word in data.iat[j, 2]:  # write each word in the last correction.
                fh.write("{} ".format(word))
            for token in data.iat[j, 3]: # write each token
                    fh.write("{} ".format(token))
                
            fh.write("\n")  # Write a new line
            i = j + 1
            

########################################## Method to change the tokens ################################################################
#######################################################################################################################################
######################################################################################################################################


def find_token_indices(data, token):
    indices = []
    for i, toks in enumerate(data.tokens):
        if token in toks:
            indices.append(i)

    return indices


def change_tokens(data, new_tokens):
    """ function takes a list of new tokens (it must be the same size as the number of tokens).
        This list describes the change of tokens it must be in the following order :
        ['ooo', 'abbrvt', 'hashtagg', 'exclmt', 'haha', 'rrrpp', 'mspld'].
        """
    
    for i, tok in enumerate(new_tokens):
        token_to_replace = list_of_tokens[i]
        # Find all the tokens to replace
        tok_ind = find_token_indices(data, token_to_replace)
        #print(tok_ind)
        for tok_err in tok_ind:
            tokens = data.iat[tok_err, 3]
            new_tokens = [tok if x == token_to_replace else x for x in tokens]
            data.iat[tok_err, 3] = new_tokens

    