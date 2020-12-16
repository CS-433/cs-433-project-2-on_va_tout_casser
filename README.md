# Text Classification on Tweets

The purpose of this project was to test and compare different text classification methods. Among the one we have tested (c.f. the latex report), Word2Vec on a specific Neural Network worked the best. This README contains infos about the external libraries we used, infos about the structure of the code, and finally about parameter choosing.

## External libraries
This is a quick overview of the libraries used.
### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the libraries used in the project.

```bash
pip install tensorflow
pip install keras
pip install gensim
pip install nltk
pip install pickle
pip install numpy
pip install tensorflow-hub
pip install matplotlib
```

### Documentation

#### TensorFlow and keras

TensorFlow is a complete and open source platform for any  of ML tasks. It takes care of the training and the testing parts. Keras is  just a high-level accessible API built on top of it. You can find more about TensorFlow at [TensorFlow](https://www.tensorflow.org).

#### Gensim

As said on [Gensim](https://pypi.org/project/gensim/), "Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community." The part that is used is mostly NLP. Gensim offered the tools for building the Word2Vec models, trained with skip-gram or CBOW. (For more infos, refer to [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)).

#### NLTK

The NLTK library was used to do the preprocessing part.

#### Numpy & matplotlib & pickle

These are general libraries used by everyone. Numpy is for data representation, matplotlib is for data visualization, and pickle is for data storage.


## The ML program

This folder should contain a file run.py which produces the same outputs as on AICrowd for our team. 

### File organization

In order to execute the code, you need to have the training data and the test data at the right place. You can specify the path to this data with the following variable: 

```python
path = "datasets\\"
pos = "pos_train_full.txt"
neg = "neg_train_full.txt"
test = "test_data.txt"
```
The folder pointed should contain the files with variable name "pos", "neg" and "test". These should be the file names of the data given by the course github for the project.  

### Structure

First, our program does a little bit of pre-processing (in fact, we did a bigger pre-processing, but it turned out that is does not increase accuracy!). After that, a NN is employed. The figure depicted below shows the global structure of the NN that is being used. The number displayed may vary depending on the given model parameters. The first layer is an embedding (see latex document). Next, we split the NN in three branches, each containing a convolutional layer and a so-called Global-Max-Pooling (which is basically an ordinary max pooling layer with a pool size equal to the input size). After concatenation, we put a dense layer (= fully connected layer) and a dropout layer (which randomly sets input units to 0 with a given frequency at each step during training time to prevent overfitting.) At the end, the activation layer which finishes the classification.

![NN](https://github.com/CS-433/cs-433-project-2-on_va_tout_casser/blob/main/NN_Model.png?raw=true)

### Parameters

Bellow you will find a list of the model parameters. 

```python

skip_vector_size = 0
cbow_vector_size = 50
window_size = 4
epochs_word2vec = 15
min_word_count =  2

```
These values specifies the Word2Vec embedding layer construction parameters. The first two define the size of the CBOW and the Skip-Gram to use. Indeed, we decided to try a mix of both models. The window_size is the maximum distance between the current and the predicted word within a tweet. The number of epochs it the number of iterations over the dataset to be done for the Word2Vec embedding. The last parameters represents the minimum number of appearance of a word in order for it to be considered by the algorithm.
```python

trainable = False
filter_number = 100
dense_number = 256
dropout = 0.1
train_percentage_validation = 0.90
```

The trainable argument will decide whether or not the NN will consider the vector values resulting from the embedding layer should be considered as trainable or not. When set to True, this will effectively augment considerably the number of parameters. You may want to decrease the number of NN epochs to avoid a long computation and overfitting. The filter and the dense numbers are for the convolutional and dense layer. Increasing them will increase the number of parameters and thus the complexity of the overall model. The dropout percentage is obviously the regularization parameter of the dropout layer to avoid overfitting (c.f. structure part). The last parameters decides how much of the data will be used for training and for validation. A standard value for validation is about 10%. 


```python
epochs_nn = 15
batch_size = 50
```

The NN epochs are the number of iterations to do with the data over the NN. If this value is to high, this may lead to overfitting. The batch size is the amount of data to be trained on at each epoch at once. When increased, this will use more RAM and less time. When decreased, it will yield more accurate results. 


```python
use_pickle = False
```

Finally, you can set this variable to True in order to re-use the data splitting and the embedding matrix if you have already run the code before. 
