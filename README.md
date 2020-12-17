# Text Classification on Tweets

The purpose of this project was to test and compare different text classification methods. Among the one we have tested (c.f. the latex report), Word2Vec on a specific Neural Network worked the best. This README contains infos about the external libraries we used, infos about the structure of the code, and finally about parameter choosing.

## External libraries
This is a quick overview of the libraries used.

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the libraries used in the project. Make sure to have the latest version of pip : use "pip install --upgrade pip" and in Windows "python -m pip install --upgrade pip". Make sure to use at least the 3.7.* python version. To use Keras, check if you have the 2.2 tensorflow version, or higher. 


```bash
pip install tensorflow
pip install keras
pip install gensim
pip install nltk
pip install pickle-mixin
pip install numpy
pip install tensorflow-hub
pip install matplotlib
pip install sklearn
pip install pandas
pip install pyspellchecker
```
### Google Colab
If you are facing any issues with these packages installation or if you want simply to run the code, we provided a Google Colab Notebook at this [link](https://drive.google.com/drive/folders/1kuJXFgFZdqRpmy8_crkD2OBNvvo1FjoF?usp=sharing). Nevertheless, we encourage you to download the python scripts, because in our point of view they could be more readable.
### Documentation

#### TensorFlow and keras

TensorFlow is a complete and open source platform for any  of ML tasks. It takes care of the training and the testing parts. Keras is  just a high-level accessible API built on top of it. You can find more about TensorFlow at [TensorFlow](https://www.tensorflow.org).

#### Gensim

As said on [Gensim](https://pypi.org/project/gensim/), "Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community." The part that is used is mostly NLP. Gensim offered the tools for building the Word2Vec models, trained with skip-gram or CBOW. (For more infos, refer to [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)).

#### NLTK

The NLTK library was used to do the preprocessing part.
Note on the use of the NLTK library : We use this library to obtain lists of words. So if the specified list doesn't exist, it must be downloaded (using the line from the error message).

#### Sklearn

Sklearn was used for the tf-idf analysis

#### Pandas 

It is used in the preprocessing, to create our dataframes

#### Pyspellchecker

Used in the preprocessing in the spell checking (optional phase)

#### Numpy & matplotlib & pickle

These are general libraries used by everyone. Numpy is for data representation, matplotlib is for data visualization, and pickle is for data storage.


## The ML program

This folder should contain a file run.py which produces the same outputs as on AICrowd for our team. The only scripts that are used during the execution are : **run.py, word2vec_final.py, cnn.py, utils.py** and **tweets_tools.py**. The other are the other models we tried and are present for completeness. 

### File organization

In order to execute the code, you need to add the training data and the test data into the datasets folder. You can specify the path to this data with the following variable: 

```python
path = "datasets/"
pos_file = "pos_train_full.txt"
neg_file = "neg_train_full.txt"
test_file = "test_data.txt"
```
The folder pointed should contain the files with variable name "pos_file", "neg_file" and "test_file". These should be the file names of the data given by the course github for the project. Make sure while running the code to be on the right location. For Windows, change the "/" into "&#92;&#92;" for all the pathes.

```python
path_processed_dataset = "processed_dataset/" 
path_results = "results/"
path_model = "model/"
```
The path_processed_dataset is the folder where the processed dataset used for the CNN is stored, the path_results where the results should be stored and path_model where the CNN model is stored after the training. All these folders have already created.
```python
name_model = "Word2Vec_CNN"
```
The name_model parameter is used when we store the results to know what model we have computed.

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
use_pickle = False
```

You can set this variable to True in order to re-use the data splitting and the embedding matrix (resulting to the given Word2Vec parameters) if you have already run the code before. If set to True, do not change the parameters of Word2Vec: it will change nothing to the computation (only the previous parameters will be used), but it will change (wrongly) the values of the parameters used in the result files.

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
activation_function = 'relu'
loss = 'binary_crossentropy'
optimizer = 'adam'
```
The activation function will be the one used all over the CNN, except in the last layer where the sigmoid activation function is kept to obtain a probability as output. The loss parameter is the loss function we want to minimize, other loss such as 'hinge' could be used. The optimizer parameter can also be changed, for example to 'SGD' if we wnat to use stochastic gradient descent.

### Runtime
Using GPU's, expect to pass approximatively 2 hours per CNN epochs. In overall, it takes for the program to finish in average 10 hours.
