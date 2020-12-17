from gensim.models.word2vec import Word2Vec
import multiprocessing




def word2vec_self_training_model(vocabulary, skip_vector_size, cbow_vector_size, window_size, epochs, seed=1, min_word_count=2):
    print("training word2vec skipgram model...")
    model_skipgram = Word2Vec(sentences=vocabulary,
                        size=skip_vector_size, 
                        window=window_size, 
                        iter=epochs,
                        seed=seed,
                        sg=1,
                        min_count=min_word_count,
                        workers=multiprocessing.cpu_count())
    print("training word2vec skipgram model terminated")
    
    print("training word2vec cbow model...")
    model_cbow = Word2Vec(sentences=vocabulary,
                        size=cbow_vector_size, 
                        window=window_size, 
                        iter=epochs,
                        seed=seed,
                        sg=0,
                        min_count=min_word_count,
                        workers=multiprocessing.cpu_count())
    print("training word2vec cbow model terminated")
    return model_skipgram, model_cbow



