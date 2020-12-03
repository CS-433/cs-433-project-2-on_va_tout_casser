import numpy as np
from sklearn import svm
import pickle

if __name__ == '__main__':
    with open('data_vectors/x.npy', 'rb') as f1:
        X = np.load(f1)
    with open('data_vectors/y.npy', 'rb') as f2:
        y = np.load(f2)

    #TODO: have to convert X_test it
    with open('dataset/x_test.npy.txt', 'rb') as f3:
        X_test = np.load(f3)
    
    possible_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    kernel = 'sigmoid'
    #if polynomial kernel, otherwise ignored
    degree = 3
    #it uses L2 norm penalty
    regularization_factor = 1.0
    max_iter = 1000
    stopping_criterion = 1e-3
    
    classifier = svm.SVC(C = regularization_factor,
                        kernel = kernel,
                        degree = degree,
                        tol = stopping_criterion,
                        max_iter = max_iter)

    print("training svm model...")
    classifier.fit(X, y)
    print("training svm model terminated")

    name = 'svm_' + kernel + '_deg' + str(degree) + '_reg' + str(regularization_factor) + 'crit' + str(stopping_criterion)+'.pkl'
    with open('classifiers/' + name, 'wb') as f4:
        pickle.dump(classifier, f4)


    #TODO : store correctly results
    results =[]
    for i in X_test.shape[0]:
        results.append(classifier.predict(X_test[i,:])[0])
    

    print("terminated")
