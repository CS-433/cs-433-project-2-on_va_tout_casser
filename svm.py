import numpy as np
from sklearn import svm
import pickle

x_path = 'data_vectors/x.npy'
y_path = 'data_vectors/y.npy'

def get_svm_model(get_stored , X=0, y=0,regularization_factor=1.0, kernel='sigmoid', degree=3, stopping_criterion=1e-3, max_iter=1000, name =''):
    if get_stored:
        with open('classifiers/' + name, 'rb') as f:
            classifier = pickle.load(f)
        return classifier
    else : 
        classifier = svm.SVC(C = regularization_factor,
                        kernel = kernel,
                        degree = degree,
                        tol = stopping_criterion,
                        max_iter = max_iter,
                        verbose=True)

        print("training svm model...")
        classifier.fit(X, y)
        print("training svm model terminated")

        with open('classifiers/' + name, 'wb') as f4:
            pickle.dump(classifier, f4)
        return classifier

if __name__ == '__main__':
    with open(x_path, 'rb') as f1:
        X = np.load(f1)
    with open(y_path, 'rb') as f2:
        y = np.load(f2)

    #TODO: have to convert X_test it
    with open('dataset/x_test.npy.txt', 'rb') as f3:
        X_test = np.load(f3)
    
    get_stored = False
    possible_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    kernel = 'sigmoid'
    #if polynomial kernel, otherwise ignored
    degree = 3
    #it uses L2 norm penalty
    regularization_factor = 1.0
    max_iter = -1
    stopping_criterion = 1e-3
    name = 'svm_' + kernel + '_deg_' + str(degree) + '_reg_' + str(regularization_factor) + 'crit_' + str(stopping_criterion)+'.pkl'
    classifier = get_svm_model(get_stored, X, y, regularization_factor, kernel, degree,stopping_criterion, max_iter, name)
 

    try:
        resFile = open("submission-test.csv","w")
        resFile.write("Id,Prediction\n")
        for i in range(X_test.shape[0]):
            pred = classifier.predict(X_test[i,:].reshape(1,-1))[0]
            resFile.write(str(i + 1) + "," + str(pred) + "\n")
    finally:
        resFile.close()

    print("svm terminated")