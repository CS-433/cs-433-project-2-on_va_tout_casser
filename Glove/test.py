import numpy as np
a = np.array([1,2,3,4])
b = np.array([6,7,8,8,1000])
with open("NUMPYSAVE.npy",'wb') as f:
    np.save(f,a)
    np.save(f,b)
with open("NUMPYSAVE.npy",'rb') as f:
    g = np.load(f)
    j = np.load(f)