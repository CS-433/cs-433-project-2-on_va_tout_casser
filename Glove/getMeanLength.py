import numpy as np
a = list()
with open("C:\\Users\\lucas\\Documents\\cs-433-project-2-on_va_tout_casser\\twitter-datasets\\train_neg_full.txt",encoding='utf-8',errors = "namereplace") as f :
    for t in f:
        a.append(t.strip(" "))
a = np.array(a)
a = np.vectorize(lambda x : len(x))(a)

print(np.mean(a),np.std(a),np.mean(a)+ 2* np.std(a))