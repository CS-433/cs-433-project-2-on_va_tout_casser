import pickle
import matplotlib.pyplot as plt



def data_path(path, neg, pos, test):
    return path + neg, path + pos, path + test 
    

def store_processed_data(X, X_test, y, embedding_matrix, max_length, path_processed_dataset):
    with open(path_processed_dataset+ "X"+ ".txt", "wb") as fp1:
        pickle.dump(X, fp1)
    with open(path_processed_dataset+ "X_test"+ ".txt", "wb") as fp2:
        pickle.dump(X_test, fp2)
    with open(path_processed_dataset+ "y"+ ".txt", "wb") as fp3:
        pickle.dump(y, fp3)
    with open(path_processed_dataset+ "embedding_matrix"+ ".txt", "wb") as fp4:
        pickle.dump(embedding_matrix, fp4)
    with open(path_processed_dataset+ "max_length"+ ".txt", "wb") as fp5:
        pickle.dump(max_length, fp5)
    
def recover_processed_data(path_processed_dataset):
    with open(path_processed_dataset+ "X"+ ".txt", "rb") as fp1:
        X = pickle.load(fp1)
    with open(path_processed_dataset+ "X_test"+ ".txt", "rb") as fp2:
        X_test = pickle.load(fp2)
    with open(path_processed_dataset+ "y"+ ".txt", "rb") as fp3:
        y = pickle.load(fp3)
    with open(path_processed_dataset+ "embedding_matrix"+ ".txt", "rb") as fp4:
        embedding_matrix = pickle.load(fp4)
    with open(path_processed_dataset+ "max_length"+ ".txt", "rb") as fp5:
        max_length = pickle.load(fp5)
    return X, X_test, y, embedding_matrix, max_length






def draw_graph_validation_epoch(history, date_time, path_results, loss_name):
    history_dict = history.history

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss '+ loss_name.replace('_', ' ').title())
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(path_results + date_time + '.png')


def store_submission(name_submission, pred_int, path_results):
    resFile = open(path_results + name_submission +  ".csv","w")
    resFile.write("Id,Prediction\n")
    for i in range(len(pred_int)):
        predicted = pred_int[i]
        if(predicted == 0):
            predicted = -1
        elif(predicted != 1):
           print("Prediction type error on ",predicted)
        resFile.write(str(i + 1)+","+str(int(predicted))+"\n")
    resFile.close()


def store_results(name_list, value_list, path_results, date_time, name_model):
  result_file = open(path_results + "RESULTS.txt", "a+")
  result_file.write(name_model + "\n")
  result_file.write(date_time + "\n")
  for i in range(len(name_list)):
    result_file.write(name_list[i] + " :  " + value_list[i] + "\n")
  result_file.write("\n" + "\n")