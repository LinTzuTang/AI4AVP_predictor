from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,accuracy_score,f1_score,matthews_corrcoef,confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt

def split(data_encoded, labels, test_size=0.1, random_state = 10, save = True, output_root = None):
    train_data, test_data, train_labels, test_labels = train_test_split(data_encoded, labels, test_size=0.1, random_state = 10, stratify = labels)
    if save:
        output_root = output_root or os.getcwd()
        if not os.path.isdir(output_root):
            os.mkdir(output_root)
        json.dump(train_data.tolist(), open(os.path.join(output_root,'train_data.json'), "w"), indent=4) 
        json.dump(test_data.tolist(), open(os.path.join(output_root,'test_data.json'), "w"), indent=4) 
        json.dump(train_labels.tolist(), open(os.path.join(output_root,'train_labels.json'), "w"), indent=4) 
        json.dump(test_labels.tolist(), open(os.path.join(output_root,'test_labels.json'), "w"), indent=4) 
        print('train test encoded data and labels saved')
    return  train_data, test_data, train_labels, test_labels

# show train history
#show_train_history(t_m ,'accuracy','val_accuracy')
#show_train_history(t_m ,'loss','val_loss')
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
# evalute metric (accuracy,precision,sensitivity,specificity,f1,mcc)
def metric_array(test_data, test_labels, model):
    labels_score = model.predict(test_data)
    accuracy = accuracy_score(test_labels, labels_score.round())
    confusion = confusion_matrix(test_labels, labels_score.round())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP / float(TP + FP)
    sensitivity = TP / float(FN + TP)
    specificity = TN / float(TN + FP)
    f1 = f1_score(test_labels, labels_score.round())
    mcc = matthews_corrcoef(test_labels, labels_score.round()) 
    metric = [accuracy,precision,sensitivity,specificity,f1,mcc]
    return {'accuracy':accuracy,
           'precision':precision,
           'sensitivity':sensitivity,
           'specificity':specificity,
           'f1':f1,
           'mcc':mcc}