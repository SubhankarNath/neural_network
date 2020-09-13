import numpy as np
from keras.utils import to_categorical

def load_data():
    
    data=[]
    labels=[]
    with open('train.txt', 'r', encoding='utf8') as f:
        
        for line in f:
            line=line.lower().strip().split(';')
            data.append(line[0])
            labels.append(line[1])
    
    m=len(data)
    max_length=0
    for i in range(m):
        words=data[i].split()
        
        if max_length<len(words):
            max_length=len(words)    
        
    return data, labels, max_length   

def load_GloVe():
    
    word_to_vec={}
    Glove_words=set()
    
    with open("glove.6B.100d.txt",'r', encoding='utf8') as f:
        for line in f:
            
            line=line.strip().split()
            word= line[0]
            Glove_words.add(word)
            
            word_to_vec[word]=np.asarray(line[1:], dtype=np.float64)
            
    word_to_index={w:i+1 for i, w in enumerate(sorted(Glove_words))}        
    
    return word_to_vec, Glove_words, word_to_index

def data_to_index(data, word_to_index, Glove_words, Tx):
    
    m=len(data)
    X_index=np.zeros((m, Tx))
    
    for i in range(m):
        words=data[i].split()
        j=0
        for w in words:
            if w in Glove_words:
                X_index[i,j]=word_to_index[w]
            else:
                X_index[i,j]=0                    ### unknown word is mapped to index 0...................
            j=j+1    
                
    return X_index             
                

def labels_to_one_hot(labels):
    
    label_set=set(labels)
    
    label_dict={l: i for i, l in enumerate(sorted(label_set))}
    
    indices=[]
    
    m=len(labels)
    for i in range(m):
        
        indices.append(label_dict[labels[i]])
        
    y_hot=to_categorical(indices, num_classes=len(label_set))    
    
    return label_dict, y_hot


def load_val_data():
    
    data=[]
    labels=[]
    with open('val.txt', 'r', encoding='utf8') as f:
        
        for line in f:
            line=line.lower().strip().split(';')
            data.append(line[0])
            labels.append(line[1])
    
    m=len(data)
    max_length=0
    
    for i in range(m):
        words=data[i].split()
        
        if max_length<len(words):
            max_length=len(words)    
        
    return data, labels, max_length 


