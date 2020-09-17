import numpy as np
import re

def read_data(dataframe):
    
    data=[]
    label=[]
    m=dataframe.shape[0]
    
    for i in range(m):
        data.append(dataframe['text'][i])
        label.append(dataframe['target'][i])
    label=np.asarray(label)
    return data, label   

def preprocess_data(data):
    
    m=len(data)
    max_length=0
    vocab=set()
    for i in range(m):
        curr_sentence=data[i].lower().strip()
        curr_sentence=re.sub("[|#@!*.[\]_/{}();+:?%'\']",'',curr_sentence)
        
        words=curr_sentence.split()
        
        if max_length<len(words):
            max_length=len(words)
            
        for w in words:
            vocab.add(w)
    
    words_set=vocab
    
    vocab.add("<unk>"); vocab.add("<pad>")
    
    vocabulary={w:i for i, w in enumerate(sorted(vocab))}
            
    return max_length, words_set, vocabulary  

def data_to_index(data, vocabulary, words_set, Tx):
    
    m=len(data)
    
    X=np.ones((m,Tx))*vocabulary['<pad>']
    
    for i in range(m):
        
        words=data[i].lower().strip().split()
        j=0
        for w in words:
            if w in words_set:
                X[i,j]=vocabulary[w]
            else:
                X[i,j]=vocabulary["<unk>"]
                
            j=j+1
    
    return X
    
def one_hot(X_index, vocab_length):
    
    m, Tx=X_index.shape
    
    X_hot=np.zeros((m, Tx, vocab_length))
    
    for i in range(m):
        
        for j in range(Tx):
            
            X_hot[i,j,int(X_index[i,j])]=1
        
    return X_hot   
        
        
        
        
        
    
            
            