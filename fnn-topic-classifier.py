import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from time import localtime, strftime
from scipy.stats import spearmanr,pearsonr
import zipfile
import gc

# fixing random seed for reproducibility
random.seed(123)
np.random.seed(123)


def extract_ngrams(x_raw, ngram_range=(1,3), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', 
                   stop_words=[], vocab=set()):

    x = re.findall(token_pattern, x_raw)
    
    extracted_x = []
    
    a,b = ngram_range
    for n in range(a,b+1):
        for i in range(0, len(x), n):
            
            if len(x) <= (i+n):
                continue
            
            # Extracts the first and last word of the ngram where if a stop word is in the middle of an ngram it isn't removed.
            # This is due to the purpose of stop words is to remove redundant popular words.
            # However when a stop word is within (middle) an ngram, for example 'time to finish', there is
            # a relevancy between the first and last word that should be kept.
            
            first_word = x[i].lower()
            last_word = x[i+n-1].lower()
            stop_words = [word.lower() for word in stop_words]
            
            if (first_word in stop_words) or (last_word in stop_words):
                continue
            else:
                chunk = x[i:i+n]
                chunk = " ".join(chunk)
                
                
                if len(vocab) != 0 and chunk not in vocab:
                    continue
                    
                extracted_x.append(chunk)


        
    x = np.array(extracted_x)
    return x


def get_vocab(X_raw, ngram_range=(1,3), token_pattern=r'\b[A-Za-z][A-Za-z]+\b', 
              min_df=0, keep_topN=0, 
              stop_words=[]):

    vocab = set()
    
    df = Counter()
    ngram_counts = Counter()
    
    for x in X_raw:
        ngrams = extract_ngrams(x, ngram_range=ngram_range, stop_words=stop_words)
        
        for i in set(ngrams):
            df[i] += 1
        
        for i in ngrams:
            ngram_counts[i] += 1
        
        vocab.update(ngrams)
    
    if min_df != 0:
        pass
    
    
    if keep_topN != 0:
        pass
    
    
    return vocab, df, ngram_counts


# Split documents into tokens where each token exists in the vocab inputted.
def tokeniseX(X_raw, ngram_range=(1,3), stop_words=stop_words, vocab=set()):

    tokenised_X = []

    for x in X_raw:
        tokenised_x = extract_ngrams(x_raw=x, ngram_range=ngram_range, stop_words=stop_words, vocab=vocab)
        tokenised_X.append(tokenised_x)
        
    return tokenised_X


def convertWordsToIndices(X_words,word2id):
    X_indices = []
    for x in X_words:
        x_indices = [word2id[i] for i in x]
        
        if x_indices: 
            X_indices.append(x_indices)

    return X_indices


def removeEmpty(doc_indicies, Y):
    for pos, document in enumerate(train_indices): # Each document
        if not document:
            del doc_indicies[pos]
            del Y[pos]

    return np.array(doc_indicies, dtype=object), Y


def convertToOneHot(y_list):
    y_list_one_hot = []
    
    for y in y_list:
        y_one_hot = []
        
        if y-1 == 0:
            y_one_hot = [1,0,0]
        elif y-1 == 1:
            y_one_hot = [0,1,0]
        else:
            y_one_hot = [0,0,1]
            
        y_list_one_hot.append(y_one_hot)
            
    return np.array(y_list_one_hot)


def network_weights(vocab_size= len(vocab), embedding_dim = 300, 
                    hidden_dim=[1], num_classes=3, init_val = 0.1):

    embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))

    W = {}
    
    W[0] = embedding_matrix

    rows = embedding_dim
    for index, item in enumerate(hidden_dim):
        columns = item
        
        embedding_matrix = np.random.uniform(-0.1, 0.1, (rows, columns))        
        W[index+1] = embedding_matrix
        
        rows = columns
    

    output_index = len(hidden_dim) + 1
    
    W[output_index] = np.random.uniform(-0.1, 0.1, (rows, num_classes))
    
    return W


def softmax(z):
    z = z.reshape(-1)
    
    # Implemented a solution: https://stackoverflow.com/a/38250088 
    e = (np.exp(z - np.max(z)))
    sig = (e / e.sum(axis=0) + 1e-9)
    
    return sig


def categorical_loss(y, y_preds):
    l = - np.sum(y * np.log(y_preds + 1e-9))
    return l


def relu(z):
    a = z * (z > 0)
    return a

def relu_derivative(z):
    dz = 1 * (z >= 0)
    return dz

def dropout_mask(size, dropout_rate): 
    zeroes_amount = int(size*dropout_rate)
    
    dropout_vec = np.ones(size)
    dropout_vec[:zeroes_amount] = 0
    np.random.shuffle(dropout_vec)
    return dropout_vec

def apply_dropout_mask(a, dropout_rate): # Seperate Function for readability
    mask = dropout_mask(a.shape[1], dropout_rate).reshape(1, -1).astype('float32')
    a = a * mask
    
    return a, mask


def forward_pass(x, W, dropout_rate=0.2):
    out_vals = {}
    h_vecs = []
    a_vecs = []
    dropout_vecs = []
    
    # Embedding Layer
    w0 = W[0]
    document_w = np.array([w0[i] for i in x])
    h = (np.sum(document_w,axis=0)/len(x)).reshape(1, -1)
    a = relu(h).astype('float32')
    
    a, mask = apply_dropout_mask(a, dropout_rate)
    
    h_vecs.append(h)
    a_vecs.append(a)
    dropout_vecs.append(mask)
    
    # Hidden Layers
    for k in range(1,len(W)-1,1):
        h = np.dot(a, W[k]).astype('float32') 
        a = relu(h).astype('float32')
        
        a, mask = apply_dropout_mask(a, dropout_rate)
        
        h_vecs.append(h)
        a_vecs.append(a)
        dropout_vecs.append(mask)
        
    #Output Layer
    z = np.dot(a, W[max(W.keys())]).astype('float32')
    y_preds = softmax(z).reshape(1, -1)
    
    out_vals['h'] = h_vecs
    out_vals['a'] = a_vecs
    out_vals['dropout_vector'] = dropout_vecs
    out_vals['y_preds'] = y_preds

    return out_vals
    

def backward_pass(x, y, W, out_vals, lr=0.001, freeze_emb=False):
    h_vecs = out_vals['h']
    a_vecs = out_vals['a']
    dropout_vecs = out_vals['dropout_vector']
    y_preds = out_vals['y_preds']
    
    # Outer Layer
    output = max(W.keys())
    
    z = (a_vecs[-1] * dropout_vecs[-1]).astype('float32')
    g = (y_preds - y).astype('float32')
    g_w = (z.T * g).astype('float32')
    g = (np.dot(g, W[output].T)).astype('float32')
    
    W[output] = W[output] - (lr * g_w)
    
    # Hidden Layers
    curr_h = len(h_vecs)-1
    for k in range(len(W)-2, 0, -1):
        g = g  * (relu_derivative(h_vecs[curr_h])).astype('float32')           
        z = (a_vecs[curr_h-1] * dropout_vecs[curr_h-1]).astype('float32')
        g_w = (np.dot(z.T, g)).astype('float32')         
        g = (np.dot(g, W[k].T)).astype('float32')           

        W[k] = (W[k] - lr * g_w).astype('float32')    
        
        curr_h =- 1
    
    # Embedding Layers
    if freeze_emb is False:
        
        x_indicies = list(set(x))
        mask_of_ones = np.ones(len(x_indicies)).reshape(1,-1)
        
        g_w = np.dot(mask_of_ones.T, g)
        
        x_indicies.sort()        
        W[0][x_indicies] = W[0][x_indicies] - (lr * g_w)   
        
    return W   

def SGD(X_tr, Y_tr, W, X_dev, Y_dev, lr, 
        dropout_rate, epochs, tolerance, freeze_emb, 
        print_progress=True):
    
    training_loss_history =  []
    validation_loss_history =  []
    
    for epoch in range(epochs):
        
        # Shuffle
        shuffled = np.arange(len(X_tr))
        np.random.shuffle(shuffled)

        X_tr = X_tr[shuffled]
        Y_tr = Y_tr[shuffled]
        
        
        epoch_losses = []        
        
        for x, y in zip(X_tr, Y_tr):           
            out_vals = forward_pass(x, W, dropout_rate)
            W = backward_pass(x, y, W, out_vals, lr, freeze_emb)
            
            y_preds = out_vals['y_preds']
            loss = categorical_loss(y, y_preds)
            epoch_losses.append(loss)            
        
        mean_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        training_loss_history.append(mean_epoch_loss)
        
        epoch_losses = []        
        
        for x, y in zip(X_dev, Y_dev):           
            out_vals = forward_pass(x, W, dropout_rate)
            
            y_preds = out_vals['y_preds']
            loss = categorical_loss(y, y_preds)
            
            epoch_losses.append(loss)
            
    
        if len(epoch_losses) <= 2: #If there are enough elements in validation loss list to compare
            if abs(epoch_losses[-1] - epoch_losses[-2]) < tolerance:
                break                
        
        mean_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        validation_loss_history.append(mean_epoch_loss)
        
        if print_progress:
            print('Epoch:',epoch+1,'   Tr:',training_loss_history[-1],'   Val:',validation_loss_history[-1])
            
    training_loss_history = np.array(training_loss_history)
    validation_loss_history = np.array(validation_loss_history)
    
    return W, training_loss_history, validation_loss_history


def grid_search(grid_search_parameters,
                X_tr, Y_tr, W, X_dev, Y_dev,
                tolerance, freeze_emb, hidden_dim=[],
                print_progress=False):
    
    epoch_list = grid_search_parameters['epoch']
    lr_list = grid_search_parameters['learning_rate']
    emb_list = grid_search_parameters['embedding_size']
    dr_list = grid_search_parameters['dropout_rate']
    
    
    parameters_df = pd.DataFrame()
    parameters = {}
    
    parameters['epochs'] = []
    parameters['learning_rate'] = []
    parameters['embedding_size'] = []
    parameters['dropout_rate'] = []
    parameters['training_loss'] = []
    parameters['validation_loss'] = []

    
    for epochs in epoch_list:
        for lr in lr_list:
            for embedding_size in emb_list:
                for dropout_rate in dr_list:
                    W = network_weights(vocab_size=len(vocab),embedding_dim=embedding_size,
                                        hidden_dim=hidden_dim, num_classes=3)


                    W, tr_loss, dev_loss = SGD(train_indices, Y_train_onehot,
                                               W,
                                               X_dev=dev_indices, 
                                               Y_dev=Y_dev_onehot,
                                               lr=lr, 
                                               dropout_rate=dropout_rate,
                                               freeze_emb=freeze_emb,
                                               tolerance=0.001,
                                               epochs=epochs,
                                               print_progress=print_progress)
                    
                    best_training_loss = tr_loss[-1]
                    best_validation_loss = dev_loss[-1]

                    print(epochs,lr,embedding_size,dropout_rate,best_training_loss,'TRAINING')
                    print(epochs,lr,embedding_size,dropout_rate,best_validation_loss,'VALIDATION')
    
                    parameters['epochs'].append(epochs)
                    parameters['learning_rate'].append(lr)
                    parameters['embedding_size'].append(embedding_size)
                    parameters['dropout_rate'].append(dropout_rate)
                    parameters['training_loss'].append(best_training_loss)
                    parameters['validation_loss'].append(best_validation_loss)

    parameters_df = pd.DataFrame(zip(*parameters.values()),
                                      columns =['epochs', 'learning_rate', 'embedding_size', 'dropout_rate', 'training_loss', 'validation_loss'])

    parameters_df['avg_loss'] = parameters_df[['training_loss', 'validation_loss']].mean(axis=1)

    return parameters_df



def getBestParameters(df):
    best_training_loss = df[df.training_loss == df.training_loss.min()]
    best_validation_loss = df[df.validation_loss == df.validation_loss.min()]
    best_avg_loss = df[df.avg_loss == df.avg_loss.min()]
    
    print("Parameters with best TRAINING LOSS:")
    display(best_training_loss)
    print("Parameters with best VALIDATION LOSS:")
    display(best_validation_loss)
    print("Parameters with best average between the TRAINING and VALIDATION LOSS:")
    display(best_avg_loss)





# ============= TRAINING DATA RETRIEVAL
# For each dataset collect the documents as a single list. Easier to work with than using the Dataframes in current form.
train_df=pd.read_csv('data_topic/train.csv', names=['Class', 'Document'])
dev_df=pd.read_csv('data_topic/dev.csv', names=['Class', 'Document'])
test_df=pd.read_csv('data_topic/test.csv', names=['Class', 'Document'])

train_X = train_df['Document'].tolist()
dev_X = dev_df['Document'].tolist()
test_X = test_df['Document'].tolist()

stop_words = ['a','in','on','at','and','or', 
              'to', 'the', 'of', 'an', 'by', 
              'as', 'is', 'was', 'were', 'been', 'be', 
              'are','for', 'this', 'that', 'these', 'those', 'you', 'i', 'if',
             'it', 'he', 'she', 'we', 'they', 'will', 'have', 'has',
              'do', 'did', 'can', 'could', 'who', 'which', 'what',
              'but', 'not', 'there', 'no', 'does', 'not', 'so', 've', 'their',
             'his', 'her', 'they', 'them', 'from', 'with', 'its']

# ============= PRE-PROCESSING PIPELINE AND CREATES INPUT REPRESENTATIONS

vocab, df, ngram_counts = get_vocab(train_X, ngram_range=(1,1))

id2word = enumerate(vocab)
id2word = dict(id2word)
word2id = {word: id for id, word in enumerate(vocab)}

train_words = tokeniseX(train_X, ngram_range=(1,1), vocab=vocab)
dev_words = tokeniseX(dev_X, ngram_range=(1,1), vocab=vocab)
test_words = tokeniseX(test_X, ngram_range=(1,1), vocab=vocab)

train_indices = convertWordsToIndices(train_words, word2id)
dev_indices = convertWordsToIndices(dev_words, word2id)
test_indices = convertWordsToIndices(test_words, word2id)

Y_train = train_df['Class'].values
Y_dev = dev_df['Class'].values
Y_test = test_df['Class'].values


train_indices, Y_train = removeEmpty(train_indices, Y_train)
dev_indices, Y_dev = removeEmpty(dev_indices, Y_dev)
test_indices, Y_test = removeEmpty(test_indices, Y_test)


Y_train_onehot = convertToOneHot(Y_train)
Y_dev_onehot = convertToOneHot(Y_dev)
Y_test_onehot = convertToOneHot(Y_test)


# ============= HYPERPARAMETER TUNING VIA GRID SEARCH

grid_search_parameters = {}
grid_search_parameters['learning_rate'] = [0.1, 0.05, 0.075]
grid_search_parameters['embedding_size'] = [50, 300, 500]
grid_search_parameters['dropout_rate'] = [0.2, 0.5, 0.8]
grid_search_parameters['epoch'] = [10]

parameters_df = grid_search(grid_search_parameters,
                                train_indices, Y_train_onehot,
                                W,
                                X_dev=dev_indices, 
                                Y_dev=Y_dev_onehot,
                                tolerance=0.001,
                                freeze_emb=False)

pd.set_option('display.max_rows', None)
parameters_df

getBestParameters(parameters_df)

W = network_weights(vocab_size=len(vocab),embedding_dim=500,
                hidden_dim=[], num_classes=3)


# ============= NEURAL NETWORK MODEL CREATION
    
W, loss_tr1, dev_loss1 = SGD(train_indices, Y_train_onehot,
                            W,
                            X_dev=dev_indices, 
                            Y_dev=Y_dev_onehot,
                            lr=0.075, 
                            dropout_rate=0.8,
                            freeze_emb=False,
                            tolerance=0.001,
                            epochs=40)

# ============= EVALUATION

plt.plot(loss_tr1, label = "Training Loss")
plt.plot(dev_loss1, label = "Validation Loss")

plt.title("Average Embedding Neural Network - Learning Process")
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.xticks(np.arange(0, len(loss_tr1), 2.0))

plt.legend()
plt.show()


preds_te_average = np.array([np.argmax(forward_pass(x, W, dropout_rate=0.0)['y_preds'])+1
            for x,y in zip(test_indices, Y_test)])

print('Accuracy:', accuracy_score(Y_test,preds_te_average))
print('Precision:', precision_score(Y_test,preds_te_average,average='macro'))
print('Recall:', recall_score(Y_test,preds_te_average,average='macro'))
print('F1-Score:', f1_score(Y_test,preds_te_average,average='macro'))