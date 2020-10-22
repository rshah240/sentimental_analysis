import torch
import torch.nn as nn
from string import punctuation
from collections import Counter
import hyperparameters
import numpy as np
import torch.nn.functional as F


device = ('cuda' if torch.cuda.is_available() else 'cpu')

def get_data():
    """Function for getting data"""
    with open('data/reviews.txt','r') as f:
        reviews = f.read()
    return reviews

def get_vocab(reviews):
    """Function for getting vocab """
    reviews.lower()
    all_text = ''.join([c for c in reviews if c not in punctuation])
    words = all_text.split()
    count = Counter(words)
    vocab = sorted(count,key=count.get,reverse=True)
    return vocab

def prerprocess(input_string,vocab):
    """Function for preprocessing the input string"""
    input_string = input_string.strip()
    input_string = input_string.lower()
    input_string = ''.join([c for c in input_string if c not in punctuation])
    if len(input_string) == 0:
        raise('The input string does not contain any character')
    else:
        vocab_to_int = {word:ii for ii,word in enumerate(vocab,1)}
        reviews_int = [[vocab_to_int[word] for word in input_string.split()]]
        return reviews_int

def pad_features(reviews_int):
    """Function for padding and truncating reviews to make it of seq_length"""
    seq_length = hyperparameters.seq_length
    features = np.zeros((len(reviews_int),seq_length),dtype =int)

    for i,row in enumerate(reviews_int):
        features[i,-len(row):] = np.array(row)[:seq_length]
    return features

class Sentimental_Analysis(nn.Module):
    """Defining the architecture of the network"""
    def __init__(self,input_size,hidden_dim,embedding_dim,output_size,n_layers):
        """Defining the parameters of the architecture"""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.input_size = input_size

        self.embedding = nn.Embedding(self.input_size,self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers=self.n_layers,batch_first=True)
        self.fc = nn.Linear(self.hidden_dim,self.output_size)
        self.dropout = nn.Dropout(hyperparameters.dropout_prob)
    def forward(self,x,hidden):
        """Performing forward pass and returning hidden state and output"""

        batch_size = x.size(0)
        x = x.long()
        #getting output from embedding and lstm layer
        x = self.embedding(x)
        x,hidden = self.lstm(x,hidden)
        #stacking lstm outputs
        x = x.contiguous().view(-1,self.hidden_dim)

        #dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        x = F.logsigmoid(x)
        x = torch.exp(x)
        x = x.view(batch_size,-1)
        x = x[:,-1] #using the output of the last cell


        return x,hidden
    def init_hidden(self,batch_size):
        weights = next(self.parameters()).data

        hidden = (weights.new(self.n_layers,batch_size,self.hidden_dim).to(device),
                  weights.new(self.n_layers,batch_size,self.hidden_dim).to(device))
        return hidden
def predict(net,features):
    net.to(device)
    net.eval()
    features = torch.from_numpy(features)
    features = features.to(device)
    h = net.init_hidden(1)
    h = tuple([each.data for each in h])
    output,h = net(features,h)
    pred = torch.round(output.squeeze())
    if(pred.item() == 1):
        print("Its a positive review")
    else:
        print("Its a negative review")


def main():
    input_string = str(input("Enter string to predict"))
    reviews = get_data()
    vocab = get_vocab(reviews)
    reviews_int = prerprocess(input_string,vocab)
    features = pad_features(reviews_int)

    with open('Sentimental_Analysis.pt','rb') as f:
        model_dict = torch.load(f)

    input_size = model_dict['input_size']
    hidden_dim = model_dict['hidden_dim']
    embedding_dim = model_dict['embedding_dim']
    n_layers = model_dict['n_layers']
    output_size = model_dict['output_size']
    state_dict = model_dict['state_dict']

    net = Sentimental_Analysis(input_size=input_size,hidden_dim=hidden_dim,
                               embedding_dim=embedding_dim,
                               n_layers = n_layers,output_size = output_size)
    net.load_state_dict(state_dict)
    predict(net,features)

if __name__ == "__main__":
    main()







