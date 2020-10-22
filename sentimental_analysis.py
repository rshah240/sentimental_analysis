#Importing Libraries
import torch
import numpy as np
import torch.nn as nn
import hyperparameters
from string import punctuation
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from torch import optim

device = ("cuda" if torch.cuda.is_available() else "cpu")


def get_data():
    """ Function returning data after reading it form file"""
    with open('data/reviews.txt','r') as f:
        reviews = f.read()
    with open('data/labels.txt','r') as f:
        labels = f.read()
    return reviews,labels

def preprocessing_data(reviews,labels):
    """Preprocessing data"""
    #getting rid of punctuation
    reviews = reviews.lower()
    all_text =''.join([c for c in reviews if c not in punctuation])

    #split by new lines and spaces
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)
    
    #create a list of words
    words = all_text.split()
    #Getting the count of each word occured in the review
    counts = Counter(words)
    vocab = sorted(counts,key=counts.get,reverse = True)
    vocab_to_int = {word:ii for ii,word in enumerate(vocab,1)}
    ## using the dict to tokenize each review in review_split
    ## store the tokenized reviews in review_ints

    reviews_int = []
    for review in reviews_split:
        reviews_int.append([vocab_to_int[word] for word in review.split()])

    #encoding labels
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

    return reviews_int,encoded_labels,len(vocab_to_int)
def remove_outliers(reviews_int,encoded_labels):
    """Function removing reviews having zero length"""
    # indexes of the reviews having non zero length
    non_zero_idx = [ii for ii,review in enumerate(reviews_int) if len(review) != 0]
    reviews_int =  [reviews_int[ii] for ii in non_zero_idx]
    encoded_labels = [encoded_labels[ii] for ii in non_zero_idx]
    return reviews_int,encoded_labels


def pad_features(reviews_int):
    """Function padding and truncating reviews to make it of seq_length"""
    seq_length = hyperparameters.seq_length
    features = np.zeros((len(reviews_int),seq_length),dtype = int)

    #for each review, I grab that review
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
        self.sig = nn.Sigmoid()
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
        x = self.sig(x)
        x = x.view(batch_size,-1)
        x = x[:,-1] #using the output of the last cell


        return x,hidden
    def init_hidden(self,batch_size):
        weights = next(self.parameters()).data

        hidden = (weights.new(self.n_layers,batch_size,self.hidden_dim).to(device),
                  weights.new(self.n_layers,batch_size,self.hidden_dim).to(device))
        return hidden

def train(net,train_loader,val_loader):
    """Training the model"""
    net.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(),lr = hyperparameters.lr)
    print_every = 100
    clip = 5
    batch_size = hyperparameters.batch_size
    net.train()
    epochs = hyperparameters.epochs
    for e in range(epochs):
        counter  = 0
        h = net.init_hidden(batch_size)
        for inputs,labels in train_loader:
            #zeroing the gradients
            optimizer.zero_grad()
            inputs,labels = inputs.to(device),labels.to(device)
            h = tuple([each.data for each in h])
            output,h = net(inputs,h)
            loss = criterion(output.squeeze(),labels.float())
            #detatching from the graph
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),clip)
            optimizer.step()

            counter += 1
            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs,labels in val_loader:
                    inputs,labels = inputs.to(device),labels.to(device)
                    val_h = tuple([each.data for each in val_h])
                    output,val_h = net(inputs,val_h)
                    val_loss = criterion(output.squeeze(),labels.float())
                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}".format(e+1,epochs),
                      "Step: {}".format(counter),
                      "Loss: {:.6f}".format(loss.item()),
                      "Val Loss:{:.6f}".format(np.mean(val_losses)))

        print("Saving Model")
        checkpoint = {'input_size':net.input_size,
                      'hidden_dim':net.hidden_dim,
                      'embedding_dim':net.embedding_dim,
                      'state_dict':net.state_dict(),
                      'n_layers':net.n_layers,
                      'output_size':net.output_size}
        with open('Sentimental_Analysis.pt','wb') as f:
            torch.save(checkpoint,f)

def test_model(net,test_loader):
    """Testing the accuracy of the model"""
    net.to(device)
    test_h = net.init_hidden(hyperparameters.batch_size)
    criterion = nn.BCELoss()
    test_losses = []
    num_correct = 0
    for inputs,labels in test_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        test_h = tuple([each.data for each in test_h])
        output , test_h = net(inputs,test_h)
        test_loss = criterion(output,labels.float())
        test_losses.append(test_loss.item())
        # convert output probabilities to predicted class (0 or 1)
        prediction = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = prediction.eq(labels.float().view_as(prediction))
        correct = np.squeeze(correct_tensor.numpy()) if device == 'cpu' else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print('Stats')
    print('Test Loss :{:.6f}'.format(np.mean(test_losses)))
    print('Test Accuracy:{:.6f}'.format(num_correct / len(test_loader.dataset)))








def main():
    reviews,labels = get_data()
    reviews_int,encoded_labels,input_size = preprocessing_data(reviews,labels)
    reviews_int,encoded_labels = remove_outliers(reviews_int,encoded_labels)
    features = pad_features(reviews_int)
    train_X, remaining_X, train_labels, remaining_labels = train_test_split(features,encoded_labels,
                                                                            train_size = hyperparameters.split_frac,
                                                                            random_state = 42)
    Val_X,test_X,Val_labels, test_labels = train_test_split(remaining_X,remaining_labels,
                                                              train_size=0.5,random_state = 42)

    #Creating dataloaders
    train_labels,Val_labels,test_labels = np.array([train_labels]), np.array([Val_labels]), np.array([test_labels])
    train_labels = train_labels.reshape(train_labels.shape[1])
    Val_labels = Val_labels.reshape(Val_labels.shape[1])
    test_labels = test_labels.reshape(test_labels.shape[1])
    train_data = TensorDataset(torch.from_numpy(train_X),torch.from_numpy(train_labels))
    val_data = TensorDataset(torch.from_numpy(Val_X),torch.from_numpy(Val_labels))
    test_data = TensorDataset(torch.from_numpy(test_X),torch.from_numpy(test_labels))

    train_loader = DataLoader(train_data,batch_size=hyperparameters.batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=hyperparameters.batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=hyperparameters.batch_size,shuffle=True)

    output_size = 1
    input_size = input_size + 1 ##padding sequence addtion
    #Instantiating the model
    net = Sentimental_Analysis(input_size=input_size,hidden_dim=hyperparameters.hidden_dim,
                               embedding_dim=hyperparameters.embedding_dim,
                               output_size=output_size,n_layers = hyperparameters.n_layers)
    train(net,train_loader,val_loader)
    test_model(net,test_loader)


if __name__ =="__main__":
    main()















    

