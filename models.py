import torch.nn as nn
from torch.autograd import Variable
import torch




class GRU4rec(nn.Module):
    def __init__(self, nitem, ninp, nhid, nlayers=1, dropout=0.1):
        super(GRU4rec, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nitem, ninp)
        self.gru = nn.GRU(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, nitem)
        #self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #self.threshold = nn.Threshold(0, neg_inf)

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        
    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.drop(self.encoder(input))
        output,hidden = self.gru(emb, hidden)
        
        #output = self.drop(output) #output size of (num_seq_len * batch * num hidden)
        #output = F.relu(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        
        #decoded = self.threshold(decoded)
        #decoded = F.relu(decoded)
        #nonneg, indices = self.nonnegative(decoded) # select nonnegative values, and do linear transform.
        

        #final activation function 
        #decoded = self.tanh(decoded)
        #decoded = F.relu(decoded)
        
        #decoded = self.sigmoid(decoded)
        
        return decoded, hidden
      
    
    def init_weight(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    
class GRU4recCB(nn.Module):
    def __init__(self, nitem, ninp, nhid, cb_feat_size, nlayers=1, dropout=0.1):
        super(GRU4recCB, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nitem, ninp)
        self.gru = nn.GRU(ninp, nhid, nlayers, dropout=dropout)
        self.gru_cb = nn.GRU(cb_feat_size, nhid, nlayers, dropout=dropout)
        
        self.decoder = nn.Linear(nhid+nhid, nitem)
        #self.decoder = nn.Linear(nhid, nitem)
        #self.decoder_cb = nn.Linear(nhid, nitem)
        #self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
        #self.threshold = nn.Threshold(0, neg_inf)

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        
    def forward(self, input, doc_emb, hidden, hidden_cb):
        
        ###############
        #article ID
        emb = self.encoder(input)
        emb = self.drop(self.encoder(input))
        sub_output,hidden = self.gru(emb, hidden)
        #decoded = self.decoder(sub_output.view(sub_output.size(0)*sub_output.size(1), sub_output.size(2)))
        #decoded = F.relu(decoded)
        
        ###############
        #article's contents
        doc_emb = self.drop(doc_emb)
        sub_output_cb, hidden_cb = self.gru_cb(doc_emb, hidden_cb)
        #decoded_cb = self.decoder_cb(sub_output_cb.view(sub_output_cb.size(0)*sub_output_cb.size(1), sub_output_cb.size(2)))
        #decoded_cb = F.relu(decoded_cb)
        
        output = torch.cat((sub_output[0], sub_output_cb[0]), 1)
        decoded = self.decoder(output)
        
        
        #output = torch.add(decoded, decoded_cb)
        
        
        #decoded = self.threshold(decoded)
        #decoded = F.relu(decoded)
        #nonneg, indices = self.nonnegative(decoded) # select nonnegative values, and do linear transform.
        

        #final activation function 
        #decoded = self.tanh(decoded)
        #decoded = F.relu(decoded)
        #decoded = self.sigmoid(decoded)
        
        return decoded, hidden, hidden_cb
      
    
    def init_weight(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    
           
