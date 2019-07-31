import models
import utils


import argparse
import os
import torch
import numpy as np
import time
import cPickle as pickle
import random

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import rank_metrics
import itertools

from aurochs.buffalo import feature

path = 'data/n_clu6tail_cnt22/'
bsz = 1024 # batch size
eval_bsz = 1024
cuda = True # Enable GPU or not

ninp=500 # size of embedding
nhid=100 # size of hidden layers
nlayers=1 # number of GRU
log_interval = 1000
epochs = 1

cb_feat_size = 120
topk=20

#k=6 # num. of clusters
#tail_cnt = np.percentile(dict_item.cnt.values(), 90) # 22



######################################################
#Which model to train?
######################################################

GRU4REC = 0
GRU4REC_RERANKING = 1
GRU4REC_CB = 2
GRU4DIV = 3
GRU4DIV_CB = 4

parser = argparse.ArgumentParser(description = 'Select 0:gru4rec, 1:gru4rec+reranking, 2:gru4recCB, 3:gru4div, 4:gru4divCB')
parser.add_argument('model', metavar = 'model', type=int)

args = parser.parse_args()
which_model = args.model


######################################################
#Load data
######################################################

#Each preprocess data is type of a list of items. Item is linked to next item.
class Dictionary(object):
    def __init__(self):
        self.item2idx = {}
        self.idx2item = []

    def add_item(self, item):
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item)-1
        return self.item2idx[item]

class Item:
    def __init__(self,initdata):
        self.data = initdata #item id
        self.next = None
        self.uid = uid #user id
        self.tail = None

#preprocessed data
train = pickle.load(open(path+'train.p', 'rb'))
test = pickle.load(open(path+'test.p', 'rb'))
valid = pickle.load(open(path+'valid.p', 'rb'))
dict_item = pickle.load(open(path+'dict_item.p', 'rb'))
#dict_user = pickle.load(open(path+'dict_user.p', 'rb'))
nitem = len(dict_item.idx2item)


#clustering results
clusterer = pickle.load(open(path+'clusterer.p', 'rb')) #type(clusterer) : KMeans
cid_items = pickle.load(open(path+'cid_items.p', 'rb')) #dict {cid, list of items}
seen = pickle.load(open(path+'seen.p', 'rb')) #dict {itemid, probability of being seen}


#######################################################
#Contents vector
######################################################
ROOT = 'data/2017092711-cb-text/'
cb_items = feature.load(os.path.join(ROOT, 'main'))

cb_feat = []
for item in dict_item.idx2item:
    item = item.replace("_", "|")

    if 'cluster' in item:
        cid = int(item[7:])
        feat = clusterer.cluster_centers_[cid]
    else:
        feat = cb_items.feats[cb_items.idmap[item]]

    cb_feat.append(feat)

cb_feat = np.array(cb_feat)


#######################################################
#Build model
######################################################
name = 'model_'
if (which_model == GRU4REC):
    model = models.GRU4rec(nitem, ninp, nhid)
    criterion = utils.ListMLE_loss()
    name +='GRU4rec'

elif (which_model == GRU4REC_RERANKING):
    name += 'GRU4rec_reranking'

elif (which_model == GRU4REC_CB):
    model = models.GRU4recCB(nitem, ninp, nhid, cb_feat_size)
    criterion = utils.ListMLE_loss()
    name += 'GRU4rec_cb'

elif (which_model == GRU4DIV):
    model = models.GRU4rec(nitem, ninp, nhid)
    criterion = utils.ListMLE_loss_tail(cuda)
    name += 'GRU4div'

elif (which_model == GRU4DIV_CB):
    model = models.GRU4recCB(nitem, ninp, nhid, cb_feat_size)
    criterion = utils.ListMLE_loss_tail(cuda)
    name +='GRU4div_cb'
else:
    print 'Error!'


if cuda:
    model = model.cuda()


#######################################################
#Train
######################################################
def training():
    model.train() # turn on training mode which enables dropout
    # iterate over batch
    total_ranking_loss = 0

    start_time = time.time()
    #Initial hidden state is input for our model
    hidden = model.init_hidden(bsz)

    batch = utils.get_batch(train, bsz, cuda)
    data = []
    iteration = 0

    optimizer = optim.Adagrad(model.parameters())

    while True:
        data, target, tails, init= next(batch, None) #gives None if the generator expires instead of StopIteration error

        if data is None:
            break

        # Starting each  batch, we detach the hidden state from how it was previous produced.
        # If we didnt' the model would try backpropagating all the way to start of the dataset
        hidden = utils.repackage_hidden(hidden)

        #reset hidden state for indepedent new session
        for i in init:
            hidden.data[0][i,:].zero_()

        optimizer.zero_grad()

        output, hidden = model(data,hidden)

        loss, temp = criterion(output.view(-1, nitem), target, tails)

        loss = torch.mean(loss)

        loss.backward()


        #'clip_grad_norm' helps prevent the exploding gradient problem in RNN
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

        #SGD
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)

        #Adagrad
        optimizer.step()

        total_ranking_loss += loss.data

        iteration += 1

        if iteration % log_interval == 0 and iteration > 0:
            cur_ranking_loss = total_ranking_loss[0] / log_interval
            elapsed = time.time()-start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss_ranking {:8.5f} '.format(epoch, iteration, len(train)//bsz,  elapsed * 1000/log_interval, cur_ranking_loss))

            total_ranking_loss = 0
            start_time=time.time()


def trainingCB(fixed_params):
    model.train() # turn on training mode which enables dropout
    # iterate over batch
    total_ranking_loss = 0

    start_time = time.time()
    #Initial hidden state is input for our model
    hidden = model.init_hidden(bsz)
    hidden_cb = model.init_hidden(bsz)

    batch = utils.get_batch(train, bsz, cuda)
    data = []
    iteration = 0

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adagrad(model.parameters())

    i=0
    for param in model.parameters():
        if i in fixed_params:
            param.requires_grad = False
        else:
            param.requires_grad = True
        i+=1


    model.gru.flatten_parameters()
    model.gru_cb.flatten_parameters()

    while True:
        data, target, tails, init= next(batch, None) #gives None if the generator expires instead of StopIteration error

        if data is None:
            break

        # Starting each  batch, we detach the hidden state from how it was previous produced.
        # If we didnt' the model would try backpropagating all the way to start of the dataset
        hidden = utils.repackage_hidden(hidden)
        hidden_cb = utils.repackage_hidden(hidden_cb)
        doc_emb = utils.get_doc_emb(data, cb_feat, cb_feat_size, cuda)

        #reset hidden state for indepedent new session
        for i in init:
            hidden.data[0][i,:].zero_()
            hidden_cb.data[0][i,:].zero_()

        optimizer.zero_grad()

        output, hidden, hidden_cb = model(data, doc_emb, hidden, hidden_cb)

        loss, temp = criterion(output.view(-1, nitem), target, tails)

        loss = torch.mean(loss)

        loss.backward()


        #'clip_grad_norm' helps prevent the exploding gradient problem in RNN
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

        #SGD
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)

        #Adagrad
        optimizer.step()

        total_ranking_loss += loss.data

        iteration += 1

        if iteration % log_interval == 0 and iteration > 0:
            cur_ranking_loss = total_ranking_loss[0] / log_interval
            elapsed = time.time()-start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss_ranking {:8.5f} '.format(epoch, iteration, len(train)//bsz,  elapsed * 1000/log_interval, cur_ranking_loss))

            total_ranking_loss = 0
            start_time=time.time()



#######################################################
# Evaluate
#######################################################
def evaluate(source):
    model.eval()

    hidden = model.init_hidden(eval_bsz)
    data = []
    batch = utils.get_batch(source, eval_bsz, cuda)
    iteration = 0
    total_ranking_loss =0

    #accuracy (head)
    total_acc_h = np.zeros(4) #mrr, map, ndcg, precision

    #accuracy (head + tail)
    total_acc_ht = np.zeros(4)

    #novelty
    total_p_unseen = 0

    #aggregate diversity
    total_rec_set = set()

    #individual diversity
    total_distance = 0
    model.gru.flatten_parameters()

    while True:
        data, target, tails, init = next(batch, None)
        if data is None:
            break

        iteration +=1

        #reset hidden state for indepedent new session
        for i in init:
            hidden.data[0][i,:].zero_()

        data = Variable(data.data, volatile=True)
        output, hidden = model(data,hidden)

        #Loss
        loss, temp = criterion(output.view(-1, nitem), target, tails)
        loss = torch.mean(loss)

        total_ranking_loss +=loss.data

        #topk
        _, indices = torch.topk(output.view(-1, nitem).data, topk)
        idx = indices.cpu().numpy()
        target = target.data.cpu().numpy()

        #Accuracy with only head
        acc_h = utils.get_accuracy(idx, target, topk)
        total_acc_h += acc_h
        #Accuracy with head and tail
        acc_ht = utils.get_accuracy_tail(idx, target, tails, topk)
        total_acc_ht += acc_ht


        #Long-tail measures
        p_unseen, rec_set, distance = utils.get_long_tail_measures(idx, dict_item, eval_bsz, cb_items, cid_items, seen, topk )
        total_p_unseen += p_unseen
        total_rec_set = total_rec_set.union(rec_set)
        total_distance += distance


    return total_ranking_loss[0]/iteration, total_acc_h/iteration, total_acc_ht/iteration, total_p_unseen/iteration, total_rec_set, total_distance/iteration




def evaluateCB(source):
    model.eval()

    hidden = model.init_hidden(eval_bsz)
    hidden_cb = model.init_hidden(eval_bsz)
    data = []
    batch = utils.get_batch(source, eval_bsz, cuda)
    iteration = 0
    total_ranking_loss =0

    #accuracy (head)
    total_acc_h = np.zeros(4) #mrr, map, ndcg, precision

    #accuracy (head + tail)
    total_acc_ht = np.zeros(4)

    #novelty
    total_p_unseen = 0
    #aggregate diversity
    total_rec_set = set()
    #individual diversity
    total_distance = 0
    model.gru.flatten_parameters()
    model.gru_cb.flatten_parameters()


    while True:
        data, target, tails, init = next(batch, None)
        if data is None:
            break

        iteration +=1

        #reset hidden state for indepedent new session
        for i in init:
            hidden.data[0][i,:].zero_()
            hidden_cb.data[0][i,:].zero_()

        doc_emb = utils.get_doc_emb(data, cb_feat, cb_feat_size, cuda)
        output, hidden, hidden_cb = model(data, doc_emb, hidden, hidden_cb)

        #Loss
        loss, temp = criterion(output.view(-1, nitem), target, tails)
        loss = torch.mean(loss)

        total_ranking_loss +=loss.data


        #topk
        _, indices = torch.topk(output.view(-1, nitem).data, topk)
        idx = indices.cpu().numpy()
        target = target.data.cpu().numpy()

        #Accuracy
        #mrr, mAP, ndcg, precision = get_accuracy(output, target)
        acc_h = utils.get_accuracy(idx, target, topk)
        total_acc_h += acc_h
        #Accuracy with tail
        acc_ht = utils.get_accuracy_tail(idx, target, tails, topk)
        total_acc_ht += acc_ht


        #Long-tail measures
        p_unseen, rec_set, distance = utils.get_long_tail_measures(idx, dict_item, eval_bsz, cb_items, cid_items, seen, topk )
        total_p_unseen += p_unseen
        total_rec_set = total_rec_set.union(rec_set)
        total_distance += distance

    return total_ranking_loss[0]/iteration, total_acc_h/iteration, total_acc_ht/iteration, total_p_unseen/iteration, total_rec_set, total_distance/iteration




########################################################
#Is there any good way of automaticallyd detecting the structure and fixing parameters?
#num_param= 0
#for param in model.parameters():
#    print(num_param, param.size())
#    num_param += 1
subnet_param = [5,6,7,8] #those parameter tensor will be fixed during optimization
subnet_param_cb = [0,1,2,3,4]


#######################################################
#Run train
######################################################
best_val_loss =None
#logs = []

#Loop over epochs
#At any point you can hit Ctrl + C to break out of training early
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()

        if (which_model == GRU4REC_CB) or (which_model == GRU4DIV_CB):
            #alternating optimization

            trainingCB(subnet_param)
            trainingCB(subnet_param_cb)
            val_ranking_loss, acc_h, acc_ht, unseen, rec_set, distance= evaluateCB(valid)

        else:
            training()
            val_ranking_loss, acc_h, acc_ht, unseen, rec_set, distance= evaluate(valid)



        print('-'*89)
        print('| end of epoch {:3d} | time: {:8.5f}s | valid ranking loss {:8.5f} '.format(epoch, (time.time() - epoch_start_time), val_ranking_loss))
        print acc_h, acc_ht, unseen, len(rec_set), distance #acc = [mrr, map, ndcg, precision]
        print('-'*89)

        #logs.append([val_ranking_loss, acc_h, acc_ht, unseen, len(rec_set), distance])

        #save the model if the validation loss is the best we've seen so far
        #if not best_val_loss or val_ranking_loss < best_val_loss:
        with open(path+name+'.p', 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_ranking_loss

except KeyboardInterrupt:
    print('-'*89)
    print('Exiting from training early')



######################################################
#Load the best saved model
######################################################
with open(path+name+'.p', 'rb') as f:
    model = torch.load(f)



#####################################################
#Run on test data
######################################################

#test1 = test[:len(test)/2]
#test2 = test[len(test)/2:]


if (which_model == GRU4REC_CB) or (which_model == GRU4DIV_CB):
    test_ranking_loss, acc_h, acc_ht, unseen, rec_set, distance= evaluateCB(test)

else:
    test_ranking_loss, acc_h, acc_ht, unseen, rec_set, distance= evaluate(test)

print('='*89)
print('| end of epoch {:3d} | time: {:8.5f}s | valid ranking loss {:8.5f} '.format(epoch, (time.time() - epoch_start_time), test_ranking_loss))
print acc_h, acc_ht, unseen, len(rec_set), distance #acc = [mrr, map, ndcg, precision]
print('='*89)
