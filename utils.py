import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import random

import sklearn.metrics.pairwise as pairwise
import numpy as np


#####################################
#Criterion for learning network
#####################################

class ListMLE_loss_tail(nn.Module):
    """Listwise ranking model. ListMLE.
    http://icml2008.cs.helsinki.fi/papers/167.pdf"""
    def __init__(self, cuda):
        super(ListMLE_loss_tail, self).__init__()
        self.cuda = cuda

    def forward(self, output, target, tails):
        """
        output: Variable type. bsz*nitem. Contain rank scores
        target: Variable type. bsz*1. Contain target item ID
        tails: List. Inconsistent length.  Contain auxillary item IDs if any
        """
        target_scores = torch.squeeze(torch.gather(output, 1, target.view(-1,1)))
        below = -torch.log(torch.sum(torch.exp(output),1))
        log_pl = target_scores + below

        temp = []

        for i in range(len(tails)):
            if len(tails[i]) > 0 : # if any tail items exist
                tail_idx = Variable(torch.LongTensor(tails[i]))
                if self.cuda: tail_idx = tail_idx.cuda()
                scores = torch.index_select(output[i,:], 0, tail_idx)
                others_scores_tail = torch.sum(torch.exp(output[i,:]))-(torch.exp(target_scores[i])+torch.sum(torch.exp(scores)))
                log_pl_tail = get_PL_dist(scores, others_scores_tail)

                log_pl[i] = log_pl[i].clone() + log_pl_tail

                temp.append((i, log_pl_tail))



        neg_like = -log_pl

        return neg_like, temp


class ListMLE_loss(nn.Module):
    """Listwise ranking model. ListMLE.
    http://icml2008.cs.helsinki.fi/papers/167.pdf"""
    def __init__(self):
        super(ListMLE_loss, self).__init__()

    def forward(self, output, target, tails):
        """
        output: Variable type. bsz*nitem. Contain rank scores
        target: Variable type. bsz*1. Contain target item ID
        tails: List. Inconsistent length.  Contain auxillary item IDs if any
        """
        target_scores = torch.squeeze(torch.gather(output, 1, target.view(-1,1)))
        below = -torch.log(torch.sum(torch.exp(output),1))
        log_pl = target_scores + below

        temp = []

        neg_like = -log_pl

        return neg_like, temp


def get_PL_dist(scores, others_scores_tail):

    above = torch.sum(scores)

    #flip
    below = flip(scores) # for cumsum
    below = torch.exp(below)
    below = torch.cumsum(below, 0)
    below = torch.log(below+ others_scores_tail)
    below_sum = torch.sum(below)

    log_pl = above - below_sum

    return log_pl


def flip(scores):
    scores_ncol = scores.size()[0]

    idx = [i for i in range(scores_ncol-1,-1,-1)]
    idx = Variable(torch.LongTensor(idx))
    if scores.is_cuda: idx = idx.cuda()

    scores = scores.index_select(0, idx)
    return scores





#####################################
#Accuracy measures
#####################################
def get_accuracy(idx, target, topk):

    rs = [] #bsz * topk

    for i, t in zip(idx, target):
        #relevance score for target
        #for target
        r = 1.0*np.in1d(i, t)
        rs.append(r) # true if item in i contain item in t


    #mrr = mean_reciprocal_rank(rs)
    mAP = mean_average_precision(rs) # relevant = target and tail. irrelevant o/w
    ndcg = np.mean([ndcg_at_k(r, topk, method=1) for r in rs])
    #precision = np.mean([precision_at_k(r, topk) for r in rs])

    #return np.array([mrr, mAP, ndcg, precision])
    return np.array([mAP, ndcg])



def get_accuracy_tail(idx, target, tails, topk):

    rs = [] #bsz * topk

    for i, t, tail in zip(idx, target, tails):
        #relevance score for target and tails

        #target
        r = 1.0*np.in1d(i, t)
        #tails
        for cid in tail: # for each long tail cid
            r += 0.8*(i==cid)
        rs.append(r) # true if item in i contain item in t


    #mrr = mean_reciprocal_rank(rs)
    mAP = mean_average_precision(rs) # relevant = target and tail. irrelevant o/w
    ndcg = np.mean([ndcg_at_k(r, topk, method=1) for r in rs])
    #precision = np.mean([precision_at_k(r, topk) for r in rs])

    #return np.array([mrr, mAP, ndcg, precision])
    return np.array([mAP, ndcg])


def get_long_tail_measures(idx, dict_item, eval_bsz, cb_items, cid_items, seen, topk):

    # compute averaged popularity of each ranking list using seen
    min_cluster_idx = dict_item.item2idx['cluster0']

    #randomly select tail item in each cluster id
    items_list = []
    for b in range(eval_bsz):
        items = []
        for i in idx[b,:]:
            if i >= min_cluster_idx: #cluster
                items.append(random.choice(cid_items[i-min_cluster_idx]))
            else:
                items.append(dict_item.idx2item[i]) #head item

        items_list.append(items)


    #popularity
    p_unseen = sum([1-seen[item] for items in items_list for item in items])/float(eval_bsz*topk)


    #semantic distance
    #sum_sem_dis = 0
    #for b in range(eval_bsz):
    #    sum_sem_dis += get_sem_dis(items_list[b], cb_items)

    #avg_sem_dis = sum_sem_dis/float(eval_bsz)


    #diversity: num of unique items in ranking
    rec_set = set([i for l in items_list for i in l])


    return p_unseen, rec_set #, avg_sem_dis




def get_long_tail_measures2(idx, dict_item, eval_bsz, cb_items,  seen, topk):

    # compute averaged popularity of each ranking list using seen
    #min_cluster_idx = dict_item.item2idx['cluster0']

    #randomly select tail item in each cluster id
    items_list = []
    for b in range(eval_bsz):
        items = []
        for i in idx[b,:]:
            items.append(dict_item.idx2item[i]) #head item

        items_list.append(items)


    #popularity
    p_unseen = sum([1-seen[item] for items in items_list for item in items])/float(eval_bsz*topk)


    #semantic distance
    #sum_sem_dis = 0
    #for b in range(eval_bsz):
    #    sum_sem_dis += get_sem_dis(items_list[b], cb_items)

    #avg_sem_dis = sum_sem_dis/float(eval_bsz)


    #diversity: num of unique items in ranking
    rec_set = set([i for l in items_list for i in l])


    return p_unseen, rec_set #, avg_sem_dis


def get_sem_dis(items, cb_items):
    #get averaged similarity using cb_feat

    cb_feats = [cb_items.feats[cb_items.idmap[item.replace("_", "|")]] for item in items]


    sem_dis = 1-pairwise.cosine_similarity(cb_feats)
    num_rank_idx = float(len(items))
    avg_sem_dis = (np.sum(np.tril(sem_dis)))/(num_rank_idx*(num_rank_idx-1)/2)

    return avg_sem_dis


#####################################
#rank metrics
#####################################
"""
from https://gist.github.com/bwhite/3726239
Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)
"""


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])




def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Utility is defined as 2^r_k-1
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            #return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return np.sum(np.subtract(np.power(2,r), 1) / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max




#####################################
#others
#####################################

def get_batch(source, bsz, cuda):
    # source: A list of linked lists containing sessions
    left_sess_id = bsz
    data = source[:bsz]
    nones = []
    while True:
        target = [item.next for item in data]
        if None in target:
            nones = [i for i,j in enumerate(target) if j is None]
            for i in nones:
                if left_sess_id < len(source):
                    data[i]= source[left_sess_id]
                    target[i] = data[i].next
                    left_sess_id += 1
                else:
                    yield None, None, None, None
                    return

        data_list =[item.data for item in data]
        data_tensor = torch.LongTensor([data_list]) # due to GRU input style..
        #data_tensor = torch.LongTensor(data_list)

        target_list = [item.data for item in target]
        target_tensor = torch.LongTensor(target_list)

        target_tail_list = [item.tail for item in target]


        if cuda:
            data_tensor = data_tensor.cuda()
            target_tensor = target_tensor.cuda()

        yield Variable(data_tensor), Variable(target_tensor), target_tail_list, nones

        data = target


def repackage_hidden(h):
    """Wraps hidden states in new Variable, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)




def get_doc_emb(input, cb_feat, cb_feat_size, cuda):
    #input: Variable type. bsz*1. contain target ids
    doc_emb = torch.FloatTensor(1, len(input[0]), cb_feat_size).zero_() # bsz * cb_feature size
    doc_emb = Variable(doc_emb)
    for i, idx in enumerate(input.data[0]):
        doc_emb[0, i, :] = torch.FloatTensor(cb_feat[idx])

    if cuda:
        doc_emb = doc_emb.cuda()

    return doc_emb

