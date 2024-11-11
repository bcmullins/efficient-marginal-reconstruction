from rem.dataset import Dataset
from rem.domain import Domain
from itertools import combinations, chain
import numpy as np
import pandas as pd
import json
import torch
import scipy

def importData(dataset):
    if dataset == 'synthetic':
        num_attributes = 3
        att_range = 3
        num_records = 200
        domain_synth = Domain([str(i) for i in range(num_attributes)], [att_range]*num_attributes)
        return(Dataset.synthetic(domain_synth, num_records))
    else:
        with open("datasets/" + dataset + "-domain.json") as f:
                domain = json.load(f)
        data_raw = pd.read_csv("datasets/" + dataset + ".csv")
        col_map = {col: str(i) for i, col in enumerate(data_raw.columns)}
        domain = {col_map[col] : domain[col] for col in data_raw.columns}
        data_raw.columns = [col_map[col] for col in data_raw.columns]
        return(Dataset(df = data_raw, domain = Domain.fromdict(domain)))

def calcErrors(true_marginals, inf_marginals, metric = 'l1'):
    if metric == 'l1':
        return np.mean([torch.linalg.vector_norm((inf_marginals[idx] - true_marginal), 1).item() / true_marginal.shape[0] 
                        for idx, true_marginal in enumerate(true_marginals)])
    elif metric == 'l2':
        return np.mean([torch.linalg.vector_norm((inf_marginals[idx] - true_marginal), 2).item() / true_marginal.shape[0] 
                        for idx, true_marginal in enumerate(true_marginals)])
    
def exponential(candidates, scores, sensitivity, epsilon):
    probabilities = scipy.special.softmax((0.5*epsilon/sensitivity)*scores)
    index = np.random.choice(range(len(candidates)), 1, p=probabilities)[0]
    return candidates[index]

def powerset(iterable_set):
    """
    Take an iterable that corresponds to a set and return the powerset of that set as an iterator
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    
    Arguments:
    iterable_set: the iterable that corresponds to the set (iterable)

    Returns:
    an iterator over the powerset of the set (iterator)
    """
    s = list(iterable_set)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

def downward_closure(W):
    """
    Take a workload and return the downward closure of the workload (the union of the power sets of the marginals)

    Arguments:
    W: the workload (list of marginals (also lists))

    Returns:
    the downward closure of the workload (list of marginals (also lists))
    """
    ans = set([tuple()])
    for marginal in W:
        ans.update(powerset(marginal))
    return list(sorted(ans, key=len))

def all_k_way(domain, k):
    return list(combinations(domain, k))

def attrMulti(candidate, domain):
    return np.prod([(domain[col] - 1)/domain[col] for col in candidate])

def attrQuot(candidate, domain):
    return np.prod([domain[col] ** -2 for col in candidate])

def domainSize(candidate, domain):
    return np.prod([domain[col] for col in candidate])

def attrSubMQ(candidate, sub, domain):
    return np.prod([attrMulti((col), domain) if col in sub else attrQuot((col), domain) for col in candidate])

def varSum(tau, workloads, domain):
    return np.sum([domainSize(wkload, domain) * attrSubMQ(wkload, tau, domain) 
                   for wkload in workloads if set(tau).issubset(wkload)])

def sigma(candidate, domain, rho):
    return ((1/(2*rho)) * attrMulti(candidate, domain)) ** 0.5

def getOptimalSigmasCF(marginals, rho, domain):
    c = 2 * rho
    
    dc = downward_closure(marginals)
    # calc p
    p = {wkload : attrMulti(wkload, domain) for wkload in dc}
    # calc v
    v = {tau : varSum(tau, marginals, domain) for tau in dc}
    # calc T
    T = (np.sum([(v[wkload] * p[wkload]) ** 0.5  for wkload in dc]) ** 2) / c
    # calc opt sigmas 
    optSigmas = [((T * p[wkload])/(c * v[wkload])) ** (0.5/2) for wkload in dc]
    
    return (dc, optSigmas)