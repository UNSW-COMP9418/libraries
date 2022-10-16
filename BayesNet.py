# Necessary libraries
import numpy as np
import pandas as pd

# combinatorics
from itertools import product

from DiscreteFactors import Factor
from Graph import Graph

def allEqualThisIndex(dict_of_arrays, **fixed_vars):
    # base index is a boolean vector, everywhere true
    first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
    index = np.ones_like(first_array, dtype=np.bool_)
    for var_name, var_val in fixed_vars.items():
        index = index & (np.asarray(dict_of_arrays[var_name])==var_val)
    return index

def estimateFactor(data, var_name, parent_names, outcomeSpace):
    var_outcomes = outcomeSpace[var_name]
    parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
    # cartesian product to generate a table of all possible outcomes
    all_parent_combinations = product(*parent_outcomes)

    f = Factor(list(parent_names)+[var_name], outcomeSpace)
    
    for i, parent_combination in enumerate(all_parent_combinations):
        parent_vars = dict(zip(parent_names, parent_combination))
        parent_index = allEqualThisIndex(data, **parent_vars)
        for var_outcome in var_outcomes:
            var_index = (np.asarray(data[var_name])==var_outcome)
            f[tuple(list(parent_combination)+[var_outcome])] = (var_index & parent_index).sum()/parent_index.sum()
            
    return f

class BayesNet():
    def __init__(self, graph, outcomeSpace=None, factor_dict=None):
        self.graph = graph
        self.outcomeSpace = dict()
        self.factors = dict()
        if outcomeSpace is not None:
            self.outcomeSpace = outcomeSpace
        if factor_dict is not None:
            self.factors = factor_dict
            
    def learnParameters(self, data):
        '''
        Iterate over each node in the graph, and use the given data
        to estimate the factor P(node|parents), then add the new factor 
        to the `self.factors` dictionary.
        '''
        graphT = self.graph.transpose()
        for node, parents in graphT.adj_list.items():
            f = estimateFactor(data, node, parents, self.outcomeSpace)
            self.factors[node] = f
            
    def joint(self):
        '''
        Join every factor in the network, and return the resulting factor.
        '''
        factor_list = list(self.factors.values())
        
        accumulator = factor_list[0]
        for factor in factor_list[1:]:
            accumulator = accumulator.join(factor)
        return accumulator
    
    def query(self, q_vars, **q_evi):
        """
        arguments 
        `q_vars`, list of variables in query head
        `q_evi`, dictionary of evidence in the form of variables names and values

        Returns a new NORMALIZED factor will all hidden variables eliminated as evidence set as in q_evi
        """     
        assert isinstance(q_vars,list) or sinstance(q_vars,tuple), "q_vars should be a list"
        
        f = self.joint()
        
        # First, we set the evidence 
        f = f.evidence(**q_evi)

        # Second, we eliminate hidden variables NOT in the query
        for var in self.outcomeSpace:
            if var not in q_vars and var not in list(q_evi):
                f = f.marginalize(var)
        return f.normalize()
