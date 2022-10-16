from itertools import product
from tabulate import tabulate
import copy
import numpy as np
        
class Factor:
    '''
    Factors are a generalisation of discrete probability distributions over one or more random variables.
    Each variable must have a name (which may be a string or integer).
    The domain of the factor specifies which variables the factor operates over.
    The outcomeSpace specifies which 
    
    
    The probabilities are stored in a n-dimensional numpy array, using the domain and outcomeSpace
    as dimension and row labels respectively.
    '''
    def __init__(self, domain, outcomeSpace, table=None):
        '''
        Inititalise a factor with a given domain and outcomeSpace. 
        All probabilities are set to zero by default.
        '''
        self.domain = tuple(domain) # tuple of variable names, which may be strings, integers, etc.
        
        if table is None:
            # By default, intitialize with a uniform distribution
            self.table = np.ones(shape=tuple(len(outcomeSpace[var]) for var in self.domain))
            self.table = self.table/np.sum(self.table)
        else:
            self.table = table
            
        self.outcomeSpace = copy.copy(outcomeSpace)
    
    def __getitem__(self, outcomes):
        '''
        This function allows direct access to individual probabilities.
        E.g. if the factor represents a joint distribution over variables 'A','B','C','D', each with outcomeSpace [0,1,2],
        then `factor[0,1,0,2]` will return the probability that the four variables are set to 0,1,0,2 respectively.
        '''
        
        # check if only a single index was used.
        if not isinstance(outcomes, tuple):
            outcomes = (outcomes,)
            
        # convert outcomes into array indicies
        indices = tuple(self.outcomeSpace[var].index(outcomes[i]) for i, var in enumerate(self.domain))
        return self.table[indices]
    
    def __setitem__(self, outcomes, new_value):
        '''
        This function is called when setting a probability. E.g. `factor[0,1,0,2] = 0.5`.        
        '''
        if not isinstance(outcomes, tuple):
            outcomes = (outcomes,)
        indices = tuple(self.outcomeSpace[var].index(outcomes[i]) for i, var in enumerate(self.domain))
        self.table[indices] = new_value
            
    def join(self, other):
        '''
        This function multiplies two factors: one in this object and the factor in `other`
        '''
        # confirm that any shared variables have the same outcomeSpace
        for var in set(other.domain).intersection(set(self.domain)):
            if self.outcomeSpace[var] != other.outcomeSpace[var]:
                print(var)
                print(self.outcomeSpace[var])
                print(other.outcomeSpace[var])
                raise IndexError('Incompatible outcomeSpaces. Make sure you set the same evidence on all factors')

        # extend current domain with any new variables required
        new_dom = list(self.domain) + list(set(other.domain) - set(self.domain)) 

        self_t = self.table
        other_t = other.table

        # to prepare for multiplying arrays, we need to make sure both arrays have the correct number of axes
        # We will do this by adding dimensions of size 1 to the end of the shape of each array.
        num_new_axes = len(set(other.domain) - set(self.domain))
        for i in range(num_new_axes):
            # add an axis to self_t. E.g. if shape is [3,5], new shape will be [3,5,1]
            self_t = np.expand_dims(self_t,-1) 
        num_new_axes = len(set(self.domain) - set(other.domain))
        for i in range(num_new_axes):
            # add an axis to other_t. E.g. if shape is [3,5], new shape will be [3,5,1]
            other_t = np.expand_dims(other_t,-1) 

        # And we need the new axes to be transposed to the correct location
        old_order = list(other.domain) + list(set(self.domain) - set(other.domain)) 
        new_order = []
        for v in new_dom:
            new_order.append(old_order.index(v))
        other_t = np.transpose(other_t, new_order)

        # Now that the arrays are all set up, we can rely on numpy broadcasting to work out which numbers need to be multiplied.
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        new_table = self_t * other_t

        # The final step is to create the new outcomeSpace
        new_outcomeSpace = self.outcomeSpace.copy()
        new_outcomeSpace.update(other.outcomeSpace)

        # in the following line, `self.__class__` is the same as `Factor` (except it doesn't break things when subclassing)
        return self.__class__(tuple(new_dom), new_outcomeSpace, table=new_table)
        
    def evidence(self, **kwargs):
        '''
        Sets evidence by modifying the outcomeSpace
        This function must be used to set evidence on all factors before joining,
        because it removes the relevant variable from the factor. 
        '''
        f = self.copy()
        evidence_dict = kwargs
        for var, value in evidence_dict.items():
            if var in f.domain:
                
                # find the row index that corresponds to var = value
                index = f.outcomeSpace[var].index(value)
                
                # find the `var` axis and select only the row that corresponds to `value`
                # on all other axes, select every row
                slice_tuple = tuple(slice(index,index+1) if v == var else slice(None) for v in f.domain)
                f.table = f.table[slice_tuple]
                
                # modify the outcomeSpace to correspond to the changes just made to self.table
            f.outcomeSpace[var] = (value,)
        return f

    def evidence(self, **kwargs):
        '''
        Sets evidence by removing the observed variables from the factor domain
        This function must be used to set evidence on all factors before joining,
        because it removes the relevant variable from the factor. 
        '''
        f = self.copy()
        evi = kwargs
        indices = tuple(self.outcomeSpace[v].index(evi[v]) if v in evi else slice(None) for v in self.domain)
        f.table = f.table[indices]
        f.domain = tuple(v for v in f.domain if v not in evi)
        return f        
    
    def marginalize(self, var):
        '''
        This function removes a variable from the domain, and sums over that variable in the table
        '''
        
        # create new domain
        new_dom = list(self.domain)
        new_dom.remove(var) 
        
        # remove an axis of the table by summing it out
        axis = self.domain.index(var)
        new_table = np.sum(self.table, axis=axis)
        
        # in the following line, `self.__class__` is the same as `Factor` (except it doesn't break things when subclassing)
        return self.__class__(tuple(new_dom),self.outcomeSpace, new_table)
    
    def copy(self):
        return copy.deepcopy(self)
    
    def normalize(self):
        '''
        Normalise the factor so that all probabilities add up to 1
        '''
        self.table = self.table/np.sum(self.table)
        return self
    
    def __mul__(self, other):
        '''
        Override the * operator, so that it can be used to join factors
        '''
        return self.join(other)
            
    def __str__(self):
        '''
        This function determines the string representation of this object.
        This function will be called whenever you print out this object, i.e., print(a_prob)
        '''
        table = []
        outcomeSpaces = [self.outcomeSpace[var] for var in self.domain]
        for key in product(*outcomeSpaces):
            row = list(key)
            row.append(self[key])
            table.append(row)
        header = list(self.domain) + ['Pr']
        return tabulate(table, headers=header, tablefmt='fancy_grid') + '\n'