# Necessary libraries
import graphviz
import copy

from DiscreteFactors import Factor
from Graph import Graph

class EliminationTree():
    def __init__(self, graph, separators, outcomeSpace, factor_dict=None):
        self.graph = graph
        self.separators = separators # dict mapping each edge (node1, node2) -> tuple of separator vars
        self.factors = copy.deepcopy(factor_dict)
        self.outcomeSpace = outcomeSpace

        self.messages = None

    def getMessages(self):
        '''
        Propagate messages around the tree
        '''
        root = list(self.graph)[0] # it doesn't matter which node we choose as the root node

        # Initialize dictionary to store messages. The message key will be concatenation of nodes names
        self.messages = dict()
        # For each neighbouring node of root, we start a depth-first search
        for v in self.graph.children(root):
            # Call pull and store the resulting message in messages[v+root]
            self.messages[v+root] = self.pull(v, root)

        # Call push to recursively push messages out from the root. To start it off, set the previous node to '' or None.                 
        self.push(root, '')
        
        return self.messages

    def pull(self, curr, previous):
        """
        argument 
        `curr`, current node.
        `previous`, node we came from in the search.
        
        Returns a factor fx with a message from node previous to root
        """
        # fx is an auxiliary factor. Initialize fx with the factor associated with the curr node
        fx = self.factors[curr].copy()
        # Depth-first search
        for v in self.graph.children(curr):
            # This is an important step: avoid returning using the edge we came from
            if not v == previous:
                # Call pull recursively since root is not an edge with a single neighbour
                self.messages[v+curr] = self.pull(v, curr)
                # Here, we returned from the recursive call. 
                # We need to join the received message with fx
                fx = fx*self.messages[v+curr]
        # fx has all incoming messages multiplied by the node factor. It is time to marginalize the variables not is S_{ij}
        for v in fx.domain:
            if not v in self.separators[''.join(sorted([previous,curr]))]:
                # Call marginalize to remove variable v from fx's domain
                fx = fx.marginalize(v)
        return fx

    def push(self, curr, previous):
        """
        argument 
        `curr`, current node.
        `previous`, previous node.
        """    
        for v in self.graph.children(curr):
            # This is an important step: avoid returning using the edge we came from        
            if not v == previous:
                # Initialize messages[curr+v] with the factor associated with the curr node
                self.messages[curr+v] = self.factors[curr].copy()
                for w in self.graph.children(curr):
                    # This is an important step: do not consider the incoming message from v when computing the outgoing message to v
                    if not v == w:
                        # Join messages coming from w into messages[curr+v]
                        self.messages[curr+v] = self.messages[curr+v]*self.messages[w+curr]

                # messages[curr+v] has all incoming messages multiplied by the node factor. It is time to marginalize the variables not is S_{ij}
                for w in self.messages[curr+v].domain:
                    if not w in self.separators[''.join(sorted([v,curr]))]:
                        # Call marginalize to remove variable v from messages[curr+v] domain
                        self.messages[curr+v] = self.messages[curr+v].marginalize(w)
                # Call push recursively and go to the next node v
                self.push(v, curr)

    def queryCluster(self, node, query):
        """
        `node`, a node in the elimination tree whose cluster contain the query variables.
        `query`, a list with query variables
        
        Returns factor with the marginal for the query variables
        """ 
        if self.messages is None:
            self.messages = self.getMessages()
        # fx is an auxiliary factor. Initialize fx with a *copy* of the factor associated with `node`    
        fx = self.factors[node].copy()
        for v in self.graph.children(node):
            # Call join to multiply the incoming messages from all neighbouring nodes to v        
            fx = fx*self.messages[v+node]
        for v in fx.domain:
            if v not in query:
                # Call marginalize to remove variable v from fx domain            
                fx = fx.marginalize(v)
        return fx

    def evidence(self, **q_evi):
        """
        `q_evi`, dictionary of evidence in the form of variables names and values
        
        Returns dictionary with evidence factors 
        """     
        # backup factors dict (so we can restore it later)
        self.backup_factors = self.factors.copy()
        # Create an empty dictionary
        lambdas = dict()
        for var, evi in q_evi.items():
            # create lambda factor
            lambdas[var] = Factor((var,), self.outcomeSpace)
            # Set probability table for the evidence indicator 
            for outcome in self.outcomeSpace[var]:
                if outcome == evi:
                    lambdas[var][outcome] = 1.
                else:
                    lambdas[var][outcome] = 0.
            
            # join factor with lambda
            for node, factor in self.factors.items():
                if var in factor.domain:
                    self.factors[node] *= lambdas[var]
                    break

        return lambdas

    def reset(self):
        self.factors = self.backup_factors


class JoinTree(EliminationTree):
    def __init__(self, graph, clusters, separators, outcomeSpace):
        self.graph = graph
        self.separators = separators # dict mapping each edge node1+node2 -> tuple of separator vars
        self.clusters = clusters
        self.outcomeSpace = outcomeSpace

        self.factors = {}
        for node in self.graph:
            # trivial factor (Factor's initialize to all 1's by default)
            self.factors[node] = Factor(self.clusters[node], outcomeSpace, trivial=True) 

        self.messages = None

    def show(self, positions=None):
        '''
        A specialised function to show JoinTrees, including the separators and clusters
        '''
        dot = graphviz.Graph(engine="neato", comment='Undirected graph', strict=True)        
        dot.attr(overlap="false", splines="true")
        for v in self.graph:
            if positions is not None:
                dot.node(str(v), label=str(v)+'\n'+','.join(self.clusters[v]), pos=positions[v])
            else:
                dot.node(str(v), label=str(v)+'\n'+','.join(self.clusters[v]))
        for v in self.graph:
            for w in self.graph.children(v):
                if v < w:
                    dot.edge(str(v), str(w), ','.join(self.separators[str(v)+str(w)]))

        return dot

    def distribute_factors(self, factor_list):
        '''
        Takes a list of factors and adds them one by one to the jointree
        '''
        for factor in factor_list:
            for node in self.graph:
                # We will find a match if the factor domain is a subset of the cluster (trivial factor) domain
                if set(factor.domain).issubset(self.clusters[node]):
                    self.factors[node] *= factor
                    break
            else:
                # This else clause will only be executed if the for loop reaches the end. Google "python for/else" for more info
                raise NameError('FamilyPreservationError')

