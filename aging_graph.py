import numpy as np
import networkx as nx

def aging_barabasi_albert_graph(N,alpha,m=2,m0=2):
    """
    Implements the modified barabasi-albert model given
    in the paper titled "Self-organized Criticality in an 
    Integrate-and-Fire Neuron Model Based on Modified Aging Networks".
    
    Each time step a random node is chosen based on its age to generate
    a new node linked to it, which also links to its (m-1) nearest neighbors
    with a probability given by the standard barabasi-albert model.
    
    N: int
        Network size
    alpha: float
        Age factor which influences probabilities based on age
    m: int
        Number of links for each new node
    m0: int
        Initial size of the network
    """
    
    # Initialize fully connected graph with m0 nodes
    graph = nx.complete_graph(m0)
    
    age = np.ones(N)
    for n in range(m0,N):
        # Get degrees for all current nodes in array form
        degrees = np.array(list(graph.degree(np.arange(n)).values()))
        
        # Calculate pmf for a node to generate new node
        i_probs = degrees / age[:n]**(alpha)
        i_probs /= np.sum(i_probs)
        
        # Choose a random node i
        i = np.random.choice(np.arange(n),p=i_probs)
        
        # neighbors j of i
        neighbors = np.array(graph.neighbors(i))
        
        # Calculate pmf for linked nearest neighbors of i
        j_probs = degrees[neighbors] / np.sum(degrees[neighbors])
        
        # Choose a random node i
        j = list(np.random.choice(neighbors,size=(m-1),replace=False,p=j_probs))
        
        # Add node to network with chosen edges
        graph.add_edges_from(zip([n]*m,[i] + j))
        
        # Advance the time and apply to age of nodes
        age[:n] += 1
    
    return graph
