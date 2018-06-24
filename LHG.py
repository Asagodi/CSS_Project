import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def LHG(adj_mat, time=10):
        #set part of neurons to inhibitory
        nnodes = adj_mat.shape[0]
        nodes = range(nnodes)
        
        v = np.zeros((nnodes,1)) # Initial values of v
        all_act = np.zeros((nnodes, time))
        u = 1
        
        #simulation
        for t in range(time):
            #random (sensory) input
            I=0.025*np.ones((nnodes,1))
##            I[neg_nodes] = 0.025*np.ones(nnodes,1) # input
            
            fired = np.where(v>=1)[0]
            print(fired.shape)

            all_act[fired,t] = 1
            
            I = I + np.sum(adj_mat[:,fired],1)/(nnodes-1)
            
##            print(all_act[:,t])    
            reset = np.multiply(np.zeros((nnodes,1)), all_act[:,t-1])
            v=v+u*I-2*np.multiply(reset, v)
##            print(v)

        return all_act

def create_network(size, neighbours=1, frac_edge=0.1, frac_neg=0.2,
                   net_type='random'):
    if net_type == 'random':
        graph = nx.gnm_random_graph(size, neighbours)
    elif net_type == 'ws':
        graph = nx.watts_strogatz_graph(size, neighbours, frac_edge)
    elif net_type == 'barabasi':
        graph = nx.barabasi_albert_graph(size, neighbours)
    elif net_type == 'full':
        graph = nx.complete_graph(size)
    
    adj_mat = nx.to_numpy_matrix(graph)

    nodes = range(size)
    neg_nodes = np.random.choice(nodes, int(np.floor(frac_neg*size)))
    for n in neg_nodes:
        adj_mat[n,:][np.nonzero(adj_mat[n,:])] = -1
    
    return adj_mat, neg_nodes


adj_mat, neg_nodes = create_network(size=5, net_type='full')
all_act = LHG(adj_mat, time=500)
fig = plt.figure()
im =  plt.imshow(all_act, animated=True)

plt.show()
