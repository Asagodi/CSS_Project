import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def izhikevich(adj_mat, neg_nodes, time=10):
        #set part of neurons to inhibitory
        nnodes = adj_mat.shape[0]
        nodes = range(nnodes)
        neg_num = len(neg_nodes)

        #random distributions
        rall = np.random.rand(nnodes,1)
        re = np.random.rand(nnodes-neg_num,1)
        ri = np.random.rand(neg_num,1)
        
        #set up parameters
        a = 0.1*np.ones((nnodes,1))
        a[neg_nodes] = 0.02+0.08*ri
        
        b = 0.26*np.ones((nnodes,1))
        b[neg_nodes] = 0.25-0.05*ri
        
        c =  -65+15*rall**2
        c[neg_nodes] = -65*np.ones((neg_num,1))
        
        d = 8-6*rall**2
        d[neg_nodes] = 2*np.ones((neg_num,1))
        
##        adj_mat = adj_mat * np.random.rand(nnodes, nnodes)
        
        v = -65*np.ones((nnodes,1)) # Initial values of v
        u = b*v                  
        all_act = np.zeros((nnodes, time))                    # spike timing
        
        #simulation
        for t in range(time):
            #random (sensory) input
            I=8*np.random.rand(nnodes,1)
            I[neg_nodes] = np.random.rand(neg_num,1) # input
            
            fired = np.where(v>=30)[0]
##            print(v)
            all_act[fired,t] = 1
            
            v[fired]=c[fired]
            u[fired]=u[fired]+d[fired]
            
            I = I + np.sum(adj_mat[:,fired],1)
            
            v=v+0.5*(0.04*np.multiply(v,v)+5*v+140-u+I)  #step 0.5 ms for numerical stability
            v=v+0.5*(0.04*np.multiply(v,v)+5*v+140-u+I)   
            u=u+np.multiply(a,np.multiply(b,v)-u)             

        return all_act

def create_network(size, neighbours, frac_edge=0.1, frac_neg=0.2,
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


adj_mat, neg_nodes = create_network(size=50, neighbours=2,
                                    frac_neg=0.5, net_type='barabasi')
all_act = izhikevich(adj_mat, neg_nodes, time=500)
fig = plt.figure()
im =  plt.imshow(all_act, animated=True)

plt.show()
