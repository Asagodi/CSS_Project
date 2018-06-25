import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def izhikevich(adj_mat, neg_nodes, time=10,
               a_pars=(.02, .02), b_pars=(.2, .25),
               c_pars=(-65, 15), d_pars=(8, -6, 2),
               randomness=(0.08, 0.05)):
        #set part of neurons to inhibitory
        nnodes = adj_mat.shape[0]
        nodes = range(nnodes)
        neg_num = len(neg_nodes)

        #random distributions
        rall = np.random.rand(nnodes,1)
        re = np.random.rand(nnodes-neg_num,1)
        ri = np.random.rand(neg_num,1)
        
        #set up parameters
        a = a_pars[0]*np.ones((nnodes,1))
        a[neg_nodes] = a_pars[1]+randomness[0]*ri
        
        b = b_pars[0]*np.ones((nnodes,1))
        b[neg_nodes] = b_pars[1]-randomness[1]*ri
        
        c =  c_pars[0]+c_pars[1]*rall**2
        c[neg_nodes] = c_pars[0]*np.ones((neg_num,1))
        
        d = d_pars[0]+d_pars[1]*rall**2
        d[neg_nodes] = d_pars[2]*np.ones((neg_num,1))
        
##        adj_mat = adj_mat * np.random.rand(nnodes, nnodes)
        
        v = c_pars[0]*np.ones((nnodes,1)) # Initial values of v
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

def get_avalanches(data):
    aval_times = []
    aval = 0
    aval_sizes = []
    proj_act = np.sum(data, 0)
    for t,i in enumerate(proj_act):
        if i > 0 and aval == 0:
            aval = 1
            start_t = t
            aval_size = int(i)
        if i > 0 and aval == 1:
            aval_size += int(i)
        
        if i == 0 and aval == 1:
            aval_times.append(int(t-start_t))
            aval_sizes.append(int(aval_size))
            aval = 0
    return aval_times, aval_sizes

