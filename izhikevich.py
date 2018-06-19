import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def set_negative_weights(adj_mat, fraction = 0.5):
        i,j = np.nonzero(adj_mat)
        ix = np.random.choice(len(i), np.floor(fraction * len(i)), replace=False)
        adj_mat[i[ix], j[ix]] = -1

def izhikevich(adj_mat, time=10, frac_neg = 0.1):
        #set part of neurons to inhibitory
        nnodes = adj_mat.shape[0]
        nodes = range(nnodes)
        neg_nodes = np.random.choice(nodes, int(np.floor(frac_neg*nnodes)))
        for n in neg_nodes:
            adj_mat[n,:][np.nonzero(adj_mat[n,:])] = -1
        neg_num = int(np.floor(frac_neg*nnodes))

        #random distributions
        rall = np.random.rand(nnodes,1)
        re = np.random.rand(nnodes-neg_num,1)
        ri = np.random.rand(neg_num,1)
        
        #set up parameters
        a = 0.02*np.ones((nnodes,1))
        a[neg_nodes] = 0.02+0.08*ri
        
        b = 0.2*np.ones((nnodes,1))
        b[neg_nodes] = 0.25-0.05*ri
        
        c =  -65+15*rall**2
        c[neg_nodes] = -65*np.ones((neg_num,1))
        
        d = 8-6*rall**2
        d[neg_nodes] = 2*np.ones((neg_num,1))
        
        adj_mat = adj_mat * np.random.rand(nnodes, nnodes)
        
        v = -65*np.ones((nnodes,1)) # Initial values of v
        u = b*v                  
        all_act = np.zeros((nnodes, time))                    # spike timings
        
        #simulation
        for t in range(time):
            #random (sensory) input
            I=9*np.random.rand(nnodes,1)
            I[neg_nodes] = np.random.rand(neg_num,1) # input
            
            fired = np.where(v>=30)[0]

            all_act[fired,t] = 1
            
            v[fired]=c[fired]
            u[fired]=u[fired]+d[fired]
            
            try:
                I = I + np.sum(adj_mat[:,fired],2)
            except:
                0
            
            v=v+0.5*(0.04*v**2+5*v+140-u+I)  #  step 0.5 ms for numerical stability
            v=v+0.5*(0.04*v**2+5*v+140-u+I)   
            u=u+a*(b*v-u)               

        return all_act


ws = nx.watts_strogatz_graph(200,2,0.2)
adj_mat = nx.to_numpy_matrix(ws)
all_act = izhikevich(adj_mat, time=500, frac_neg = 0.25)
fig = plt.figure()
im =  plt.imshow(all_act, animated=True)

plt.show()
