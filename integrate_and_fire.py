import tqdm

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def create_if_network(num_nodes,num_edges=10,p=0.5,net_type="full"):
    if net_type == 'random':
        graph = nx.gnp_random_graph(num_nodes,p)
    elif net_type == 'ws':
        graph = nx.watts_strogatz_graph(num_nodes, num_edges, p)
    elif net_type == 'ba':
        graph = nx.barabasi_albert_graph(num_nodes, num_edges)
    elif net_type == 'full':
        graph = nx.complete_graph(num_nodes)
        
    return graph

class simple_integrate_and_fire_model:
    """
    Implements a simplified version of the integrate and fire
    model described in https://arxiv.org/pdf/0712.1003.pdf using
    separated time-scales where the driving rate is much slower
    than the relaxation rate. This makes a rule-based algorithm with
    a driving step followed by a relaxation step where an avalanche
    is allowed to complete before the next drive step, much like
    the BTW sandpile model.
    
    For the sake of simplicity the leak terms and dynamic synapses
    are left out due to the smaller time scales at which they are influential.
    """
    
    def __init__(self,network,v_ext=0.025,v_th=1,u=0.2,J=4):
        # Set parameters
        self.network = network
        self.v_ext = v_ext
        self.v_th = v_th
        self.u = u
        self.J = J
        
        # Retrieve J_ij (weight matrix w) from network
        self.w = self.J * nx.adjacency_matrix(network)
        
        # Initialize membrane potentials randomly
        self.v = np.random.uniform(0,self.v_th,self.w.shape[0])
        
        # Statistics
        self.avalanche_size = np.array([],dtype=int)
        
    def simulate(self,steps):
        avalanche_size = np.zeros(steps,dtype=int)
        
        with tqdm.tqdm(total=steps) as pbar:
            for t in range(steps):
                # Drive step
                i = np.random.randint(self.v.size)
                self.v[i] += self.v_ext

                # Initialize check list
                check_nodes = [i]

                # Relaxation step
                s = 0
                while len(check_nodes) > 0:
                    i = check_nodes.pop(0)

                    if self.v[i] > self.v_th:
                        # number of neighbors of i
                        n = np.sum(self.w[:,i].size)

                        # neighbor indices
                        j = self.w[:,i].nonzero()[0]

                        # Spiking results in firing potential to neighbors
                        self.v[j] += self.u*self.w[j,i].toarray().flatten() / n

                        # Add neighbors to check list
                        check_nodes += [elem for elem in list(j) if elem not in check_nodes]

                        # Subtract threshold potential after spike
                        self.v[i] -= self.v_th

                        # Increase current avalanche size
                        s += 1

                avalanche_size[t] = s
                
                pbar.update()
            
        self.avalanche_size = np.concatenate((self.avalanche_size,avalanche_size))
        
    def reset_avalanche_stats(self):
        self.avalanche_size = np.array([],dtype=int)
        
    def avalanche_size_pdf(self):
        pdf = np.bincount(self.avalanche_size) / self.avalanche_size.size
        
        nonzeros = (pdf != 0)
        indices = np.arange(nonzeros.size)[nonzeros]
        
        return indices,pdf[nonzeros]

class LHG_integrate_and_fire_model:
    """
    Implements the leaky integrate and fire model described in 
    https://ediss.uni-goettingen.de/bitstream/handle/11858/00-1735-0000-0006-B3B3-B/levina.pdf?sequence=1.
    
    The algorithm allows for leak terms, dynamic synapses and inhibition.
    
    The model operates on the assumption of separated time scales. Meaning that a neuronal
    avalanche is allowed to complete long before the next external input is applied. This
    means that the system is slowly driven and reacts quickly to any perturbation. The
    discretization is done by setting the driving rate to 1 and setting the time step
    size equal to the driving rate. The relaxation is assumed to be fast enough to 
    happen instantly after the driving step and to finish at around the same time (t + dt ~ t). 
    This is done due to the event driven nature of the system dynamics.
    
    Leakage and synapse recovery are both applied using the values at the start 
    of the time step for the time integration.
    
    Inhibition is implemented in two possible ways: 
    (1) Ignoring Dale's principle and taking a percentage of the 
    synapses to be inhibitory or (2) obeying Dale's principle and
    taking a percentage of the neurons to be inhibitory.
    
    TODO:
    - Implement leak terms - done (Levina claims: no relevant dynamics changes)
    - Implement dynamic synapses - done
    - Implement inhibitory neurons - done (Levina claims: no relevant dynamics changes)
    """
    
    def __init__(self,network,v_ext=0.025,v_th=1,u=0.2,a=0.5,
                 nu=10,tl=40,C=0.98,leakage=False,p_inh=0.0,inh_type="neuron"):
        """
        network: networkx network object
            Network used for the simulation
        v_ext: float
            External input added to the potential 
            of a neuron during the driving step
        v_th: float
            Membrane potential treshold
        u: float
            Transmitter resource usage / 
            saturation constant of synaptic strength
        
        Dynamic synapse parameters:
            a: float
                Maximum connection strength parameter (a / u = J_max)
            nu: float
                Synaptic recovery time scale parameter
        
        Leak term parameters:
            tl: float
                Rate of leakage from a node
            C: float
                Compensatory synaptic current
            leakage: bool
                Determine if leakage occurs
                
        p_inh: float
            Percentage of inhibitory synapses
        inh_type: string ("neuron" or "synapse")
            Type of inhibition: per synapse (ignoring Dale's principle)
            or per neuron (obeying Dale's principle)
        """
        
        # Set parameters
        self.network = network
        self.v_ext = v_ext
        self.v_th = v_th
        self.u = u
        self.a = a
        self.nu = nu
        self.tl = tl
        self.C = C
        self.leakage = leakage
        
        # Retrieve J_ij (weight matrix w) from network
        self.w = nx.adjacency_matrix(network)
        
        # Link indices in weight matrix
        self.link_idx = (self.w != 0)
        
        # Network size
        self.N = self.w.shape[0]
        
        # Determine synapse types
        self.synapse_types = self.w.astype(int)
        
        # Inhibition determined per synapse
        if inh_type == "synapse":
            n_inh = int(p_inh * self.synapse_types.size)

            type_vals = np.ones(self.synapse_types.size)
            if n_inh > 0:
                inhibitory = np.random.choice(np.arange(self.synapse_types.size),size=n_inh,replace=False)
                type_vals[inhibitory] = -1

            self.synapse_types[self.link_idx] = type_vals
        
        # Inhibition determined per neuron
        elif inh_type == "neuron":
            n_inh = int(p_inh * self.N)
            
            if n_inh > 0:
                inhibitory = np.random.choice(np.arange(self.N),size=n_inh,replace=False)
                self.synapse_types[:,inhibitory] *= -1
        
        # Randomize synaptic connection strengths
        self.w = self.w.astype(float)
        self.w[self.link_idx] = np.random.uniform(0,self.a/self.u,self.link_idx.size)
        
        # Synaptic recovery time-scale
        self.tj = self.nu * self.N
        
        # Initialize membrane potentials randomly
        self.v = np.random.uniform(0,self.v_th,self.N)
        
        # Statistics
        self.avalanche_size = np.array([],dtype=int)
        
    def simulate(self,steps):
        # Analytics to be collected
        avalanche_size = np.zeros(steps,dtype=int)
        
        with tqdm.tqdm(total=steps) as pbar:
            for t in range(steps):
                # Apply leakage terms
                if self.leakage:
                    self.v += self.C - self.v / self.tl
                
                # Drive step
                i = np.random.randint(self.v.size)
                self.v[i] += self.v_ext

                # Synaptic recovery term
                J_rec = (self.a / self.u - self.w[self.link_idx]) / self.tj
                
                # Initialize check list
                check_nodes = [i]

                # Relaxation step
                s = 0
                while len(check_nodes) > 0:
                    i = check_nodes.pop(0)

                    if self.v[i] > self.v_th:
                        # number of neighbors of i
                        n = np.sum(self.w[:,i].size)

                        # neighbor indices
                        j = self.w[:,i].nonzero()[0]

                        # Spiking results in firing potential to neighbors
                        # Inhibitory neurons subtract potential instead of adding
                        self.v[j] += self.synapse_types[j,i].toarray().flatten() * \
                                     self.u * self.w[j,i].toarray().flatten() / n
                        
                        # Decrease synaptic connection strength
                        self.w[j,i] -= self.u*self.w[j,i]
                        
                        # Add neighbors to check list
                        check_nodes += [elem for elem in list(j) if elem not in check_nodes]

                        # Subtract threshold potential after spike
                        self.v[i] -= self.v_th

                        # Increase current avalanche size
                        s += 1

                # Enforce minimum potential of zero
                self.v = np.maximum(np.zeros(self.v.size),self.v)
                
                # Apply synaptic recovery after relaxation
                self.w[self.link_idx] += J_rec
                
                avalanche_size[t] = s
                
                pbar.update()
            
        self.avalanche_size = np.concatenate((self.avalanche_size,avalanche_size))
        
    def reset_avalanche_stats(self):
        self.avalanche_size = np.array([],dtype=int)
        
    def avalanche_size_pdf(self):
        pdf = np.bincount(self.avalanche_size) / self.avalanche_size.size
        
        nonzeros = (pdf != 0)
        indices = np.arange(nonzeros.size)[nonzeros]
        
        return indices,pdf[nonzeros]