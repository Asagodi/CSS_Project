def izhikevich(self, time, frac_neg = 0.1, record = 1):
        #set part of neurons to inhibitory
        self.set_negative_nodes(fraction = frac_neg)
        neg_num = np.floor(frac_neg*self.nnodes)
#         print neg_num, len(self.neg_nodes)
        
        #random distributions
        rall = np.random.rand(self.nnodes,1)
        re = np.random.rand(self.nnodes-neg_num,1)
        ri = np.random.rand(neg_num,1)
        
        #set up parameters
        a = 0.02*np.ones((self.nnodes,1))
        a[self.neg_nodes] = 0.02+0.08*ri
        
        b = 0.2*np.ones((self.nnodes,1))
        b[self.neg_nodes] = 0.25-0.05*ri
        
        c =  -65+15*rall**2
        c[self.neg_nodes] = -65*np.ones((neg_num,1))
        
        d = 8-6*rall**2
        d[self.neg_nodes] = 2*np.ones((neg_num,1))
        
        #S=[0.5*rand(Ne+Ni,Ne), -rand(Ne+Ni,Ni)];
        #should multiply wights with random number?
        self.matrix = self.matrix * np.random.rand(self.nnodes, self.nnodes)
        
        v = -65*np.ones((self.nnodes,1)) # Initial values of v
        u = b*v                  
        all_act = np.zeros((self.nnodes, time))                    # spike timings
                        
                        
        recorded_act = np.zeros((record, time))
        
        #simulation
        for t in range(time):
            #random (sensory) input
            I=5*np.random.rand(self.nnodes,1)
            I[self.neg_nodes] = np.random.rand(neg_num,1) # thalamic input
            
            fired = np.where(v>=30)[0]

            all_act[fired,t] = 1
            
            v[fired]=c[fired]
            u[fired]=u[fired]+d[fired]
            
            try:
                I = I + np.sum(self.matrix[:,fired],2)
            except:
                0
            
            v=v+0.5*(0.04*v**2+5*v+140-u+I)  #  step 0.5 ms for numerical stability
            v=v+0.5*(0.04*v**2+5*v+140-u+I)   
            u=u+a*(b*v-u)               
            
            if record > 0:
                recorded_act[:, t] = v[:record]

        return all_act, recorded_act
