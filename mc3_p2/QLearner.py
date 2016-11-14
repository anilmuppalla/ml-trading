"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.q = np.random.uniform(low=-1.0,high=1.0,size=(num_states,num_actions))
        self.T = np.zeros((num_states, num_actions, num_states))
        self.Tc = np.full((num_states, num_actions, num_states), 0.000001)
        self.R = np.zeros((num_states, num_actions))
        self.alphaR = 0.9

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        prob = np.random.rand()
        if prob < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = self.q[s,:].argmax()
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        prob = np.random.rand()
        if prob<self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = self.q[s_prime,:].argmax()
        
        self.rar = self.rar * self.radr
        
        self.q[self.s,self.a] = (1-self.alpha) * self.q[self.s,self.a] + self.alpha * (r + self.gamma * self.q[s_prime, :].max())

        if self.dyna > 0:
            self.Tc[self.s,self.a,s_prime] += 1
            self.T[self.s,self.a,s_prime] = self.Tc[self.s,self.a,s_prime]/self.Tc[self.s,self.a,:].sum()
            self.R[self.s,self.a] = (1 - self.alpha)*(self.R[self.s,self.a]) + (self.alpha*r)
        
            for i in xrange(self.dyna):
                dynastate = rand.randint(0,self.num_states-1)
                dynaction = rand.randint(0,self.num_actions-1)
                s_new = self.T[dynastate,dynaction,:].argmax()
                dynaR = self.R[dynastate,dynaction]
                self.q[dynastate,dynaction] = (1 - self.alphaR) * (self.q[dynastate,dynaction]) + (self.alphaR)*(dynaR + self.gamma*self.q[s_new, :].max())

        self.s = s_prime
        self.a = action
        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
