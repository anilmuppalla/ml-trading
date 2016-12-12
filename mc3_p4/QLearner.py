"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.99, \
        radr = 0.999, \
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
        self.q = np.random.uniform(low=-1.0, high=1.0, size=(num_states,num_actions))
        
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        
        self.s = s
        prob = rand.uniform(0.0, 1.0)
        if prob <= self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q[s,:])
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
        if prob < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = self.q[s_prime,:].argmax()
        
        self.rar = self.rar * self.radr
        
        self.q[self.s,self.a] = (1-self.alpha) * self.q[self.s,self.a] + self.alpha * (r + self.gamma * self.q[s_prime, :].max())

        self.s = s_prime
        self.a = action
        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
