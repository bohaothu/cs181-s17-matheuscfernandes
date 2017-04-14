#!/usr/bin/env python 

from util import * 
from numpy import *
import numpy as np
from math import log
import copy
import sys


# Pretty printing for 1D/2D numpy arrays
MAX_PRINTING_SIZE = 30

def format_array(arr):
    s = shape(arr)
    if s[0] > MAX_PRINTING_SIZE or (len(s) == 2 and s[1] > MAX_PRINTING_SIZE):
        return "[  too many values (%s)   ]" % s

    if len(s) == 1:
        return  "[  " + (
            " ".join(["%.6f" % float(arr[i]) for i in range(s[0])])) + "  ]"
    else:
        lines = []
        for i in range(s[0]):
            lines.append("[  " + "  ".join(["%.6f" % float(arr[i,j]) for j in range(s[1])]) + "  ]")
        return "\n".join(lines)



def format_array_print(arr):
    print format_array(arr)


def string_of_model(model, label):
    (initial, tran_model, obs_model) = model
    return """
    Model: %s 
    initial: 
    %s
    
    transition: 
    %s
    
    observation: 
    %s
    """ % (label,
           format_array(initial),
           format_array(tran_model),
           format_array(obs_model))

    
def check_model(model):
    """Check that things add to one as they should"""
    (initial, tran_model, obs_model) = model
    for state in range(len(initial)):
        assert((abs(sum(tran_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(obs_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(initial) - 1)) <= 0.01)


def print_model(model, label):
    check_model(model)
    print string_of_model(model, label)    

def max_delta(model, new_model):
    """Return the largest difference between any two corresponding 
    values in the models"""
    return max( [(abs(model[i] - new_model[i])).max() for i in range(len(model))] )


class HMM:
    """ HMM Class that defines the parameters for HMM """
    def __init__(self, states, outputs):
        """If the hmm is going to be trained from data with labeled states,
        states should be a list of the state names.  If the HMM is
        going to trained using EM, states can just be range(num_states)."""
        self.states = states
        self.outputs = outputs
        n_s = len(states)
        n_o = len(outputs)
        self.num_states = n_s
        self.num_outputs = n_o
        self.initial = zeros(n_s)
        self.transition = zeros([n_s,n_s])
        self.observation = zeros([n_s, n_o])

    def set_hidden_model(self, init, trans, observ):
        """ Debugging function: set the model parameters explicitly """
        self.num_states = len(init)
        self.num_outputs = len(observ[0])
        self.initial = array(init)
        self.transition = array(trans)
        self.observation = array(observ)
        
    def get_model(self):
        return (self.initial, self.transition, self.observation)

    def compute_logs(self):
        """Compute and store the logs of the model (helper)"""
        raise Exception("Not implemented")

    def __repr__(self):
        return """states = %s
        observations = %s
        %s
        """ % (" ".join(array_to_string(self.states)),
               " ".join(array_to_string(self.outputs)),
               string_of_model((self.initial, self.transition, self.observation), ""))

     
    # declare the @ decorator just before the function, invokes print_timing()
    @print_timing
    def learn_from_labeled_data(self, state_seqs, obs_seqs):
        """
        Learn the parameters given state and observations sequences. 
        The ordering of states in states[i][j] must correspond with observations[i][j].
        Use Laplacian smoothing to avoid zero probabilities.
        Implement for (a).
        """

        num_states=self.num_states

        theta=ones((num_states))
        for i in range(len(state_seqs)):
            theta[state_seqs[i][0]]+=1
        theta=theta/sum(theta)
        self.initial=theta

        t=ones((num_states,num_states))
        for i in range(len(state_seqs)):
            for k in range(len(state_seqs[i])-1):
                t[state_seqs[i][k],state_seqs[i][k+1]]+=1
        for s in range(num_states):
            t[s,:]=t[s,:]/sum(t[s,:])
        self.transition=t

        pi=ones((num_states,num_states))
        for i in range(len(state_seqs)):
            for k in range(len(state_seqs[i])):
                pi[state_seqs[i][k],obs_seqs[i][k]]+=1

        for s in range(num_states):
            pi[s,:]=pi[s,:]/sum(pi[s,:])
        self.observation=pi

        # raise Exception("Not implemented")
        

    def most_likely_states(self, sequence, debug=True):
        """Return the most like sequence of states given an output sequence.
        Uses Viterbi algorithm to compute this.
        Implement for (b) and (c).
        """
        #this is being coded as according to the wikipedia page and notation
        #https://en.wikipedia.org/wiki/Viterbi_algorithm

        pi=self.initial
        A=self.transition
        B=self.observation

        num_states=self.num_states
        num_sequence=len(sequence)

        #initializing the T's
        T1 = zeros((num_states, num_sequence))
        T2 = zeros((num_states, num_sequence))

        for i in range(num_states):
            T1[i,0]=np.log(pi[i])+np.log(B[i,sequence[0]])
            # T2[i,0]=0 this is implied, but not computed for efficiency
        for i in range(1,num_sequence):
            for j in range(num_states):
                T1[j,i]=np.log(B[j,sequence[i]])+max(T1[:,i-1]+np.log(A[:,j]))
                T2[j,i]=argmax(T1[:,i-1]+np.log(A[:,j]))
        zI=argmax(T1[:,-1])
        X=zeros(num_sequence)
        X[-1]=zI

        for i in range(num_sequence-1,0,-1):
            zI=T2[int(zI),i]
            X[i-1]=int(zI)

        X=list(X.astype(int))

        return X

        # raise Exception("Not implemented")

def get_wikipedia_model():
    # From the rainy/sunny example on wikipedia (viterbi page)
    hmm = HMM(['Rainy','Sunny'], ['walk','shop','clean'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.4,0.5], [0.6,0.3,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm

def get_toy_model():
    hmm = HMM(['h1','h2'], ['A','B'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.9], [0.9,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm
