import numpy as np
from warnings import warn
from collections import deque, Sequence

'''
Portfolio Vector Memory
'''
class PVM(Sequence): 
    def __init__(self, LEN, DIM):
        '''
            A PVM is a stack of portfolio vectors in chronological
            order. PVM is initialised with uniform weights. In each 
            training step, a policy network loads the portfolio vector
            of the previous period from the memory location at t-1, and 
            overwrites the memory at t with its output. As the parameters
            of the policy networks converge through many training epochs, 
            the values in the memory also converge. 
            
            Args: 
                len: Length of PVM
                dim: The dimension of the portfolio vector
        '''
        # TODO: Need to initialise deque with uniformly distributed 
        # weights of length 'len'.
        v = np.expand_dims(np.array([1/DIM for _ in range(DIM)]), axis=1)
        vec = [v for _ in range(LEN)]
        self.deque = deque(vec)
        super().__init__()

    def __getitem__(self,i): 
        return self.deque[i]

    def __len__(self): 
        return len(self.deque)

    def append(self, pv): 
        self.deque.append(pv)
    
    def pop(self): 
        return self.deque.pop()
    