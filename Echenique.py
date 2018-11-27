# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.6
# ---

# %% [markdown]
# This notebook implements the algorithm in:
#
# Echenique, F. (2007). Finding all equilibria in games of strategic complements. Journal of Economic Theory, 135(1), 514-532.
#
# I replicate the two examples described in the original paper and compare the execution time with an exhaustive search method.
#
# My implementation works for games in which individual strategy spaces are of the form $S_i = [k, k+1, k+2,..., k+m]$, where $k$ and $m$ are integers. One must simply define the set of players, strategy spaces as a list $S = [S_1, S_2,..., S_n]$ and the payoff function.
#
# For ease of implementation, payoff functions are implemented as a single function that receives the index of the player to which the payoff corresponds: $$\forall i\in N, s\in S, \quad U(i,s) = u_i(s)$$

# %% [markdown]
# # Class definitions

# %%
import numpy as np
import itertools as it
import time
from numba import jit

class game:
    """
    A class representing non-cooperative games.
    """
    def __init__(self, N, S, U):
        self.N = N
        self.S = S
        self.U = U
        self.n = len(N)
    
    def eval_util(self,i,si,s):
        """
        Evaluate the utility of player i from actions si in S[i],
        given that the rest of agents act according to s
        """
        
        # If no strategy is passed, return -Inf
        if len(si) == 0:
            
            return([-float("Inf")])
            
        else:
            
            # Generate strategy profiles
            profs = [ s[:i] + (x,) + s[(i+1):] for x in si ]
            
            utils = [self.U(x,i) for x in profs]
            
            return(utils)
    
    def best_response(self,i,s):
        """
        Find the utility maximizing actions of player i given that
        all other players act according to s.
        """
        # Find utilities
        utils = self.eval_util(i,self.S[i],s)
        
        # Find utility maximizing actions
        best = [self.S[i][x] for x in range(0,len(utils)) if utils[x] == max(utils)]
        
        return best
    
    def joint_best_response(self,s):
        """
        Find the best response of all agents to a strategy profile s.
        """
        br = [self.best_response(x,s) for x in range(self.n)]
        
        return br
    
    def joint_util(self,s):
        """
        Find the utility of all agents for a given strategy profile.
        """
        util = [self.U(s,x) for x in range(self.n)]
        return(util)
            
    def nash_eq_exhaus(self):
        """
        Exhaustively find all the pure-strategy Nash equilibria of game.
        """
        # Generate strategy space
        S = list(it.product(*self.S))
        
        # Find nash equiibria
        neq = [s for s in S\
               if all([s[x] in self.joint_best_response(s)[x] for x in range(self.n)])]
        
        return(set(neq))
        
    
class sm_game(game):
    """
    Class providing special methods for supermodular games.
    """
    
    def sup_eq(self):
        """
        Find supremum of the pure-strategy equilibrium set.
        """
        a0 = tuple([max(x) for x in self.S])
        a1 = tuple([max(x) for x in self.joint_best_response(a0)])
        
        while a0 != a1:
            
            a0 = a1[:]
            a1 = tuple([max(x) for x in self.joint_best_response(a0)])
            
        return(a1)
    
    def inf_eq(self):
        """
        Find infimum of the pure-strategy equilibrium set.
        """
        # Find least action of all players
        a0 = tuple([min(x) for x in self.S])
        a1 = tuple([min(x) for x in self.joint_best_response(a0)])
        
        while a0 != a1:
            
            a0 = a1[:]
            a1 = tuple([min(x) for x in self.joint_best_response(a0)])
            
        return(a1)
    
    def restrict(self,s):
        """
        Create a supermodular game in which all players are restricted
        to strategies greater or equal to s. 
        """
        # Reduce the strategy space to those greater or equal to s
        restr_s = [ [y for y in self.S[x] if y >= s[x] ] for x in range(self.n) ]
        
        return( sm_game(self.N, restr_s, self.U) )
    
    
    
    
    def echenique(self, print_state = False):
        """
        Apply the algorithm in
        Echenique, F. (2007).
        "Finding all equilibria in games of strategic complements."
        The algorithm finds the pure-strategy equilibrium set.
        """
        # Find extremal equilibria
        infeq = self.inf_eq()
        supeq = self.sup_eq()
        
        # If the supremum and infimum equilibria are the same, return them as the only equilibrium.
        if infeq == supeq:
            return set([infeq])
        
        # Initialize equilibrium list
        eq_list = set([infeq,supeq])
        
        # Initialize state
        M = set([(infeq,infeq)])
        if print_state:
            print("M = ",[x[0] for x in M])
        
        # Find stopping state
        M_final = set([(supeq,
                        tuple([min(x) for x in self.joint_best_response(supeq)]))])
        
        # Apply a subroutine until the state of the algorithm reaches its final form
        while M != M_final:
            
            M_prime = set()
            
            for s_s in M:
                
                s = s_s[0][:]
                s_star = s_s[1][:]
                
                for i in range(self.n):
                    
                    if s[i] < supeq[i]:
                        
                        # Step (1)
                        
                        # Create profile where i increments its action by 1.
                        s_greater = s[:i] + (s[i]+1,) + s[(i+1):]
                        
                        # Find infimum equilibrium of game with greater actions.
                        gamma_r = self.restrict(s_greater)
                        s_hat = gamma_r.inf_eq()
                        
                        # Step (2)
                        
                        u_hat = self.joint_util(s_hat)
                        
                        # Find the restricted set of strategies
                        z = [[y for y in self.S[x] if s_star[x] <= y and y < s_greater[x]]\
                             for x in range(self.n)]
                        
                        # Find maximum utilities that all agents would receive from
                        # deviating from s_hat to a strategy in z[i]
                        u_alt = [ max(self.eval_util(x,z[x],s_hat)) for x in range(self.n)]
                        
                        # Check if any player is strictly better off switching.
                        # If none is, s_hat is an equilibrium
                        if not any([ u_alt[x] > u_hat[x] for x in range(self.n)]):
                            
                            eq_list.add(s_hat)
                        
                        # Step (3)
                        
                        # Find inf best response (s_hat)
                        brep = tuple([min(x) for x in self.joint_best_response(s_hat)])
                        
                        # Add to state
                        M_prime.add((s_hat,brep))
            
            # Update the state
            M = M_prime.copy()
            
            if print_state:
                print("M = ",[x[0] for x in M])
            
        return(eq_list)

# %% [markdown]
# # Section 5 Example

# %%
# Create game
N = [1, 2]
S = [[1, 2, 3, 4], [1, 2, 3, 4]]
def util(s, i):

    if i == 0:
        
        pay = np.matrix([[3,3,3,0],
                         [2,2,2,0],
                         [1,1,1,0],
                         [0,0,0,0]])
        
    elif i == 1:
        
        pay = np.matrix([[3,2,1,0],
                         [3,2,1,0],
                         [3,2,1,0],
                         [0,0,0,0]])
    
    return pay[s[0]-1,s[1]-1]

game = sm_game(N, S, util)

t0 = time.time()
eq = game.nash_eq_exhaus()
t1 = time.time()
print("Exhaustive result:",eq)
print("Exhaustive time:",t1-t0)

t0 = time.time()
eq = game.echenique(print_state=True)
t1 = time.time()
print("Echenique (2007) result:",eq)
print("Echenique (2007) time:",t1-t0)

# %% [markdown]
# # Section 7 game

# %%
# Parameters
K = 20
alpha = (0.5,0.5)
beta = (0.5,0.5)

# create game
N = [1,2]
S = [range(0,K+1) for x in range(2)]
def util(s, i, K, alpha, beta):
    s = tuple([100*x/K for x in s])
    u = -0.01*alpha[i]*(s[i]-s[1-i])**2 + 2*beta[i]*np.sin(100*s[i])+\
    0.0001*( (1-alpha[i])*s[i]*(1+s[1-i]) - 0.01*(0.5-beta[i])*s[i]**2 )
    return(u)

u = lambda s,i: util(s,i,K,alpha,beta)

game2 = sm_game(N,S,u)

t0 = time.time()
eq = game2.nash_eq_exhaus()
t1 = time.time()
print("Exhaustive result:",eq)
print("Exhaustive time:",t1-t0)

t0 = time.time()
eq = game2.echenique()
t1 = time.time()
print("Echenique (2007) result:",eq)
print("Echenique (2007) time:",t1-t0)
