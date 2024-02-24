import numpy as np
import random

class TabularQLearner:

    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9,
                  exploration = 'eps', epsilon = 0.98, epsilon_decay = 0.999, dyna = 0):

        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna

        #Create the q table as a table of zeros with states and each action possible
        #Policy is decided by going to a state and choosing the action with the highest Q value
        self.q_table = np.zeros(states, actions)


    def train (self, s, r, eval_function = None,
               select_function = None):

        # Receive new state s and new reward r. Update Q-table and return selected action.
        #
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        # In this part of the project, the optional parameters are not used.
        # How will you know the previous state and action?
        # Answer: Look at the Table

        if self.exploration == 'eps' and random.random() < self.epsilon:
            #Performing exploration
            a = random.randint(0, self.actions - 1)
            
            #Perform epsilon decay
            self.epsilon *= self.epsilon_decay
        
        else:
            #Peforming exploitation
            a = np.argmax(self.q_table[s]) #Select the action with the highest Q value from state S

        #Update Q table
        self.q_table[s, a] += self.alpha * (r + self.gamma * np.max(self.q_table[s]) - self.q_table[s, a])

        return a


    def test (self, s, select_function = None, allow_random = False):

        # Receive new state s. Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # You sometimes will, and sometimes won't, want to allow random actions...

        return a


    def getStateValues (self):

        # Return the max Q value for every state as a 1-D numpy array.
        # This is needed for robot_env to draw its plots.

        return None