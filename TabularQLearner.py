import numpy as np
import random
import pandas as pd

class TabularQLearner:

    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9,
                  exploration = 'eps', epsilon = 0.98, epsilon_decay = 0.999, dyna = 0):

        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploartion = exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna
        
        #Create an array to hold every experience tuple; used for dyna
        self.experience_history = []

        #Create the Q Table where each State(row) has a q value for each action(column)
        self.Q_table = pd.DataFrame(np.random.uniform(0,0.05, size=(states, actions)))
        #print(self.Q_table.to_string())

        #Holders for the most recent s and a pair
        self.old_s = None
        self.old_a = None

    def train (self, s, r, old_s = None, old_a = None, eval_function = None,
               select_function = None):

        # Receive new state s and new reward r. Update Q-table and return selected action.
        #
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        # In this part of the project, the optional parameters are not used.
        # How will you know the previous state and action?

        #I'm being given the <s', r> I need to figure out <s, a>
        #I could use instance variables to hold the most recent s and A
        
        #Decide whether to explore or exploit
        if select_function == "eps" and round(np.random.uniform(0.00, 1.00), 2) < self.epsilon:
            #Perform exploration
            a = random.randint(0, 3)
            self.epsilon *= self.epsilon_decay #Perform decay
        else:
            #Perform exploit
            a = self.Q_table.loc[s].idxmax()

        #Update Q Table
        part_one = (1 - self.alpha) * self.Q_table.loc[self.old_s, self.old_a]
        part_two = r + (self.gamma * (self.Q_table.iloc[s].max()))
        self.Q_table.loc[self.old_s, self.old_a] = part_one + (self.alpha * part_two)
            
        #Apped the experience tuple to the history
        self.experience_history.append((self.old_s, self.old_a, s, r))

        #Perform hallucination
        for _ in range(self.dyna):
            rand_index = np.random.randint(low=0, high=len(self.experience_history))
            rand_experience = self.experience_history[rand_index]
            hal_part_one = (1 - self.alpha) * self.Q_table.loc[rand_experience[0], rand_experience[1]]
            hal_part_two = rand_experience[3] + (self.gamma * (self.Q_table.iloc[rand_experience[2]].max()))
            self.Q_table.loc[rand_experience[0], rand_experience[1]] = hal_part_one + (self.alpha * hal_part_two)

        #Replace the old with the new
        self.old_s = s 
        self.old_a = a
        return a


    def test (self, s, select_function = None, allow_random = False):

        # Receive new state s. Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # You sometimes will, and sometimes won't, want to allow random actions...

        #If random actions is allowed and we select a number within the E threshhold, perform exploartion
        if allow_random and round(random.uniform(0.00, 1.00), 2) < self.epsilon:
            a = random.randint(0, 3)
            self.epsilon *= self.epsilon_decay #Perform decay
            #print("rand")
        else:
            #Peform exploitation, where we simply select the action with the highest Q value
            a = self.Q_table.loc[s].idxmax()

        #Assign the <s,a> pair to holders for later use
        self.old_s = s 
        self.old_a = a
        #print(s, a)
        return a


    def getStateValues (self):

        # Return the max Q value for every state as a 1-D numpy array.
        # This is needed for robot_env to draw its plots.
        max_values = self.Q_table.max(axis=1)
        return np.array(max_values)