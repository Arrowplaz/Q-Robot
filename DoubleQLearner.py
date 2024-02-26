from TabularQLearner import TabularQLearner

class DoubleQLearner:

    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9,
                  exploration = 'eps', epsilon = 0.98, epsilon_decay = 0.999, dyna = 0):
        # Double Q init.
        self.Q1 = TabularQLearner(states= states, actions= actions, alpha= alpha, gamma=gamma, exploration=exploration, epsilon=epsilon,epsilon_decay=epsilon_decay, dyna=dyna)
        self.Q2 = TabularQLearner(states= states, actions= actions, alpha= alpha, gamma=gamma, exploration=exploration, epsilon=epsilon,epsilon_decay=epsilon_decay, dyna=dyna)
    def train (self, s, r):
        # Double Q training.
        return a

    def test (self, s, allow_random = False):
        # Double Q test.
        return a

    def getStateValues (self):
        # Still needed for the test environment to work.
        # The exact same as Tabular Q Learner
        max_values = self.Q_table.max(axis=1)
        return np.array(max_values)