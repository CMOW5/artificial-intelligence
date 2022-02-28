# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    NO_ACTION_NO_VALUE = {None: 0}

    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        # todo: should we get rid of self.values() ??
        # self.values holds the values (V* or Q*) of the last iteration
        # which we can easily get calling self.get_q_star_for_k_and_state(self.iterations, state)
        self.values = util.Counter()  # A Counter is a dict with default 0

        # q_k_values holds the q values for every iteration k
        # q_k_values (key) => (k, state)
        #            (value) => dict({action1: value, action2: value, ...})
        self.q_k_values = dict()
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        """ 
        we need to do 1 more iteration since a policy synthesized from values of depth k 
        (which reflect the next k rewards) will actually reflect the next k+1 rewards (i.e. you return πk+1). 
        Similarly, the Q-values will also reflect one more reward than the values (i.e. you return Qk+1).
        """

        # init V0(s)
        self.init_v_0()

        for k in range(1, self.iterations + 2):
            for state in self.mdp.getStates():
                q_values = dict()
                actions = self.mdp.getPossibleActions(state)

                """
                  Note: Make sure to handle the case when a state has no available actions in an MDP 
                  (think about what this means for future rewards).
                """
                if len(actions) == 0:
                    q_values = ValueIterationAgent.NO_ACTION_NO_VALUE

                # ArgMax (Sum s' T(s, a, s') * [R(s, a, s') + Gamma * Vk(s')])
                # s = state, a = actions, s' = next_state, T(s, a, s') = probability
                for action in actions:
                    q_value = self.calculate_q_value(state, action, k)
                    q_values[action] = q_value

                self.q_k_values[(k, state)] = q_values

        # get the last iteration values, no really needed since we can get these using
        # get_q_star_for_k_and_state(self.iterations, state)
        for (i, s) in self.q_k_values:
            if i is self.iterations:
                self.values[s] = self.q_k_values[(i, s)]

    def init_v_0(self):
        for state in self.mdp.getStates():
            # no action and no value for V0s
            self.q_k_values[(0, state)] = ValueIterationAgent.NO_ACTION_NO_VALUE

    def calculate_q_value(self, state, action, k):
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0

        # Sum s' T(s, a, s') * [R(s, a, s') + Gamma * Vk(s')]
        # s = state, a = actions, s' = next_state, T(s, a, s') = probability
        for (next_state, probability) in transition:
            # R(s, a, s')
            reward = self.mdp.getReward(state, action, next_state)

            # Vk(s')
            v_k_action, v_k_value = self.get_q_star_for_k_and_state(k - 1, next_state)

            q_value += probability * (reward + (self.discount * v_k_value))

        return q_value

    def get_q_values_for_k_and_state(self, k, state):
        return self.q_k_values[(k, state)]

    def get_q_star_for_k_and_state(self, k, state):
        """
        Return highest => v_i_action, v_i_value for iteration k and state
        """
        q_values = self.get_q_values_for_k_and_state(k, state)
        return sorted(q_values.items(), key=lambda item: item[1], reverse=True)[0]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        #q_values = self.values[state] # todo: should we get rid of self.values() ??
        action, value = self.get_q_star_for_k_and_state(self.iterations, state)
        return value

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          returns the Q-value of the (state, action) pair given by the value function given by self.values

          we need to do 1 more iteration since a policy synthesized from values of depth k
          (which reflect the next k rewards) will actually reflect the next k+1 rewards (i.e. you return πk+1).
          Similarly, the Q-values will also reflect one more reward than the values (i.e. you return Qk+1).

        """
        "*** YOUR CODE HERE ***"
        # get Q(k + 1)
        q_values = self.get_q_values_for_k_and_state(self.iterations + 1, state)

        # todo: we can get rid of this safeguard
        if action not in q_values:
            return 0

        return q_values[action]

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          computes the best action according to the value function given by self.values.

          we need to do 1 more iteration since a policy synthesized from values of depth k
          (which reflect the next k rewards) will actually reflect the next k+1 rewards (i.e. you return πk+1).
          Similarly, the Q-values will also reflect one more reward than the values (i.e. you return Qk+1).
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        # get PI(k + 1)
        action, value = self.get_q_star_for_k_and_state(self.iterations + 1, state)
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)