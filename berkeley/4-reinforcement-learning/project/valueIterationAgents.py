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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        v_values = dict()  # need to select the max of these

        # init V0(s) = 0
        for s in self.mdp.getStates():
            v_values[(0, s)] = dict()  # (action, value)

        for k in range(1, self.iterations):
            print('k = ', k)
            for s_state in self.mdp.getStates():

                local_q_values = dict()

                for action in self.mdp.getPossibleActions(s_state):

                    transition = self.mdp.getTransitionStatesAndProbs(s_state, action)

                    q_value = 0

                    # Sum s' T(s, a, s') * [R(s, a, s') + Gamma * Vk(s')]
                    # s = state, a = actions, s' = next_state, T(s, a, s') = probability
                    for (next_state, probability) in transition:
                        # R(s, a, s')
                        reward = self.mdp.getReward(s_state, action, next_state)

                        # Vk(s') todo
                        v_i_action, v_i_value = self.get_max_q_value_for_k_and_state(k-1, next_state, v_values)

                        q_value += probability * (reward + (self.discount * v_i_value))

                    local_q_values[action] = q_value

                # todo when actions = 0
                """
                   Note: Make sure to handle the case when a state has no available actions in an MDP 
                   (think about what this means for future rewards).
                """

                v_values[(k, s_state)] = local_q_values if (len(local_q_values) > 0) else dict()

                # get the last iteration values
                for (i, s) in v_values:
                    if i is (self.iterations - 2):
                        self.values[s] = v_values[(i, s)]

        print('end v iteration')
        #print('v_values = ', v_values)
        #print('self_values = ', self.values)

    def get_max_q_value_for_k_and_state(self, k, state, v_values):
        """
        return highest => v_i_action, v_i_value
        """
        filtered = v_values[(k, state)]
        if len(filtered) == 0:
            return None, 0
        return sorted(filtered.items(), key=lambda item: item[1], reverse=True)[0]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        print('calling get values state = ', state)
        print('get values result = ', self.values[state])
        q_values = self.values[state]

        if len(q_values) == 0:
            return 0

        action, value = sorted(q_values.items(), key=lambda item: item[1], reverse=True)[0]

        return value

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          returns the Q-value of the (state, action) pair given by the value function given by self.values
        """
        "*** YOUR CODE HERE ***"
        q_values = self.values[state]

        if len(q_values) == 0 or (action not in q_values):
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
        """
        "*** YOUR CODE HERE ***"
        q_values = self.values[state]
        if len(q_values) == 0:
            return None

        action, value = sorted(q_values.items(), key=lambda item: item[1], reverse=True)[0]
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
