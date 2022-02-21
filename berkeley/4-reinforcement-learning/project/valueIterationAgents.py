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
        self.q_k_values = dict()
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #v_k_values = dict()  # need to select the max of these

        # init V0(s) = 0
        """
        for s in self.mdp.getStates():
            local_q_values = dict()
            for a in self.mdp.getPossibleActions(s):
                local_q_values[a] = 0
            v_values[(0, s)] = local_q_values
        """

        """ 
        we need to do 1 more iteration since a policy synthesized from values of depth k 
        (which reflect the next k rewards) will actually reflect the next k+1 rewards (i.e. you return πk+1). 
        Similarly, the Q-values will also reflect one more reward than the values (i.e. you return Qk+1).
        """
        for k in range(0, self.iterations + 2):
            for state in self.mdp.getStates():
                if k == 0:
                    self.q_k_values[(0, state)] = {None: 0}
                    continue

                local_q_values = dict()

                actions = self.mdp.getPossibleActions(state)

                """
                  Note: Make sure to handle the case when a state has no available actions in an MDP 
                  (think about what this means for future rewards).
                """
                if len(actions) == 0:
                    local_q_values[None] = 0

                # ArgMax (Sum s' T(s, a, s') * [R(s, a, s') + Gamma * Vk(s')])
                # s = state, a = actions, s' = next_state, T(s, a, s') = probability
                for action in actions:
                    transition = self.mdp.getTransitionStatesAndProbs(state, action)
                    q_value = 0

                    # Sum s' T(s, a, s') * [R(s, a, s') + Gamma * Vk(s')]
                    # s = state, a = actions, s' = next_state, T(s, a, s') = probability
                    for (next_state, probability) in transition:
                        # R(s, a, s')
                        reward = self.mdp.getReward(state, action, next_state)

                        # Vk(s')
                        v_k_action, v_k_value = self.get_max_q_value_for_k_and_state(k - 1, next_state, self.q_k_values)

                        q_value += probability * (reward + (self.discount * v_k_value))

                    local_q_values[action] = q_value

                self.q_k_values[(k, state)] = local_q_values  # if (len(local_q_values) > 0) else dict()

        # get the last iteration values
        for (i, s) in self.q_k_values:
            if i is self.iterations:
                self.values[s] = self.q_k_values[(i, s)]


    def get_max_q_value_for_k_and_state(self, k, state, v_k_values):
        """
        return highest => v_i_action, v_i_value
        """
        if k <= 0:
            return None, 0

        filtered = v_k_values[(k, state)]

        if len(filtered) == 0:
            return None, 0

        return sorted(filtered.items(), key=lambda item: item[1], reverse=True)[0]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """

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


          we need to do 1 more iteration since a policy synthesized from values of depth k
          (which reflect the next k rewards) will actually reflect the next k+1 rewards (i.e. you return πk+1).
          Similarly, the Q-values will also reflect one more reward than the values (i.e. you return Qk+1).

        """
        "*** YOUR CODE HERE ***"
        values = util.Counter()

        for (i, s) in self.q_k_values:
            if i is self.iterations + 1:  # Qk+1(s, a)
                values[s] = self.q_k_values[(i, s)]

        q_values = values[state]

        if len(q_values) == 0:
            return 0

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

          we need to do 1 more iteration since a policy synthesized from values of depth k
          (which reflect the next k rewards) will actually reflect the next k+1 rewards (i.e. you return πk+1).
          Similarly, the Q-values will also reflect one more reward than the values (i.e. you return Qk+1).
        """
        "*** YOUR CODE HERE ***"
        values = util.Counter()

        for (i, s) in self.q_k_values:
            if i is self.iterations + 1:  # PIk+1(s)
                values[s] = self.q_k_values[(i, s)]

        q_values = values[state]

        if self.mdp.isTerminal(state):
            return None

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
