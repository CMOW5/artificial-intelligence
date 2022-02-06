import random
import math


"""
If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.
"""
epsilon = 0.1

a = dict()
a[(0, 0, 0, 2), (3, 2)] = -1
a[(0, 0, 0, 2), (1, 2)] = 0
a[(0, 0, 0, 2), (1, 4)] = 1
a[(0, 0, 0, 3), (1, 3)] = 5

highest_value = -math.inf
highest_value_action = None

m_state = (0, 0, 0, 2)

m = set(map(lambda k: k[0], a.keys()))

if (0, 0, 0, 2) in m:
    print('YESSS')

print('m = ', m)

for key, value in a.items():
    q_state, q_action = key
    if q_state == m_state and highest_value < value:
        highest_value = value
        highest_value_action = q_action

all_actions = {action for (state, action), value in a.items() if state == m_state}

chosen_action = random.choice(list(all_actions))

chosen_action = random.choices([highest_value_action, chosen_action], [0.9, 0.1])

print('all_actions = ', all_actions)
print('highest_value_action = ', highest_value_action)
print('chosen_action = ', chosen_action)

