# multiAgents.py
# --------------
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
import random
from typing import List, Tuple

import util

from game import Agent
import graphSearchProblem
from pacman import GameState


class ReflexAgent(Agent):
    test_count = 0  # define static class variable

    ghost_penalization = 10000

    eaten_food_points = 1200

    all_food_eaten_points = 5000

    distance_coefficient = 1000

    pacman_distance_from_ghost_coefficient = 1

    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legal_moves = game_state.getLegalActions()  # {NORTH, SOUTH, WEST, EAST, STOP}

        print('\n###### new iteration ######\n')
        print('legalMoves = ', legal_moves)

        # Choose one of the best actions
        scores = [self.evaluationFunction(game_state, action) for action in legal_moves]
        print('scores to choose = ', scores)

        best_score = max(scores)
        print('bestScore = ', best_score)

        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        print('bestIndices = ', best_indices)

        chosenIndex = random.choice(best_indices)  # Pick randomly among the best
        print('chosenIndex = ', chosenIndex)

        print('chosen move = ', legal_moves[chosenIndex])

        "Add more of your code here if you want to"

        return legal_moves[chosenIndex]

    def evaluationFunction(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        # child_game_state = current_game_state.getPacmanNextState(action)  # GameState (layout ??)
        # newPos = child_game_state.getPacmanPosition()  # current pacman coordinates (3,1)
        # newFood = child_game_state.getFood()  # newFood = layout with food F = not food, T = food
        # newGhostStates = child_game_state.getGhostStates()  # [game.AgentState] // for each ghost
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  # [0]

        "*** YOUR CODE HERE ***"
        best_score = self.calculate_best_score(current_game_state, action)
        print("returning best score = ", best_score)

        return best_score['score']

    def calculate_best_score(self, game_state: GameState, action):
        print('\ncurrent Action = ', action)

        #  pacman
        next_game_state = game_state.getPacmanNextState(action)
        next_pacman_position = next_game_state.getPacmanPosition()
        print('\nnext pacman position = ', next_pacman_position)

        # ghost
        # next_ghost_position = tuple(map(lambda v: int(v), next_game_state.getGhostPosition(1)))
        # print('\nnext Ghost position = ', next_ghost_position)

        next_ghost_positions = next_game_state.getGhostPositions()
        print('\nnext Ghost positions = ', next_ghost_positions)

        next_food_layout = next_game_state.getFood()

        print('next_food_layout = \n')
        print(next_food_layout, '\n')

        food_points = ReflexAgent.eaten_food_points if next_game_state.getNumFood() < game_state.getNumFood() else 0

        ghost_penalization = ReflexAgent.ghost_penalization if self.pacman_will_die(next_pacman_position,
                                                                                    next_ghost_positions) else 0

        scores = []

        for x in range(next_food_layout.width):
            for y in range(next_food_layout.height):
                if self.has_food(next_food_layout, x, y):
                    pacman_distance_to_food = self.calculate_food_distance(game_state,
                                                                             pacman_position=next_pacman_position,
                                                                             food_position=(x, y))

                    distance_points = ReflexAgent.distance_coefficient / pacman_distance_to_food

                    score = distance_points + food_points - ghost_penalization
                    scores.append({'distance': pacman_distance_to_food, 'score': score,
                                   'foodAt': [x, y], 'action': action})

        if len(scores) == 0:
            print('\nall food has been eaten == 0\n')

            score = ReflexAgent.all_food_eaten_points - ghost_penalization

            scores.append({'distance': 0, 'score': score, 'foodAt': None})

        # best score first
        scores.sort(key=lambda s: s['score'], reverse=True)

        print('sorted scores = ', scores)

        # get the best score
        return scores[0]

    def calculate_food_distance(self, game_state, pacman_position, food_position):
        """
        problem = graphSearchProblem.PositionSearchProblem(game_state,
                                                           start=pacman_position,
                                                           goal=food_position,
                                                           warn=False, visualize=False)
        path_to_food = graphSearchProblem.aStarSearch(problem)
        distance = len(path_to_food)
        """
        distance = util.euclidean_distance(pacman_position, food_position)
        return distance

    # def pacman_will_die(self, next_pacman_position, next_ghost_positions: List[Tuple]):
    def pacman_will_die(self, next_pacman_position, next_ghost_positions):
        for next_ghost_position in next_ghost_positions:
            pacman_distance_from_ghost = util.euclidean_distance(next_pacman_position, next_ghost_position)
            if pacman_distance_from_ghost <= ReflexAgent.pacman_distance_from_ghost_coefficient:
                return True
        return False

    """
    def pacman_will_die(self, next_pacman_position, next_ghost_position):
        pacman_distance_from_ghost = util.euclidean_distance(next_pacman_position, next_ghost_position)
        return pacman_distance_from_ghost <= ReflexAgent.pacman_distance_from_ghost_coefficient
    """

    def has_food(self, food_layout, x, y):
        return food_layout[x][y]

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
