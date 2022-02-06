"""
Tic Tac Toe Player
"""

import math
import random
import copy
from typing import Union

X = "X"
O = "O"
EMPTY = None


def track():
    Tracker.count += 1


class Tracker:
    count = 0


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board) -> Union[X, O, None]:
    """
    Returns player who has the next turn on a board.

    The player function should take a board state as input, and return which player’s turn it is (either X or O).
    - In the initial game state, X gets the first move. Subsequently, the player alternates with each additional move.
    - Any return value is acceptable if a terminal board is provided as input (i.e., the game is already over).

    """
    if board == initial_state():
        return X

    # calculate the count for each player
    x_count = 0
    o_count = 0

    for row in board:
        for column in row:
            if column == X:
                x_count += 1
            if column == O:
                o_count += 1

    return O if o_count < x_count else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.

    The actions function should return a set of all of the possible actions that can be taken on a given board.

    - Each action should be represented as a tuple (i, j) where i corresponds to the row of the move (0, 1, or 2) and j corresponds to 
      which cell in the row corresponds to the move (also 0, 1, or 2).
    - Possible moves are any cells on the board that do not already have an X or an O in them.
    - Any return value is acceptable if a terminal board is provided as input.
    """
    possible_actions = []

    for row in range(len(board)):
        for column in range(len(board[row])):
            if board[row][column] == EMPTY:
                possible_actions.append((row, column))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.

    The result function takes a board and an action as input, and should return a new board state, without modifying the original board.

    - If action is not a valid action for the board, your program should raise an exception.
    - The returned board state should be the board that would result from taking the original input board, and letting the player whose turn it is make their move at 
      the cell indicated by the input action.
    - Importantly, the original board should be left unmodified: since Minimax will ultimately require considering many different board states during its computation. 
      This means that simply updating a cell in board itself is not a correct implementation of the result function. You’ll likely want to make a deep copy of the board first 
      before making any changes.
    """
    current_player = player(board)
    board_copy = copy.deepcopy(board)
    row, column = action
    board_copy[row][column] = current_player
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.

    The winner function should accept a board as input, and return the winner of the board if there is one.

    - If the X player has won the game, your function should return X. If the O player has won the game, your function should return O.
    - One can win the game with three of their moves in a row horizontally, vertically, or diagonally.
    - You may assume that there will be at most one winner (that is, no board will ever have both players with three-in-a-row, since that would be an invalid board state).
    - If there is no winner of the game (either because the game is in progress, or because it ended in a tie), the function should return None.
    """

    # too lazy to come up with a genius mathematical function to solve this :)

    if board[0][0] == board[0][1] == board[0][2] != EMPTY:
        return board[0][0]
    if board[1][0] == board[1][1] == board[1][2] != EMPTY:
        return board[1][0]
    if board[2][0] == board[2][1] == board[2][2] != EMPTY:
        return board[2][0]
    if board[0][0] == board[1][0] == board[2][0] != EMPTY:
        return board[0][0]
    if board[0][1] == board[1][1] == board[2][1] != EMPTY:
        return board[0][1]
    if board[0][2] == board[1][2] == board[2][2] != EMPTY:
        return board[0][2]
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

        # no winner
    return None


def terminal(board) -> bool:
    """
    Returns True if game is over, False otherwise.

    The terminal function should accept a board as input, and return a boolean value indicating whether the game is over.

    - If the game is over, either because someone has won the game or because all cells have been filled without anyone winning, the function should return True.
    - Otherwise, the function should return False if the game is still in progress.
    """
    if winner(board) is not None:
        return True

    # Check if there's a cell EMPTY, otherwise the game is over
    for row in board:
        for column in row:
            if column == EMPTY:
                return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.


    The utility function should accept a terminal board as input and output the utility of the board.
    - If X has won the game, the utility is 1. If O has won the game, the utility is -1. If the game has ended in a tie, the utility is 0.
    - You may assume utility will only be called on a board if terminal(board) is True
    """
    who_is_winner = winner(board)

    if who_is_winner == X:
        return 1
    if who_is_winner == O:
        return -1

    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.

    The minimax function should take a board as input, and return the optimal move for the player to move on that board.
    - The move returned should be the optimal action (i, j) that is one of the allowable actions on the board. If multiple moves are equally optimal, any of those moves is acceptable.
    - If the board is a terminal board, the minimax function should return None.
    """
    # TODO: return (i,j)

    # alpha - beta pruning can be disabled by setting these 2 variables = None
    alpha = -math.inf  # worst case for the maximizer
    beta = math.inf  # worst case for the minimizer

    if terminal(board):
        print("tracker = " + str(Tracker.count))
        return None

    if player(board) == X:
        # return max_value(board)  # action in actions(board_state)
        legal_moves = actions(board)

        # min_value() here since i'm calling result(board, action), which means that i'm moving down in the tree by one
        # were the next node will be a minimizer
        scores = [min_value(result(board, action), alpha, beta) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    if player(board) == O:
        legal_moves = actions(board)

        # max_value() here since i'm calling result(board, action), which means that i'm moving down in the tree by one
        # were the next node will be a maximizer
        scores = [max_value(result(board, action), alpha, beta) for action in legal_moves]
        best_score = min(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]


def max_value(board_state, alpha=None, beta=None):
    track()

    if terminal(board_state):
        return utility(board_state)

    v = -math.inf

    for action in actions(board_state):

        successor = result(board_state, action)
        v = max(v, min_value(successor, alpha, beta))

        if is_alpha_beta_pruning_enabled(alpha, beta):
            if v >= beta:
                return v
            alpha = max(alpha, v)

    return v


def min_value(board_state, alpha=None, beta=None):
    track()

    if terminal(board_state):
        return utility(board_state)

    v = math.inf

    for action in actions(board_state):
        successor = result(board_state, action)
        v = min(v, max_value(successor, alpha, beta))

        if is_alpha_beta_pruning_enabled(alpha, beta):
            if v <= alpha:
                return v
            beta = min(beta, v)

    return v


def is_alpha_beta_pruning_enabled(alpha, beta) -> bool:
    return (alpha is not None) and (beta is not None)

# max tic tac toe 255168 total possible tic tac toe games
