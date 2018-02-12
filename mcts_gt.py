from math import sqrt, log
import random
from queue import Queue
from copy import copy, deepcopy


class GameState(object):
    """
    "Skeleton" Game State class

    A state of the game, i.e. the game board. These are the only functions
    which are absolutely necessary to implement UCT in any 2-player complete
    information deterministic zero-sum game.

    By convention the players are numbered 1 and 2.
    """
    def __init__(self):
        # At the root, this is who has the first move
        self.whose_move = 1

    def clone(self):
        """
        Create a deep clone of this game state.
        """
        state = GameState()
        state.whose_move = self.whose_move
        return state

    def do_move(self, move):
        """
        Update a state by carrying out the given move.
        Must update whose_move.
        """
        self.whose_move = 3 - self.whose_move

    def get_moves(self):
        """
        Get all possible moves from this state.
        """
        pass

    def get_random_move(self):
        """
        Not required but can improve efficiency of the MCTS simulations
        """
        pass

    def get_result(self):
        """
        Get the game result for all players.
        """
        return {1: 1,
                2: 0}

    def to_name(self):
        """
        Return a string based representation of current state
        """
        # Example for Tic-Tac-Toe
        return "000-100-200"

    def __repr__(self):
        """
        Don't need this - but good style.
        """
        pass


class GameTree(object):
    def __init__(self, starting_tree=None, root=None,
                 early_end_possible=True):
        if starting_tree:
            self.tree = starting_tree
        else:
            self.tree = {}

        self.root = None
        self.early_end_possible = early_end_possible

    def create_node(self, state, parent=None):
        """
        Create game state node
        """
        node = copy(state.to_name())

        if node in self.tree:  # If node already exists
            assert self.tree[node]["whose_move"] == copy(state.whose_move)

            if parent:
                self.tree[node]["parents"].add(parent)
        else:
            self.tree[node] = {}

            self.tree[node]["whose_move"] = copy(state.whose_move)

            if parent:
                self.tree[node]["parents"] = set([parent])
            else:  # None for root
                self.tree[node]["parents"] = set()

            self.tree[node]["checked_for_terminal_moves"] = \
                not self.early_end_possible

            self.tree[node]["tried_moves"] = {}
            self.tree[node]["untried_moves"] = copy(state.get_moves())

            self.tree[node]["wins"] = 0
            self.tree[node]["visits"] = 0

            self.tree[node]["EVs"] = {}  # Expected values

        return node

    def select_child(self, parent, how="EV_UCB1", K=1):
        selection_methods = set(["UCB1", "EV_UCB1"])

        assert how in selection_methods

        if how == "UCB1":
            selected_child, move = \
                self.UCB1_select_child(parent=parent, UCTK=K)
        elif how == "EV_UCB1":
            selected_child, move = \
                self.EV_UCB1_select_child(parent=parent, UCTK=K)

        return selected_child, move

    def UCB1_select_child(self, parent, UCTK=1):
        """
        Use the UCB1 formula to select a child node.
        Often a constant UCTK is applied to vary the amount of exploration
        versus exploitation. So we have lambda c:
            child_wins/child_visits +
            constant * sqrt(2*log(parent_visits)/child_visits
        """
        parent = self.tree[parent]
        children = [(name, node) for name, node in self.tree.items()
                    if name in parent["tried_moves"].values()]

        children = sorted(children, key=lambda c:
                          c[1]["wins"]/c[1]["visits"] +
                          UCTK * sqrt(2*log(parent["visits"]) / c[1]["visits"])
                          )
        selected_child = children[-1][0]

        move = {child: move for move, child in parent["tried_moves"].items()}
        move = move[selected_child]

        return selected_child, move

    def EV_UCB1_select_child(self, parent, UCTK=1):
        """
        Use an Expected Value (EV) adaptation of the UCB1 formula to select
        child nodes.
        Often a constant UCTK is applied to vary the amount of exploration
        versus exploitation. So we have lambda c:
            EV[whose_move] +
            constant * sqrt(2*log(parent_visits)/child_visits
        """
        parent = self.tree[parent]
        children = [(name, node) for name, node in self.tree.items()
                    if name in parent["tried_moves"].values()]

        children = sorted(children, key=lambda c:
                          c[1]["EVs"][parent["whose_move"]] +
                          UCTK * sqrt(2*log(parent["visits"]) / c[1]["visits"])
                          )
        selected_child = children[-1][0]

        move = {child: move for move, child in parent["tried_moves"].items()}
        move = move[selected_child]

        return selected_child, move

    def add_child(self, parent, move, state):
        """
        Remove move from untried_moves and add a new child node for this move
        Return the added child node
        """
        child = self.create_node(state=state, parent=parent)
        parent = self.tree[parent]

        if move in parent["untried_moves"]:
            parent["untried_moves"].remove(move)
        parent["tried_moves"][move] = child

        return child

    def check_for_terminal_move(self, node, state):
        assert node == state.to_name()
        node = self.tree[node]

        terminal_moves = set()

        for move in node["untried_moves"]:
            state_copy = deepcopy(state)
            state_copy.do_move(move)

            if state_copy.game_over:
                terminal_moves.add(move)

        node["checked_for_terminal_moves"] = True
        if terminal_moves:
            node["untried_moves"] = copy(terminal_moves)

    def get_random_move(self, node):
        node = self.tree[node]
        if node["untried_moves"]:
            return random.sample(node["untried_moves"], 1)[0]
        elif node["tried_moves"]:
            return random.sample(node["tried_moves"].keys(), 1)[0]
        else:
            return None

    def update_node(self, node, result):
        """
        Update this node - one additional visit and result additional wins.
        result must be from the viewpoint of playerJustmoved.
        """
        node = self.tree[node]
        if node["parents"]:
            parent = random.sample(node["parents"], 1)[0]
            whose_move = self.tree[parent]["whose_move"]
        else:
            whose_move = node["whose_move"]

        # Standard UCB1 updating
        node["visits"] += 1
        node["wins"] += result[whose_move]

        # Game theory updating of expected value / outcome
        if node["tried_moves"]:
            children_EVs = [child["EVs"] for name, child
                            in self.tree.items()
                            if name in node["tried_moves"].values()]

            node["EVs"] = max(children_EVs, key=lambda e:
                              e[node["whose_move"]])
        else:
            node["EVs"] = result

    def print_moves_analysis(self, node):
        for move, child in sorted(self.tree[node]["tried_moves"].items()):
            child = self.tree[child]
            print("Move: " + str(move) +
                  "    W/V: " + str(child["wins"]) + "/" +
                  str(child["visits"]) +
                  "    EV: " + str(child["EVs"]))

    def to_dict(self):
        return self.tree


def MCTS(root_state, iterations, method="EV_UCB1", K=1,
         starting_tree=None, root_node=None, early_end_possible=True,
         verbose=False):
    """
    - Conduct a MCTS for "iterations" iterations starting from root_state.
    - Return the best move from the root_state.
    - Assumes 2 alternating players (player 1 starts), with game results in the
    range [0.0, 1.0].
    """
    if starting_tree:
        assert root_node in starting_tree
        GT = GameTree(starting_tree=starting_tree, root=root_node,
                      early_end_possible=early_end_possible)  # GameTree
    else:
        assert root_node is None
        GT = GameTree(early_end_possible=early_end_possible)  # GameTree
        root_node = GT.create_node(state=root_state)

    for i in range(iterations):
        node = root_node
        state = deepcopy(root_state)

        # Select
        # while node is fully expanded and non-terminal
        while GT.tree[node]["tried_moves"] != {} and \
                not GT.tree[node]["untried_moves"]:
            node, move = GT.select_child(parent=node, how=method, K=K)
            state.do_move(move)

        # Expand
        # if we can expand (i.e. state/node is non-terminal)
        if GT.tree[node]["untried_moves"]:
            if not GT.tree[node]["checked_for_terminal_moves"]:
                GT.check_for_terminal_move(node, state)

            move = random.sample(GT.tree[node]["untried_moves"], 1)[0]
            state.do_move(move)

            # Add child and descend tree
            node = GT.add_child(parent=node, move=move, state=state)
            if not GT.tree[node]["checked_for_terminal_moves"]:
                GT.check_for_terminal_move(node, state)

        # Simulate
        while not state.game_over:
            move = GT.get_random_move(node)
            state.do_move(move)

            # Add child and descend tree
            # NOT required but this should improve AI game performance at the
            # cost of additional memory (not appropriate for larger games)
            node = GT.add_child(parent=node, move=move, state=state)

        # Backpropagate
        # backpropagate from the expanded node and work back to the root node
        # state is terminal. Update node with results
        result = state.get_result()
        to_update = Queue()
        to_update.put(node)

        while to_update.qsize():
            node = to_update.get()
            GT.update_node(node, result)
            for parent in GT.tree[node]["parents"]:
                to_update.put(parent)

    # Return the move with highest expected value
    children = [(name, child) for name, child in GT.tree.items()
                if name in GT.tree[root_node]["tried_moves"].values()]

    # Get best child by sorting by expected value then by win/visit ratio
    best_child = sorted(children,
                        key=lambda c: (c[1]["EVs"][root_state.whose_move],
                                       c[1]["wins"]/c[1]["visits"])
                        )[-1]  # Last child in sorted list is best
    best_child = best_child[0]  # Get just the child name

    move = {child: move for move, child in
            GT.tree[root_node]["tried_moves"].items()}
    move = move[best_child]

    if verbose:
        GT.print_moves_analysis(root_node)

    return move


def MCTS_play_game(state, player1_iterations=500, player2_iterations=100,
                   verbose=False):
    """
    Play a game between two MCTS players where each player gets a
    different number of MCTS iterations (simulations / tree nodes).
    """
    while state.get_moves():
        if verbose:
            print(str(state))

        if state.whose_move == 1:
            move = MCTS(root_state=state, iterations=player1_iterations,
                        verbose=verbose)
        else:
            move = MCTS(root_state=state, iterations=player2_iterations,
                        verbose=verbose)

        state.do_move(move)

    if verbose:
        print(str(state))

    if state.get_result()[1] == 1.0:
        result_message = "Player 1 Wins!"
        result = 1
    elif state.get_result()[2] == 1.0:
        result_message = "Player 2 Wins!"
        result = 2
    else:
        result_message = "Tie!"
        result = 0

    if verbose:
        print(result_message)

    return result


class TicTacToe(object):
    """
    A state of the game Tic-Tac-Toe:
     - Squares in the board are in this arrangement
        0 1 2
        3 4 5
        6 7 8
     - 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
     - Get 3 in a row (vertically, horizontally, or diagonally to win)
     - If neither player wins it's a draw
    """
    def __init__(self):
        self.game_over = False
        self.whose_move = 1

        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        available_moves = {i for i, val in enumerate(self.board) if val == 0}
        self.available_moves = copy(available_moves)

    def clone(self):
        """
        Create a deep clone of this game state.
        """
        state = TicTacToe()

        state.game_over = self.game_over
        state.whose_move = self.whose_move
        state.board = self.board[:]
        state.available_moves = self.available_moves.copy()
        return state

    def do_move(self, move):
        """
        Update a state by carrying out the given move.
        Must update whose_move.
        """
        assert move >= 0 and move <= 8 and move == int(move) and \
            self.board[move] == 0

        self.board[move] = self.whose_move
        self.available_moves.remove(move)

        if self.get_result():
            self.game_over = True
            self.whose_move = None
        else:
            self.whose_move = 3 - self.whose_move

    def get_moves(self):
        """
        Get all possible moves from this state.
        """
        if self.game_over:
            return {}
        else:
            return self.available_moves

    def get_random_move(self):
        """
        Not required but can improve efficiency of the MCTS simulations
        """
        if self.get_moves():
            return random.sample(self.get_moves(), 1)[0]
        else:
            return None

    def get_result(self):
        """
        Get the game result for all players.
        """
        result = {}

        for (x, y, z) in [(0, 1, 2),  # Top row
                          (3, 4, 5),  # Middle row
                          (6, 7, 8),  # Bottom row
                          (0, 3, 6),  # Left column
                          (1, 4, 7),  # Middle column
                          (2, 5, 8),  # Right column
                          (0, 4, 8),  # Diagonal from Top Left
                          (2, 4, 6)]:  # Diagonal from Top Right
            if self.board[x] == self.board[y] == self.board[z] != 0:
                self.game_over = True

                if self.board[x] == 1:
                    # Player 1 Wins
                    result = {1: 1.0,
                              2: 0.0}
                    return result
                else:
                    # Player 2 Wins
                    result = {1: 0.0,
                              2: 1.0}
                    return result

        if not self.get_moves():
            # Draw
            result = {1: 0.5,
                      2: 0.5}

        return result

    def to_name(self):
        """
        Return a string based representation of current state / board
        """
        return "".join([str(x) for x in self.board])

    def __repr__(self):
        s = ""
        for i, val in enumerate(self.board):
            s += " " + ".XO"[val] + " "
            if i % 3 == 2:
                s += "\n"

        return s


if __name__ == "__main__":
    MCTS_play_game(TicTacToe(), verbose=True)
