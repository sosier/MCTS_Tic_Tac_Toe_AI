"""
MONTE CARLO TREE SEARCH:

Apapted by Sean Osier from the Python 2.7 code at the Monte Carlo Tree Search
(MCTS) research hub (http://mcts.ai/code/python.html):
  1. Updated code to be Python 3 compatible
  2. To keep code succinct and focused, example GameStates code removed
"""

"""
ORIGINAL COMMENTS FROM MCTS.AI:

This is a very simple implementation of the UCT Monte Carlo Tree Search
algorithm in Python 2.7.
The function UCT(rootstate, itermax, verbose = False) is towards the bottom of
the code.
It aims to have the clearest and simplest possible code, and for the sake of
clarity, the code is orders of magnitude less efficient than it could be made,
particularly by using a state.GetRandomMove() or state.DoRandomRollout()
function.

Example GameState classes for Nim, OXO and Othello are included to give some
idea of how you can write your own GameState use UCT in your 2-player game.
Change the game to be played in the UCTPlayGame() function at the bottom of the
code.

Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK)
September 2012.

Licence is granted to freely use and distribute for any sensible/legal purpose
so long as this comment remains in any distributed code.

For more information about Monte Carlo Tree Search check out our web site at
www.mcts.ai
"""

from math import *
import random

# To keep code succinct and focused example GameStates code removed


class GameState:
    """
    "Skeleton" game state class

    A state of the game, i.e. the game board. These are the only functions
    which are absolutely necessary to implement UCT in any 2-player complete
    information deterministic zero-sum game, although they can be enhanced and
    made quicker, for example by using a GetRandomMove() function to generate a
    random move during rollout.

    By convention the players are numbered 1 and 2.
    """
    def __init__(self):
        # At the root pretend the player just moved is player 2
        # Thus, player 1 has the first move
        self.playerJustMoved = 2

    def Clone(self):
        """
        Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """
        Update a state by carrying out the given move.
        Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved

    def GetMoves(self):
        """
        Get all possible moves from this state.
        """
        pass

    def GetResult(self, playerjm):
        """
        Get the game result from the viewpoint of playerjm.
        """
        pass

    def __repr__(self):
        """
        Don't need this - but good style.
        """
        pass


class Node:
    """
    A node in the game tree. Note wins is always from the viewpoint of
    playerJustMoved. Crashes if state not specified.
    """
    def __init__(self, move=None, parent=None, state=None):
        # The move that got us to this node - "None" for the root node
        self.move = move

        # "None" for the root node
        self.parentNode = parent

        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.expectedOutcome = 0.5

        # Future child nodes
        self.untriedMoves = state.GetMoves()

        # The only part of the state that the Node needs later
        self.playerJustMoved = state.playerJustMoved

    def UCTSelectChild(self, UCTK=1):
        """
        Use the UCB1 formula to select a child node. Often a constant UCTK is
        applied so we have lambda c:
        c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits
        to vary the amount of exploration versus exploitation.
        """
        s = sorted(self.childNodes, key=lambda c:
                   c.wins/c.visits +
                   UCTK * sqrt(2*log(self.visits)/c.visits)
                   )[-1]
        return s

    def AddChild(self, m, s):
        """
        Remove m from untriedMoves and add a new child node for this move
        Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """
        Update this node - one additional visit and result additional wins.
        result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result
        if self.childNodes:
            self.expectedOutcome = 1 - min(
                self.childNodes, key=lambda c: c.expectedOutcome
                ).expectedOutcome
        else:
            self.expectedOutcome = result

    def __repr__(self):
        return "[Move:" + str(self.move) + \
            " Wins/Visits:" + str(self.wins) + "/" + str(self.visits) + \
            " E(X):" + str(self.expectedOutcome) + \
            "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent+1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, UCTK=1, verbose=False):
    """
    Conduct a UCT search for itermax iterations starting from rootstate.
    Return the best move from the rootstate.
    Assumes 2 alternating players (player 1 starts), with game results in the
    range [0.0, 1.0].
    """

    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        # while node is fully expanded and non-terminal
        while node.untriedMoves == [] and node.childNodes != []:
            node = node.UCTSelectChild(UCTK)
            state.DoMove(node.move)

        # Expand
        # if we can expand (i.e. state/node is non-terminal)
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree

        # Simulate - this can often be made orders of magnitude quicker using a
        # state.GetRandomMove() function
        while state.GetMoves() != []:  # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        # backpropagate from the expanded node and work back to the root node
        # state is terminal. Update node with result from POV of
        # node.playerJustMoved
        while node is not None:
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose):
        print(rootnode.TreeToString(0))
    else:
        print(rootnode.ChildrenToString())

    # Return the move that was most visited
    return sorted(rootnode.childNodes, key=lambda c: c.wins/c.visits)[-1].move


def UCTPlayGame(state):
    """
    Play a sample game between two UCT players where each player gets a
    different number of UCT iterations (= simulations = tree nodes).
    """
    while (state.GetMoves() != []):
        print(str(state))
        if state.playerJustMoved == 1:
            # play with values for itermax and verbose = True
            m = UCT(rootstate=state, itermax=100, verbose=False)
        else:
            m = UCT(rootstate=state, itermax=1000, verbose=False)
        print("Best Move: " + str(m) + "\n")
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
    else:
        print("Nobody wins!")

"""
##############################################################################
"""


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        # At the root pretend player just moved is p2 / p1 has the first move
        self.playerJustMoved = 2
        self.gameOver = False

        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and \
            self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        if self.GetResult(self.playerJustMoved) is not None:
            self.gameOver = True

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        if self.gameOver:
            return []
        else:
            return [i for i in range(9) if self.board[i] == 0]

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        for (x, y, z) in [(0, 1, 2),
                          (3, 4, 5),
                          (6, 7, 8),
                          (0, 3, 6),
                          (1, 4, 7),
                          (2, 5, 8),
                          (0, 4, 8),
                          (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z] != 0:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0

        if self.GetMoves() == []:
            return 0.5  # draw
        else:
            return None

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2:
                s += "\n"

        return s


if __name__ == "__main__":
    UCTPlayGame(OXOState())
