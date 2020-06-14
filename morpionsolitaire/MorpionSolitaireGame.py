from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .MorpionSolitaireLogic import *
import numpy as np
#import matplotlib.pyplot as plt
import copy

class MorpionSolitaireGame(Game):
    def __init__(self, n):
        self.n = n

    def getInitBoard(self,initallline=False):
        # return initial board (numpy board)
        b = Board(self.n, initallline)
        return b

    def getInitMove(self,point,line):
        return Move(point,line)

    def getInitLine(self,p1,p2):
        return Line(p1,p2)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n+1

    def getNextState(self, board, move):
        # if player takes action on board, return next (board)
        # action must be a valid move
        #b=copy.deepcopy(board)
        newboard=board.playMove(move)
        return newboard

    def getValidMoves(self, board):
        #b=copy.deepcopy(board)
        validmoves=board.getPossibleMoves()
        if validmoves is None:
            return None
        else:
            return np.array(validmoves)

    def getGameEnded(self, board):
        # return 0 if not ended, 1 ended
        #b = copy.deepcopy(board)
        if len(self.getValidMoves(board))!=0:
            return 0
        else:
            return self.getScore(board)

    def getCanonicalForm(self, board):
        return np.array(board.pieces)

    def getSymmetries(self, board, pi):
        # mirror, rotational
        #board=copy.deepcopy(b).pieces
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # numpy array (canonical board)
        return board.tostring()

    def getScore(self, b):
        return b.getScore()


def display(board):
    n=board.pieces.shape[0]
    b=Board(n)
    b.performedmoves=board.performedmoves.copy()
    initdot_x=[]
    initdot_y=[]
    for x in range(n):
         for y in range(n):
            if b[x][y]==1:
                initdot_x.append(x)
                initdot_y.append(y)

    #plt.grid(linestyle='--', linewidth=0.5)
    #plt.xticks(np.arange(0,n, step=1))
    #plt.yticks(np.arange(0, n, step=1))
    #plt.scatter(initdot_x,initdot_y)
    for move in board.performedmoves:
        movedots_x=[]
        movedots_y=[]
        movedots_x.append(move.point[0])
        movedots_y.append(move.point[1])
        #plt.scatter(movedots_x,movedots_y, color='', marker='o', edgecolor='g')
        #move_line.append(move.line)
        theline_x=[]
        theline_y=[]
        theline_x.append(move.line.p1[0])
        theline_x.append(move.line.p2[0])
        theline_y.append(move.line.p1[1])
        theline_y.append(move.line.p2[1])
        #plt.plot(theline_x,theline_y,"r")
        #plt.pause(0.1)
        #plt.show()
    #plt.show()

