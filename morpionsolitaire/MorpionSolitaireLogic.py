'''
Author: Hui Wang
Date: Apr 15, 2019.
Board class.
Board data:
  1=cross, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

import numpy as np
from collections import Counter
import datetime as dt

class Move:
    def __init__(self, point, line):
        self.point = point
        self.line = line

    def __str__(self):
        return str(self.point)

class Line:
    """
    Line length is always 4 (ie 4 intervals between 5 points)
    Line instance is carrying a list of lines that it overlaps that is initialized with solitaire game
    p1, p2  are tuples (x,y)
     """
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        if self.p1 > self.p2: #switching points if p1 > p2 (for sorting reasons)
            p2 = self.p1
            self.p1 = self.p2
            self.p2 = p2
        self.dir= self.getDirection() # getting the direction of the line (frome beginning to the end)
        self.origin = self.getOrigin()
        self.points = [(self.p1[0] + x*self.dir[0],self.p1[1] +x*self.dir[1]) for x in range(5)] # list of coords of points for the line
        self.overlaps = []

    def isValid(self):
        """
        A valid line is a line that is len 4 and whiwh the direction is valid
        :return: boolean
        """
        #Check length at least x2-x1= 4 or y2-y1=4
        xgap = abs(self.p1[0]-self.p2[0])
        ygap = abs(self.p1[1]-self.p2[1])
        tot= xgap+ygap
        return (xgap == 4 or ygap == 4) and (tot == 4 or tot == 8)

    def getDirection(self):
        """
        Return the couple of integer that represents a direction from startpoint to endpoint of line
        :return: (1,0) or (0,1) etc.. or None
        """
        return (int((self.p2[0]-self.p1[0])/4), int((self.p2[1]-self.p1[1])/4))

    def getOrigin(self):
        """
        :return: tuple (0,N) or (1,N) where 0 is x axe and 1 is y axe and N the origin on the axe
        """
        if self.dir==(0,1):
            return (0,self.p1[0])
        if self.dir == (1,0):
            return (1,self.p1[1])
        if self.dir == (1,1):
            dif = self.p1[0]-self.p1[1]
            return (0 if dif >= 0 else 1, abs(dif))
        if self.dir == (1,-1):
            return (1, self.p1[0]+self.p1[1]) #only on y axe

    def isOverlapping(self, otherline):
        """
        Checks whether 2 lines ovelaps
        :param Line otherline:
        :return:boolean
        """
        if self.dir != otherline.dir:
            return False
        if otherline in self.overlaps or self in otherline.overlaps:
            return True
        intersect = set(self.points).intersection(set(otherline.points))
        if len(intersect) >= 1: ### >1 is 5T version >=1 is 5D version
            return True
        return False


    def __str__(self):
        return str(sorted([self.p1,self.p2])).replace(' ','')

    def equals(self, other):
        return self.p1 == other.p1 and self.p2 == other.p2

    def isAdjacentTo(self,otherline):
        if self.dir != otherline.dir:
            return False
        return self.p1 == otherline.p2 or self.p2 == otherline.p1

    def getGapWith(self, otherline):
        """
        :param Line otherline:
        :return: int
        """
        if self.dir != otherline.dir or self.origin != otherline.origin:
            return -1
        else:
            divisor = sum([abs(i) for i in self.dir])
            gap = abs((self.p1[0]-otherline.p1[0])*self.dir[0] + (self.p1[1]-otherline.p1[1])*self.dir[1])//divisor-4
            return gap


class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    #__directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    #cross = 1

    def __init__(self, n, initallline=False):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n
        # Set up the initial 36 pieces.
        self.pieces[int(self.n/2)-5][int(self.n/2)-1] = 1
        self.pieces[int(self.n/2)-5][int(self.n/2)-2] = 1
        self.pieces[int(self.n/2-5)][int(self.n/2)] = 1
        self.pieces[int(self.n/2-5)][int(self.n/2+1)] = 1
        self.pieces[int(self.n/2)-4][int(self.n/2)-2] = 1
        self.pieces[int(self.n/2)-3][int(self.n/2)-2] = 1
        self.pieces[int(self.n/2)-2][int(self.n/2)-2] = 1
        self.pieces[int(self.n/2)-2][int(self.n/2)-3] = 1
        self.pieces[int(self.n/2)-2][int(self.n/2)-4] = 1
        self.pieces[int(self.n/2)-2][int(self.n/2)-5] = 1
        self.pieces[int(self.n/2)-1][int(self.n/2)-5] = 1
        self.pieces[int(self.n/2)][int(self.n/2)-5] = 1
        self.pieces[int(self.n/2)+1][int(self.n/2)-5] = 1
        self.pieces[int(self.n/2)+1][int(self.n/2)-4] = 1
        self.pieces[int(self.n/2)+1][int(self.n/2)-3] = 1
        self.pieces[int(self.n/2)+1][int(self.n/2)-2] = 1
        self.pieces[int(self.n/2)+2][int(self.n/2)-2] = 1
        self.pieces[int(self.n/2)+3][int(self.n/2)-2] = 1
        self.pieces[int(self.n/2)+4][int(self.n/2)-2] = 1
        self.pieces[int(self.n/2)+4][int(self.n/2)-1] = 1
        self.pieces[int(self.n/2)+4][int(self.n/2)] = 1
        self.pieces[int(self.n/2)+4][int(self.n/2)+1] = 1
        self.pieces[int(self.n/2)+3][int(self.n/2)+1] = 1
        self.pieces[int(self.n/2)+2][int(self.n/2)+1] = 1
        self.pieces[int(self.n/2)+1][int(self.n/2)+1] = 1
        self.pieces[int(self.n/2)+1][int(self.n/2)+2] = 1
        self.pieces[int(self.n/2)+1][int(self.n/2)+3] = 1
        self.pieces[int(self.n/2)+1][int(self.n/2)+4] = 1
        self.pieces[int(self.n/2)][int(self.n/2)+4] = 1
        self.pieces[int(self.n/2)-1][int(self.n/2)+4] = 1
        self.pieces[int(self.n/2)-2][int(self.n/2)+4] = 1
        self.pieces[int(self.n/2)-2][int(self.n/2)+3] = 1
        self.pieces[int(self.n/2)-2][int(self.n/2)+2] = 1
        self.pieces[int(self.n/2)-2][int(self.n/2)+1] = 1
        self.pieces[int(self.n/2)-3][int(self.n/2)+1] = 1
        self.pieces[int(self.n/2)-4][int(self.n/2)+1] = 1
        self.pieces=np.array(self.pieces)
        self.moves = []
        self.possiblemoves = {} # hash of game with possible moves association
        self.performedmoves=[]
        self.width = len(self.pieces)
        self.height = len(self.pieces[0])
        self.allLines =[]
        if initallline:
           self.allLines=self.getAllPossibleLines()#stocke toutes les définitiones de lignes possibles
        #self.getPossibleMoves()

        #print(self.pieces.shape)
    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def playMove(self, move):
        """
        Play a move without verifying
        :param move:
        :return: move
        """
        #self.moves.append(move)
        self.pieces[move.point[0],move.point[1]] = 1
        self.performedmoves.append(move)
        return self
        #self.hash = self.getHash()


    def checkMoveFromLine(self, line):
        if line.isValid():
            return self.getValidMoveFromLine(line)
        return None

    def getValidMoveFromLine(self, line):
        """
        check if move (line instance) is valid in the board.
        Valid means: the line is 4 length crossed 4 points and one free space on the board
        the move will be one point and a line
        :param line: line (2 points) tuple
        :return: move (valid one) or None
        """
        if not line.isValid(): #line must be OK at first
            return None
        linePattern = [self.pieces[p[0], p[1]] for p in line.points]

        #one and only one coord must be empty DONE optimising with Counter
        #print(Counter(linePattern)[0])
        if Counter(linePattern)[0] != 1:
            return None
        for m in self.performedmoves:
            #print("here")
            for tl in line.overlaps:
                if m.line.equals(tl):
                    return None
        return Move(line.points[linePattern.index(0)],line)
        #First we retrieve all coords on line

    def getAllPossibleLines(self):
        """
        :return: list of all lines - initialisation of overlapping lines
        That done we don't have to instanciate new lines -
        """
        lines = []
        lenX = len(self.pieces)
        lenY = len(self.pieces[0])
        for j in range(lenY):
            for i in range(lenX):
                if j+4 < lenY:
                    lines.append(Line((i,j),(i,j+4)))
                if j+4 < lenY and i+4 < lenX:
                    lines.append(Line((i,j),(i+4,j+4)))
                if i+4 < lenX:
                    lines.append(Line((i,j),(i+4,j)))
                if j >=4 and i+4 < lenX:
                    lines.append(Line((i,j),(i+4,j-4)))
        for line in lines:
            for l2 in lines: #cross join check
                if line.isOverlapping(l2):
                    line.overlaps.append(l2) #add this element to the line

        #print('end of init lines: '+ str(dt.datetime.now()))
        return lines # make dictionary


    def getPossibleMoves(self):
        moves = []
        # for each line possible of the board, check if it is a possible move.
        # TODO optimize without allline stuff ??
        # allLines should be allPlayableLines for the current instance of the game
        for l in self.allLines:
            m = self.getValidMoveFromLine(l)
            if m:
                moves.append(m)
        return moves

    def getScore(self):
        return len(self.performedmoves)

