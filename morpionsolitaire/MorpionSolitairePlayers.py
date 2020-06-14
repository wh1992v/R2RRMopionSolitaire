import numpy as np
from MCTS import MCTS

class NNtBasedMCTSPlayer():
    def __init__(self, game, nnt, args, temp=1, percentile=0.68):
        self.game=game
        self.nnt=nnt
        self.args=args
        self.temp=temp
        self.percentile=percentile

    def play(self, board):
        mcts=MCTS(self.game, self.nnt, self.args, self.percentile)
        prob, validmoves=mcts.getActionProb(board, self.temp)
        action=np.random.choice(len(prob), p=prob)
        return validmoves[action]

class PureNNtPlayer():
    def __init__(self, game, nnt, args, temp=1):
        self.game=game
        self.nnt=nnt
        self.args=args
        self.temp=temp

    def play(self, board):
        canonicalBoard = self.game.getCanonicalForm(board)
        pi, v = self.nnt.predict(canonicalBoard)
        legalmoves = self.game.getValidMoves(board)
        validmoves=[0]*self.game.getActionSize()
        valids=[0]*self.game.getActionSize()
        if len(legalmoves)==0:
            validmoves=np.array(validmoves)
            valids[-1]=1
        for lm in legalmoves:
            validmoves[self.game.n*lm.point[0]+lm.point[1]]=lm
            valids[self.game.n*lm.point[0]+lm.point[1]]=1
        validmoves=np.array(validmoves)
        valids=np.array(valids)
        pi=pi*valids
        pi=[float(p/sum(pi)) for p in pi]
        if self.temp==0:
            tempmaxindex=pi.index(max(pi))
            pi=[p*0 for p in pi]
            pi[tempmaxindex]=1
        selectedindex = np.random.choice(len(pi), p=pi)
        return validmoves[selectedindex] 

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        #a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board)
        #for v in valids:
        ##   print(v,end="")
        #print()
        #for v in valids:
            #print(v,end = "")
        #print()
        if len(valids) != 0:
            #print(len(valids))
            a=np.random.randint(len(valids))
            #print(len(valids))
        else:
            print("No legal move")
            return None
            #assert(ValueError,"valid moves is None")
        return valids[a]

class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True:
            a = input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
