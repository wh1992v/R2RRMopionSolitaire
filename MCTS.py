import math
import numpy as np
EPS = 1e-8
import copy

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, percentile=0.35):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves.point for board s
        self.Vm={}          # stores game.getValidMoves for board s
        #self.num=0
        self.percentilevalue=percentile

    def getActionProb(self, board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        mctsboard=copy.deepcopy(board)
        for i in range(self.args.numMCTSSims):
            #print("Simulation:", i)
            self.search(mctsboard)

        s = self.game.stringRepresentation(self.game.getCanonicalForm(board))
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        #mctslegalmoves=self.game.getValidMoves(board)
        #flag=False
        #for mclm in mctslegalmoves:
            #flag=False
            #for j in range(self.game.getActionSize()):
                #if counts[j] !=0:
                   #if mclm.point[0]*self.game.n+mclm.point[1]==j:
                        #flag=True
            #if flag==False:
                #print("legal move", mclm.point,mclm.line, "doesn't show up in counts")

        #flag=False
        #for k in range(self.game.getActionSize()):
            #flag=False
            #if counts[k]!=0:
                #for mclm in mctslegalmoves:
                    #if mclm.point[0]*self.game.n+mclm.point[1]==k:
                        #flag=True
                #if flag==False:
                    #print("counts index", k, "doesn't show up in legalmoves")

        #print(counts)
        #print(self.Ns[s])
        #print(board.pieces)
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs, self.Vm[s]

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        
        #action=np.random.choince(len(probs), pi=probs)
        return probs, self.Vm[s]


    def search(self, board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the value of the current canonicalBoard
        """
        #self.num+=1
        #print(self.num)
        #board=copy.deepcopy(mctsboard)
        #board=self.game.getInitBoard()
        #board.pieces=np.array(mctsboard.pieces.copy())
        #board.allLines=mctsboard.allLines.copy()
        #board.performedmoves=mctsboard.performedmoves.copy()

        s = self.game.stringRepresentation(self.game.getCanonicalForm(board))
        if s not in self.Es:
            self.Es[s] = float(self.game.getGameEnded(board)/121)
            #print("GetScore", self.Es[s])
        if self.Es[s]!=0:
            # terminal node
            #print("terminal node visited!")
            self.Es[s]= -1 if self.Es[s]<self.percentilevalue else 1
            if self.Es[s]==1:
                print("New Record:", len(board.performedmoves), end="")
                for move in board.performedmoves:
                    print("|", move.point, end="")
                    print(move.line, end="")
                print("|")
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(self.game.getCanonicalForm(board))
            legalmoves = self.game.getValidMoves(board)
            valids=[0]*self.game.getActionSize()
            validmoves=[0]*self.game.getActionSize()
            #validslines=[0]*self.game.getActionSize()
            if len(legalmoves)==0:
                valids[-1]=1
                valids=np.array(valids)
                validmoves=np.array(validmoves)
                #validslines=np.array(validslines)
            for lm in legalmoves:
                if valids[self.game.n*lm.point[0]+lm.point[1]]==0:
                    valids[self.game.n*lm.point[0]+lm.point[1]]=1
                    validmoves[self.game.n*lm.point[0]+lm.point[1]]=lm
                #validslines[self.game.n*lm.point[0]+lm.point[1]]=lm.line
            valids=np.array(valids)
            #validslines=np.array(validslines)
            validmoves=np.array(validmoves)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            #print(self.Ps[s])
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            #self.Ls[s]=validslines
            self.Vm[s]=validmoves
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        validmoves=self.Vm[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        bestmove = validmoves[a]
        bb=copy.deepcopy(board)
        next_s= self.game.getNextState(bb, bestmove)
        #next_s = self.game.getCanonicalForm(next_s)
        v = self.search(next_s)
        
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        #print(self.Ns[s])
        
        return v
