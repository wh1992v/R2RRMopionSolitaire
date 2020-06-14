from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from morpionsolitaire.MorpionSolitaireGame import display
from morpionsolitaire.MorpionSolitairePlayers import *
#import pandas as pd

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        #self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        self.recentrewards=[]
        self.maxLenofRewardBuffer=200
        self.upperbound=121
        self.rewardpercentage=0.75
        self.currecord=float(82/self.upperbound)
        self.recentrewards.append(float(41/self.upperbound))
        self.r_percentile=0
    def executeEpisode(self):
        """
        This function executes one episode of self-play.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard(initallline=True)
        shapedreward=1
        episodeStep = 0
        while True:
            episodeStep += 1
            #print(episodeStep)
            canonicalBoard = self.game.getCanonicalForm(board)
            temp = int(episodeStep < self.args.tempThreshold)
            pi, v = self.nnet.predict(canonicalBoard)
            #print(pi,v)
            legalmoves = self.game.getValidMoves(board)
            #print(canonicalBoard)
            #print(performedmoves)
            #print(len(legalmoves))
            #action=legalmoves[0]
            
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
            if temp==0:
                tempmaxindex=pi.index(max(pi))
                pi=[p*0 for p in pi]
                pi[tempmaxindex]=1

            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, p, None])

            selectedindex = np.random.choice(len(pi), p=pi)
            #print(validmoves)

            action = validmoves[selectedindex]

            #print(action)
            board = self.game.getNextState(board, action)
            r = self.game.getGameEnded(board)

            if r!=0:
                r_end=float(r/self.upperbound)
                if r_end >=self.currecord:
                    print()
                    print("New record performedmoves:", r,  end="")
                    for move in board.performedmoves:
                        print("|", move.point, end="")
                        print(move.line, end="")
                    print("|")
                temprewards=sorted(self.recentrewards, reverse=False)
                self.r_percentile=temprewards[int(self.rewardpercentage*len(temprewards))]
                #print(self.recentrewards, r_percentile)
                if r_end >= 1:
                    shapedreward=1
                if r_end< 1 and r_end ==self.r_percentile:
                    shapedreward=np.random.choice([-1,1])
                    #print("I am a random win at step", r)
                if r_end< 1 and r_end < self.r_percentile:
                    shapedreward=-1
                    #print("I am a loss at step", r)
                if r_end< 1 and r_end > self.r_percentile:
                    #print("I am a win at step", r)
                    shapedreward=1
                self.recentrewards.append(r_end)
                if len(self.recentrewards)>self.maxLenofRewardBuffer:
                    self.recentrewards.pop(0)

                #display(board)
                print()
                print("Game Steps:", episodeStep,shapedreward)
                return [(x[0],x[1],shapedreward) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        learn_time=time.time()
        for i in range(1, self.args.numIters+1):
            # bookkeeping
            learn_iter_time=time.time()
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
    
                for eps in range(self.args.numEps):
                    iterationTrainExamples += self.executeEpisode()
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            #self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.nnet.train(trainExamples)
            
            print('Following is played by PureNNt:')
            purenntplayer=PureNNtPlayer(self.game, self.nnet, self.args, temp=0).play
            arena = Arena(purenntplayer, self.game)
            print('PureNNt Real Performance:', arena.playGames(self.args.arenaCompare))

            print('Following is played by NNtBasedMCTS:')
            nntbasedmctsplayer = NNtBasedMCTSPlayer(self.game, self.nnet, self.args, temp=0, percentile=self.r_percentile).play
            arena = Arena(nntbasedmctsplayer, self.game)
            print('NNtBasedMCTS Real Performance:', arena.playGames(self.args.arenaCompare))

            print('ACCEPTING NEW MODEL DIRECTLY')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
