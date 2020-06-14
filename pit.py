import Arena
from MCTS import MCTS
from morpionsolitaire.MorpionSolitaireGame import MorpionSolitaireGame, display
from morpionsolitaire.MorpionSolitairePlayers import *
from morpionsolitaire.tensorflow.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""

"""

g = MorpionSolitaireGame(20)

# all players
rp = RandomPlayer(g).play
#rp2 = RandomPlayer(g).play
#gp = GreedyOthelloPlayer(g).play
#hp = HumanOthelloPlayer(g).play

# nnet players
nn = NNet(g)
nn.load_checkpoint('../morpionsolitairewithouttree20/models','100_best.pth.tar')
args = dotdict({'numMCTSSims': 1000, 'cpuct':1.0})
nntbasedMCTSPlayer=NNtBasedMCTSPlayer(g, nn, args, temp=0).play
#purenntplayer=PureNNtPlayer(g, nn, args, temp=0).play

#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(nntbasedMCTSPlayer, g, display=display)
print(arena.playGames(1, verbose=False))
#print(results)
