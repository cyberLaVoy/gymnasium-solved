import gym, time, multiprocessing, os
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import LearnerAgentPolicy, ActorAgent
from memory import PolicyReplayMemory, ActorReplayMemory
# do not import tensorflow here, as it will break the multiprocessing library

def train(game, agentName, loadPolicy, cpuCount, memSizePolicy, expChanCap, render, actionSpace, enableLearnerGPU):

    # each actor gets access to a central shared best score tracker
    oracleScore = multiprocessing.Value( "i", 0 )
    # each actor gets access to the experience queue
    expChanPolicy = multiprocessing.Queue( expChanCap )
    # each actor gets access to a fresh weights queue
    weightsChan = multiprocessing.Queue( 1 )

    for actorID in range( cpuCount ):
        actorMemPolicy = ActorReplayMemory(expChanPolicy)
        actor = ActorAgent(game, actorMemPolicy, weightsChan, actorID, cpuCount, oracleScore, 
                           render=(actorID == (cpuCount-1) and render), actionSpace=actionSpace)

        proc = multiprocessing.Process(target=actor.explore)
        # start actor process upon creation
        proc.start()

        print("Actor", actorID, "started")

    learnerMemPolicy = PolicyReplayMemory(expChanPolicy, size=memSizePolicy)
    learner = LearnerAgentPolicy(learnerMemPolicy, agentName, weightsChan, loadPolicy, actionSpace=actionSpace, enableGPU=enableLearnerGPU)
    learner.learn()

def main():
    games = [
             "Breakout", # 0
             "Pong", # 1
             "MsPacman", # 2
             "SpaceInvaders", # 3
             "Asteroids", # 4

             "MontezumaRevenge", # 5
             "Skiing", # 6
             "Pitfall", # 7
             "Solaris", # 8

             "Enduro", # 9
            ]
    option = 0
    game = Atari( games[option]+"Deterministic-v4" )
    actionSpace = game.getActionSpace()
    print(game.env.unwrapped.get_action_meanings())
    agentName = "atari_agent_" + games[option]

    # set to None if no model to load
    loadPolicy = None
    #load = "models/atari_agent_" + games[option] + "_best.h5"
    #load = "atari_agent_" + games[option] + ".h5"

    render = True
    enableLearnerGPU = True
    cpuCount = os.cpu_count()

    # accounts for majority of the memory used by program
    memSizePolicy = 2**18
    """
    Syncronizes transfer of experiences between learner and actors.
    Note a tradeoff that occurs: if higher, then actors will be faster
    and learner will be slower (and vice versa).
    """
    expChanCap = 256

    train(game, agentName, loadPolicy, cpuCount, memSizePolicy, expChanCap, render, actionSpace, enableLearnerGPU)


if __name__ == "__main__":
    main()