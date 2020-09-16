import gym, time, multiprocessing, threading, os
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import LearnerAgentPolicy, LearnerAgentRND, LearnerAgentEmbedding, ActorAgent
from memory import ExperienceReplayMemory
# do not import tensorflow here, as it will break the multiprocessing library

def train(game, agentName, loadPolicy, loadRND, loadEmbedding, cpuCount, replayMemSize, expChanCap, render, actionSpace, enableLearnerGPU):

    # each actor gets access to a central shared best score tracker
    oracleScore = multiprocessing.Value( "i", 0 )
    # each actor gets access to the experience queue
    expChan = multiprocessing.Queue( expChanCap )
    # each actor gets access to a fresh weights queue
    weightsChan = multiprocessing.Queue( 1 )

    for actorID in range( cpuCount ):
        actor = ActorAgent(game, expChan, weightsChan, actorID, cpuCount, oracleScore, 
                           render=(actorID == (cpuCount-1) and render), actionSpace=actionSpace)

        proc = multiprocessing.Process(target=actor.explore)
        # start actor process upon creation
        proc.start()

        print("Actor", actorID, "started")

    # create and start loading memory from actors (policy)
    learnerMemory = ExperienceReplayMemory(expChan, size=replayMemSize)
    memLoadThread = threading.Thread(target=learnerMemory.load_nolock)
    memLoadThread.start()

    learnerPolicy = LearnerAgentPolicy(learnerMemory, agentName, weightsChan, load=loadPolicy, actionSpace=actionSpace, enableGPU=enableLearnerGPU)
    learnerEmbedding = LearnerAgentEmbedding(learnerMemory, agentName, weightsChan, load=loadEmbedding, actionSpace=actionSpace, enableGPU=enableLearnerGPU)
    learnerRND = LearnerAgentRND(learnerMemory, agentName, weightsChan, load=loadRND, actionSpace=actionSpace, enableGPU=enableLearnerGPU)

    # start policy learner
    threadPolicy = threading.Thread(target=learnerPolicy.learn)
    threadPolicy.start()
    # start embedding learner
    threadEmbedding = threading.Thread(target=learnerEmbedding.learn)
    threadEmbedding.start()
    # start rnd learner
    threadRND = threading.Thread(target=learnerRND.learn)
    threadRND.start()

    # join with policy learner
    threadPolicy.join()

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
    option = 5
    game = Atari( games[option]+"Deterministic-v4" )
    #game = Atari( games[option]+"Deterministic-v0" )
    actionSpace = game.getActionSpace()
    print(game.env.unwrapped.get_action_meanings())
    agentName = "atari_agent_" + games[option]

    # set to None if no model to load
    loadPolicy = None
    #loadPolicy = "models/atari_agent_" + games[option] + "_best.h5"
    #loadPolicy = "atari_agent_" + games[option] + "_policy.h5"
    loadRND = None
    #loadRND = "models/atari_agent_" + games[option] + "_rnd_best.h5"
    #loadRND = "atari_agent_" + games[option] + "_rnd.h5"
    loadEmbedding = None
    #loadEmbedding = "models/atari_agent_" + games[option] + "_embedding_best.h5"
    #loadEmbedding = "atari_agent_" + games[option] + "_embedding.h5"

    render = True
    enableLearnerGPU = True
    cpuCount = os.cpu_count()
    # accounts for majority of the memory used by program
    replayMemSize = 2**17
    # simply ensures that the experience chan doesn't keep growing infinitely
    expChanCap = 256

    train(game, agentName, loadPolicy, loadRND, loadEmbedding, cpuCount, replayMemSize, expChanCap, render, actionSpace, enableLearnerGPU)


if __name__ == "__main__":
    main()