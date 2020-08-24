import gym, time, multiprocessing, os
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import LearnerAgentPolicy, LearnerAgentAE, ActorAgent
from memory import PolicyReplayMemory, AEReplayMemory, ActorReplayMemory
# do not import tensorflow here, as it will break the multiprocessing library

def train(game, agentName, loadPolicy, loadAE, cpuCount, memSizePolicy, memSizeAE, actorThreshPolicy, actorThreshAE, render, enableLearnerGPU):
    processes = []
    weightChans = []
    expChansPolicy = []
    expChansAE = []
    oracleScore = multiprocessing.Value("i", 0)
    actionSpace = game.getActionSpace()

    for actorID in range( cpuCount ):

        expChanPolicy = multiprocessing.Queue( cpuCount*actorThreshPolicy )
        expChanAE = multiprocessing.Queue( cpuCount*actorThreshAE )
        expChansPolicy.append(expChanPolicy)
        expChansAE.append(expChanAE)

        weightsChan = multiprocessing.Queue()
        weightChans.append(weightsChan)

        actorMemPolicy = ActorReplayMemory(expChanPolicy, thresh=actorThreshPolicy)
        actorMemAE = ActorReplayMemory(expChanAE, thresh=actorThreshAE)

        actor = ActorAgent(game, actionSpace, actorMemPolicy, actorMemAE, weightsChan, actorID, cpuCount, oracleScore, 
                           actorID == cpuCount-1 and render)

        proc = multiprocessing.Process(target=actor.explore)
        processes.append(proc)

        print("Actor", actorID, "created")

    learnerMemAE = AEReplayMemory(expChansAE, size=memSizeAE)
    learner = LearnerAgentAE(actionSpace, learnerMemAE, agentName, weightChans, loadAE)
    processes.append( multiprocessing.Process(target=learner.learn) )

    for proc in processes:
        proc.start()

    learnerMemPolicy = PolicyReplayMemory(expChansPolicy, size=memSizePolicy)
    learner = LearnerAgentPolicy(actionSpace, learnerMemPolicy, agentName, weightChans, loadPolicy, enableGPU=enableLearnerGPU)
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
    print(game.env.unwrapped.get_action_meanings())
    agentName = "atari_agent_" + games[option]

    # set to None if no model to load
    loadPolicy = None
    loadAE = None
    #load = "models/atari_agent_" + games[option] + "_best.h5"
    #load = "atari_agent_" + games[option] + ".h5"

    cpuCount = os.cpu_count() // 2

    actorThreshPolicy = 2**10
    memSizePolicy = 2**18

    actorThreshAE = 2**8
    memSizeAE = 2**18

    render = True
    enableLearnerGPU = True
    train(game, agentName, loadPolicy, loadAE, cpuCount, memSizePolicy, memSizeAE, actorThreshPolicy, actorThreshAE, render, enableLearnerGPU)


if __name__ == "__main__":
    main()