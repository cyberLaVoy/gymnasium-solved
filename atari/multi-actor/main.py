import gym, time, multiprocessing, os
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import LearnerAgentPolicy, LearnerAgentRND, ActorAgent
from memory import PolicyReplayMemory, RNDReplayMemory, ActorReplayMemory
# do not import tensorflow here, as it will break the multiprocessing library

def train(game, agentName, loadPolicy, loadRND, cpuCount, memSizePolicy, memSizeRND, actorThreshPolicy, actorThreshRND, render, enableLearnerGPU):
    processes = []
    weightChans = []
    expChansPolicy = []
    expChansRND = []
    oracleScore = multiprocessing.Value("i", 0)

    for actorID in range( cpuCount ):

        expChanPolicy = multiprocessing.Queue()
        expChanRND = multiprocessing.Queue()
        expChansPolicy.append(expChanPolicy)
        expChansRND.append(expChanRND)

        weightsChan = multiprocessing.Queue()
        weightChans.append(weightsChan)

        actorMemPolicy = ActorReplayMemory(expChanPolicy, thresh=actorThreshPolicy)
        actorMemRND = ActorReplayMemory(expChanRND, thresh=actorThreshRND, normalized=True)

        actor = ActorAgent(game, actorMemPolicy, actorMemRND, weightsChan, actorID, oracleScore, 
                           actorID == 0 and render)

        proc = multiprocessing.Process(target=actor.explore)
        processes.append(proc)

        print("Actor", actorID, "created")

    learnerMemRND = RNDReplayMemory(expChansRND, size=memSizeRND)
    learner = LearnerAgentRND(learnerMemRND, agentName, weightChans, loadRND)
    processes.append( multiprocessing.Process(target=learner.learn) )

    for proc in processes:
        proc.start()

    learnerMemPolicy = PolicyReplayMemory(expChansPolicy, size=memSizePolicy)
    learner = LearnerAgentPolicy(learnerMemPolicy, agentName, weightChans, loadPolicy, enableGPU=enableLearnerGPU)
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
    option = 5
    game = Atari( games[option]+"Deterministic-v4" )
    print(game.env.unwrapped.get_action_meanings())
    agentName = "atari_agent_" + games[option]

    # set to None if no model to load
    loadPolicy = None
    loadRND = None
    #load = "models/atari_agent_" + games[option] + "_best.h5"
    #load = "atari_agent_" + games[option] + ".h5"

    cpuCount = os.cpu_count() - 4

    actorThreshPolicy = 2**9
    memSizePolicy = 2**17

    actorThreshRND = 2**9
    memSizeRND = 2**17

    render = True
    enableLearnerGPU = True
    train(game, agentName, loadPolicy, loadRND, cpuCount, memSizePolicy, memSizeRND, actorThreshPolicy, actorThreshRND, render, enableLearnerGPU)


if __name__ == "__main__":
    main()