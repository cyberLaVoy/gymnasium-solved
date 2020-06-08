import gym, time, multiprocessing, os
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import LearnerAgent, ActorAgent
from memory import LearnerReplayMemory, ActorReplayMemory
# do not import tensorflow here, as it will break the multiprocessing library

def train(game, agentName, load):
    processes = []
    weightChans = []
    expChans = []
    oracleScore = multiprocessing.Value("i", 0)
    for actorID in range( os.cpu_count() // 2 - 1 ):
        weightsChan = multiprocessing.Queue()
        expChan = multiprocessing.Queue()
        weightChans.append(weightsChan)
        expChans.append(expChan)
        render = False
        if actorID == 0:
            render = False
        actorMemory = ActorReplayMemory(expChan, thresh=2**10)
        actor = ActorAgent(game, actorMemory, weightsChan, actorID, render, oracleScore)
        proc = multiprocessing.Process(target=actor.explore)
        processes.append(proc)
        print("Actor", actorID, "created")

    learnerMemory = LearnerReplayMemory(expChans, size=250000)
    print("Starting actors...")
    for proc in processes:
        proc.start()
    print("Begin learning...")
    learner = LearnerAgent(learnerMemory, agentName, game.getActionSpace(), weightChans, load)
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
            ]
    option = 2
    game = Atari(games[option]+"Deterministic-v4")
    agentName = "atari_agent_" + games[option]

    # set to None if no model to load
    load = None
    load = "models/atari_agent_" + games[option] + "_best.h5"

    train(game, agentName, load)


if __name__ == "__main__":
    main()