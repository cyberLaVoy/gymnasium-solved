import gym, time, multiprocessing, os
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import LearnerAgent, ActorAgent
from memory import LearnerReplayMemory, ActorReplayMemory
# do not import tensorflow here, as it will break the multiprocessing library

def buildMemory(game, memory, episodes):
    for eps in range(episodes):
        done = False
        s0 = game.reset() 
        while not done:
            a = np.random.choice( game.getActionSpace() )
            s1, r, done, info = game.step(a)
            memory.append( (s0, s1, a, r, info["life_lost"]) )
            s0 = s1
        print("Episode", str(eps) + '/' + str(episodes) )

def main():
    game = Atari("BreakoutDeterministic-v4")
    agentName = "atari_agent_breakout"
    #game = Atari("PongDeterministic-v4")
    #agentName = "atari_agent_pong"
    #game = Atari("MsPacmanDeterministic-v4")
    #agentName = "atari_agent_ms_pacman"
    #game = Atari("SpaceInvadersDeterministic-v4")
    #agentName = "atari_agent_space_invaders"
    print("Action meanings:", game.getActionMeanings())

    # set to None if no model to load
    load = None
    #load = "atari_agent_breakout_best.h5"
    #load = "atari_agent_pong_best.h5"
    #load = "atari_agent_ms_pacman_best.h5"
    #load = "atari_agent_space_invaders.h5"

    trainAndSave = True

    if trainAndSave:

        processes = []

        weightChans = []
        expChans = []
        for actorID in range( os.cpu_count() // 2 ):
            weightsChan = multiprocessing.Queue()
            expChan = multiprocessing.Queue()
            weightChans.append(weightsChan)
            expChans.append(expChan)
            actorMemory = ActorReplayMemory(expChan)
            render = False
            if actorID == 0:
                render = True
            actor = ActorAgent(game, actorMemory, weightsChan, actorID, render, load)
            proc = multiprocessing.Process(target=actor.explore)
            processes.append(proc)
            print("Actor", actorID, "created")

        learnerMemory = LearnerReplayMemory(expChans)
        print("Building random replay...")
        buildMemory(game, learnerMemory, 128)
        print("Starting actors...")
        for proc in processes:
            proc.start()
        print("Begin learning...")
        learner = LearnerAgent(learnerMemory, agentName, game.getActionSpace(), weightChans, load)
        learner.learn()

if __name__ == "__main__":
    """
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    with tf.device('/CPU:0'):
    """
    main()