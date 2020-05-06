import gym, time, multiprocessing, os
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import LearnerAgent, ActorAgent
from memory import ReplayMemory
# do not import tensorflow here, as it will break the multiprocessing library

def buildMemory(game, memory, amount):
    while len(memory) < amount:
        done = False
        game.reset() 
        while not done:
            a = np.random.choice( game.getActionSpace() )
            s, r, done, info = game.step(a)
            s = np.ones((84,84)) # temporary garbage
            memory.append( (s, a, r, info["life_lost"]) )
        print("Memory:", str(len(memory)) + '/' + str(amount))

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

        memory = ReplayMemory(np.ones((84,84)), game.getActionSpace(), prioritized=True)
        processes = []

        actorChans = []
        for actorID in range( os.cpu_count() ):
            learnerChan, actorChan = multiprocessing.Pipe()
            actorChans.append(actorChan)
            actor = ActorAgent(game, learnerChan, actorID, load)
            proc = multiprocessing.Process(target=actor.explore, args=(100,))
            processes.append(proc)
            proc.start()
            print("Actor", actorID, "created")

        buildMemory(game, memory, 64)
        learner = LearnerAgent(agentName, game.getActionSpace(), actorChans, load)
        learner.learn(memory, 10000)
        #proc = multiprocessing.Process(target=learner.learn, args=(memory,100))
        #processes.append(proc)
        #proc.start()

        for proc in processes:
            proc.join()


if __name__ == "__main__":
    """
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    with tf.device('/CPU:0'):
    """
    main()