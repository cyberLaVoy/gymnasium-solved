import gym
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import Agent
from memory import ReplayMemory

def displayFrames(f):
    plt.imshow( np.mean(f, axis=2)/255, cmap="gray" )
    plt.show()

def buildMemory(game, memory, amount):
    while len(memory) < amount:
        done = False
        game.reset() 
        while not done:
            a = np.random.choice( game.getActionSpace() )
            s, r, done, info = game.step(a)
            memory.append( (s, a, r, info["life_lost"]) )
        print("Memory:", str(len(memory)) + '/' + str(amount))

def train(game, agent, memory, episodes=20000, render=True):
    act = 0
    for eps in range(1, episodes+1):
        epsReward = 0
        done = False
        game.reset() 
        while not done:
            if render:
                game.render()
            # choose action
            if game.getFramesAfterDeath() <= 2:
                a = 1
            else:
                a = agent.getAction( memory.getState() )
            # step
            s, r, done, info = game.step(a)
            act += 1
            # add to memory
            memory.append( (s, a, r, info["life_lost"]) )
            # learn
            if act % 4 == 0:
                agent.learn( memory.sample(32) )
            # upkeep for next step 
            epsReward += r
        print( "Episode " + str(eps)+'/'+str(episodes), "Action count:", act, "Reward:", epsReward)
        if eps % 50 == 0:
            epochScore, actionCounts = testAgent(game, agent, 5)
            print("Epoch score:", epochScore, "Action counts:", actionCounts)
    game.close()

def testAgent(game, agent, episodes):
    epochScore = 0
    actionCounts = [0]*game.getActionSpace()
    memory = ReplayMemory(game.reset(), game.getActionSpace())
    for _ in range(episodes):
        episodeScore = 0
        game.reset()
        done = False
        while not done:
            game.render()
            # choose action
            if game.getFramesAfterDeath() <= 2:
                a = 1
            else:
                a = agent.getAction( memory.getState() )
            actionCounts[a] += 1
            # step
            s, r, done, info = game.step(a)
            # add to memory
            memory.append( (s, a, r, info["life_lost"]) )
            # upkeep for next step
            episodeScore += r 
        epochScore += episodeScore 
    return epochScore/episodes, actionCounts

def main():
    #game = Atari("BreakoutDeterministic-v4")
    #agentName = "atari_agent_breakout"
    game = Atari("PongDeterministic-v4")
    agentName = "atari_agent_pong"

    # set to None if no model to load
    #load = None
    #load = "atari_agent_breakout_best.h5"
    load = "atari_agent_pong_best.h5"

    trainAndSave = True
    testAndObserve = False

    if trainAndSave:
        agent = Agent(agentName, game.getActionSpace(), modelLoad=load, targetLoad=load)
        memory = ReplayMemory(game.reset(), game.getActionSpace(), size=1000000)
        if load is None:
            buildMemory(game, memory, 1024)
        train(game, agent, memory, render=True)
    if testAndObserve:
        agent = Agent(agentName, game.getActionSpace(), modelLoad=load)
        testAgent(game, agent, 100)


if __name__ == "__main__":
    main()