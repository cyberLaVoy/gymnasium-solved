import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from atari import Atari
from agent import Agent
from memory import ReplayMemory

def displayFrames(f):
    plt.imshow( f/255, cmap="gray" )
    plt.show()

def buildMemory(game, memory, amount):
    while len(memory) < amount:
        done = False
        game.reset() 
        while not done:
            a = np.random.choice( game.getActionSpace() )
            s, r, done, info = game.step(a)
            #displayFrames(s)
            memory.append( (s, a, r, info["life_lost"]) )
        print("Memory:", str(len(memory)) + '/' + str(amount))

def train(game, agent, memory, episodes=10000, render=False, epochSteps=50):
    act = 0
    epochReward = 0
    epochEsp = 1
    bestScore = 0
    for eps in range(1, episodes+1):
        epsReward = 0
        done = False
        game.reset() 
        t0 = time.time()
        while not done:
            # choose action
            if game.getFramesAfterDeath() < 2:
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
                agent.learn( memory )
            # upkeep for next step 
            epsReward += r
            if render:
                game.render()
                agent.viewAttention( memory.getState() )

        epochReward += epsReward
        timeTaken = time.time() - t0
        avgReward = epochReward / epochEsp
        epochEsp += 1
        efficiency = epsReward / timeTaken
        print( "Episode " + str(eps)+'/'+str(episodes), "Action count:", act, "Reward:", epsReward, "Time taken(s):", round(timeTaken, 2), "AVG Reward:", round(avgReward, 2), "Efficiency:", round(efficiency, 2) )
        if game.getScore() > bestScore:
            game.saveEpisode()
            bestScore = game.getScore()
        if eps % epochSteps == 0:
            epochReward = 0
            epochEsp = 1
            testScore, actionCounts = testAgent(game, agent, 5, render)
            print("Avg test score:", testScore, "Action counts:", actionCounts)

    game.close()

def testAgent(game, agent, episodes, render=False):
    epochScore = 0
    actionCounts = [0]*game.getActionSpace()
    memory = ReplayMemory(game.reset(), game.getActionSpace(), size=0, prioritized=False)
    for _ in range(episodes):
        episodeScore = 0
        game.reset()
        done = False
        while not done:
            # choose action
            if game.getFramesAfterDeath() < 2:
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
            if render:
                game.render()
        epochScore += episodeScore 
    return epochScore/episodes, actionCounts

def main():
    # A list of all possible Atari games
    games = [
        "Adventure", # 0
        "AirRaid", # 1
        "Alien", # 2
        "Amidar", # 3
        "Assault", # 4
        "Asterix", # 5
        "Asteroids", # 6
        "Atlantis", # 7
        "BankHeist", # 8
        "BattleZone", # 9
        "BeamRider", # 10
        "Berzerk", # 11
        "Bowling", # 12
        "Boxing", # 13
        "Breakout", # 14
        "Carnival", # 15
        "Centipede", # 16
        "ChopperCommand", # 17
        "CrazyClimber", # 18
        "Defender", # 19
        "DemonAttack", # 20
        "DoubleDunk", # 21
        "ElevatorAction", # 22
        "Enduro", # 23
        "FishingDerby", # 24
        "Freeway", # 25
        "Frostbite", # 26
        "Gopher", # 27
        "Gravitar", # 28
        "Hero", # 29
        "IceHockey", # 30
        "Jamesbond", # 31
        "JourneyEscape", # 32
        "Kangaroo", # 33
        "Krull", # 34
        "KungFuMaster", # 35
        "MontezumaRevenge", # 36
        "MsPacman", # 37
        "NameThisGame", # 38
        "Phoenix", # 39
        "Pitfall", # 40
        "Pong", # 41
        "Pooyan", # 42
        "PrivateEye", # 43
        "Qbert", # 44
        "Riverraid", # 45
        "RoadRunner", # 46
        "RobotTank", # 47
        "Seaquest", # 48
        "Skiings", # 49
        "Solaris", # 50
        "SpaceInvaders", # 51
        "StarGunner", # 52
        "Tennis", # 53
        "TimePilot", # 54
        "Tutankham", # 55
        "UpNDown", # 56
        "Venture", # 57
        "VideoPinball", # 58
        "WizardofWor", # 59
        "Zaxxon" # 60
    ]
    option = 14
    game = Atari( games[option]+"Deterministic-v4" )
    agentName = "atari_agent_" + games[option]

    # set to None if no model to pretainedModel
    pretainedModel = None
    #pretainedModel = "atari_agent_breakout_best.h5"
    #pretainedModel = "atari_agent_pong_best.h5"
    #pretainedModel = "atari_agent_ms_pacman_best.h5"
    #pretainedModel = "atari_agent_space_invaders.h5"

    trainAndSave = True
    testAndObserve = False

    render = False
    attention = True

    if trainAndSave:
        agent = Agent(agentName, game.getActionSpace(), modelLoad=pretainedModel, attentionView=attention)
        memory = ReplayMemory(game.reset(), game.getActionSpace(), prioritized=True)
        if pretainedModel is None:
            buildMemory(game, memory, 2**12)
        train(game, agent, memory, episodes=50000, render=render)
    if testAndObserve:
        agent = Agent(agentName, game.getActionSpace(), modelLoad=pretainedModel)
        testAgent(game, agent, 100, render=True)


if __name__ == "__main__":
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    main()