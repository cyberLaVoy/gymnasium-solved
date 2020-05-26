import time
import numpy as np
from matplotlib import pyplot as plt
from atari import Atari
from agent import ActorAgent, LearnerAgent
from memory import ExperienceReplay

def train(game, actor, learner, memory, episodes=10000, render=False, epochSteps=50):
    bestScore = 0
    act = 0
    for eps in range(1, episodes+1):
        print(eps)
        done = False
        s0 = game.reset() 
        episode = []
        while not done:
            # choose action
            if game.getFramesAfterDeath() < 2:
                a = 1
            elif eps <= 32:
                a = np.random.choice( game.getActionSpace() )
            else:
                a = np.argmax( actor.predict( [s0] ) )
            # step
            s1, r, done, info = game.step(a)
            act += 1
            # add to memory
            mem = np.reshape(np.array([[s0], [s1], a, r, info["life_lost"]]), (1,5))
            episode.append(mem)
            # upkeep for next step 
            s0 = s1
            if render:
                game.render()
            if act % 4 == 0 and eps > 32:
                t0 = time.time()
                learner.learn(memory)
                print("Learn time:", time.time() - t0)
                t0 = time.time()
                actor.setWeights(learner.getWeights())
                print("Update time:",time.time() - t0)
        memory.append(episode[:])
        actor.model.reset_states()
        if game.getScore() > bestScore:
            game.saveEpisode()
            bestScore = game.getScore()
    game.close()

def main():
    game = Atari("BreakoutDeterministic-v4")
    agentName = "atari_agent_breakout"
    #game = Atari("PongDeterministic-v4")
    #agentName = "atari_agent_pong"
    #game = Atari("MsPacmanDeterministic-v4")
    #agentName = "atari_agent_ms_pacman"
    #game = Atari("SpaceInvadersDeterministic-v4")
    #agentName = "atari_agent_space_invaders"
    #game = Atari("MontezumaRevengeDeterministic-v4")
    #agentName = "atari_agent_montezuma_revenge"
    print("Action meanings:", game.getActionMeanings())

    # set to None if no model to load
    load = None
    #load = "atari_agent_breakout.h5"
    #load = "atari_agent_pong_best.h5"
    #load = "atari_agent_ms_pacman_best.h5"
    #load = "atari_agent_space_invaders.h5"

    trainAndSave = True

    render = True

    if trainAndSave:
        memory = ExperienceReplay()
        actor = ActorAgent(game.getActionSpace(), modelLoad=load)
        learner = LearnerAgent(agentName, game.getActionSpace(), modelLoad=load)
        train(game, actor, learner, memory, episodes=50000, render=render)

if __name__ == "__main__":
    main()