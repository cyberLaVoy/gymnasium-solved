#!/usr/bin/env python3
import gym, joblib
import numpy as np 

# this number is how many sections each coninuous observation varibable is split into
# note: since the environment isn't complex, performance is best when number is low (but not too low)
NUM_DISCRETE_SECTIONS = 25

def saveModel(model, modelPath):
    joblib.dump(model, modelPath) 
def loadModel(modelPath):
    return joblib.load(modelPath)

def observeAgent(env, Q):
    while True:
        s = getDiscreteObservation(env, env.reset()) 
        while True:
            a = np.argmax(Q[s,:]) 
            s,_,_,_ = env.step(a)
            if s[0] >= env.goal_position:
                print("Made it!")
                break
            s = getDiscreteObservation(env, s) 
            env.render()


def learnQTable(env, Q, periods=250, alpha=.7, gamma=.8, epochs=5000, decay=.99, render=False):
    randomInfluence = 1
    almost = False
    goalCount = 0
    for epoch in range(1, epochs+1):
        if epoch % 100 == 0:
            print("Epoch", str(epoch)+'/'+str(epochs), "Acheivement", str(goalCount)+'/'+"100")
            goalCount = 0
        s = getDiscreteObservation(env, env.reset()) 
        for _ in range(periods):
            if render:
                env.render()
            # Choose action from Q table; influence of randomness decreases iterations
            randomInfluence *= decay
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*randomInfluence) 
            # get new state and reward from environment, after taking action
            s1,reward,_,_ = env.step(a)
            # reward engineering
            reward = -1
            if s1[0] >= env.goal_position*2/3 and not almost:
                reward = .25 # a one-time reward to help the agent know what to shoot for
                almost = True
            if s1[0] >= env.goal_position:
                goalCount += 1
                reward = 1
            s1 = getDiscreteObservation(env, s1)
            # update Q-Table with new knowledge
            Q[s,a] = (1-alpha)*Q[s,a] + alpha*(reward + gamma*np.max(Q[s1,:]))
            s = s1
            if reward == 1:
                break


# provides a unique integer for any given pair of positive integers
def cantorPairing(a, b):
    return int( 1/2*(a+b)*(a+b+1)+b )
def getDiscreteObservation(env, observation, numSections=NUM_DISCRETE_SECTIONS):
    positionRange = abs(env.observation_space.high[0]) + abs(env.observation_space.low[0])
    normalizedPosition = (observation[0] + abs(env.observation_space.low[0]) ) / positionRange # a value between 0 and 1
    discretePosition = int( normalizedPosition * numSections )
    speedRange = abs(env.observation_space.high[1]) + abs(env.observation_space.low[1])
    normalizedSpeed = (observation[1] + abs(env.observation_space.low[1]) ) / speedRange # a value between 0 and 1
    discreteSpeed = int( normalizedSpeed * numSections )
    return cantorPairing(discretePosition, discreteSpeed)
def getNumDiscreteObservations(numDiscreteSections=NUM_DISCRETE_SECTIONS):
    return cantorPairing(numDiscreteSections, numDiscreteSections)


def main():
    QTableName = "MountainCarAI-Qtable"
    env = gym.make('MountainCar-v0')

    learnAndSave = False
    loadAndObserve = True

    if learnAndSave:
        # env.observation_space.low = [min_position, min_speed]
        # env.observation_space.high = [max_position, max_speed]
        Q = np.zeros( [getNumDiscreteObservations(), env.action_space.n] )
        learnQTable(env, Q, render=False)
        saveModel(Q, QTableName+".joblib")

    if loadAndObserve:
        Q = loadModel(QTableName+".joblib")
        observeAgent(env, Q)

if __name__ == "__main__":
    main()