#!/usr/bin/env python3
import gym, joblib, sys
import numpy as np 

NUM_DISCRETE_SECTIONS = 500

def saveModel(model, modelPath):
    joblib.dump(model, modelPath) 
def loadModel(modelPath):
    return joblib.load(modelPath)

def learnQTable(env, Q, completionTries=100, alpha=.5, gamma=.5, epochs=5000, randomInfluence=1, visuaDisplay=False):
    rev_list = [] # rewards per epochs calculate
    maxPosition = -.45
    # Q-learning Algorithm
    for i in range(epochs):
        s = getDiscreteObservation(env, env.reset()) 
        rAll = 0
        done = False
        tries = 0 
        # The Q-Table learning algorithm
        while tries < completionTries:
            tries += 1
            # Choose action from Q table; influence of randomness decreases as i increases
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(randomInfluence/(i+1))) 
            #Get new state & reward from environment
            s1,reward,done,_ = env.step(a)
            position = s1[0]
            if position >= env.goal_position: 
                #print("Yay - I made it!!")
                reward = 1
            if position > maxPosition:
                reward += .5
                maxPosition = position
            if abs(position+.5) > 0:
                reward += .15
            if s1[1] < 0 and a == 0:
                reward += .25
            if s1[1] > 0 and a == 2:
                reward += .25
            
            s1 = getDiscreteObservation(env, s1)
            #Update Q-Table with new knowledge
            Q[s,a] = (1-alpha)*Q[s,a] + alpha*(reward + gamma*np.max(Q[s1,:]))
            rAll += reward
            s = s1
            if done and position >= env.goal_position:
                break
            if visuaDisplay:
                env.render()
        rev_list.append(rAll)
    print( "Reward Sum on all epochs " + str(sum(rev_list)/epochs) )
    print( "Final Values Q-Table" )
    #np.set_printoptions(threshold=sys.maxsize)
    print( Q )
    print( Q.shape )

def evaluateQTable(env, Q, numTests, visuaDisplay=False):
    totalOfRewards = 0
    totalCompleteRuns = 0
    totalIncompleteRuns = 0
    for i in range(numTests):
        observation = env.reset()
        position = observation[0]
        observation = getDiscreteObservation(env, observation)
        totalReward = 0 
        done = False
        while not done:
            action = np.argmax(Q[observation,:])
            observation, reward, done, _ = env.step(action)
            position = observation[0]
            observation = getDiscreteObservation(env, observation)
            totalReward += reward
            if visuaDisplay:
                env.render()
        if done and position >= env.goal_position:
            totalCompleteRuns += 1
            totalOfRewards += totalReward
        else:
            totalIncompleteRuns += 1
    print("Total complete runs:", totalCompleteRuns)
    print("Total incomplete runs:", totalIncompleteRuns)
    print("Average steps to complete:", totalOfRewards/totalCompleteRuns)

def getNumDiscretePositions(env, numSections):
    sizeOfPositionRange = env.observation_space.high[0] - env.observation_space.low[0]
    return sizeOfPositionRange // ( sizeOfPositionRange / numSections )
def getNumDiscreteSpeeds(env, numSections):
    sizeOfSpeedRange = env.observation_space.high[1] - env.observation_space.low[1]
    return sizeOfSpeedRange // ( sizeOfSpeedRange / numSections )

def getDiscreteObservation(env, observation, numSections=NUM_DISCRETE_SECTIONS):
    discretePosition =  ( observation[0] + abs(env.observation_space.low[0]) ) * numSections 
    discreteSpeed =  ( observation[1] + abs(env.observation_space.low[1]) ) * numSections 
    return int( discreteSpeed*getNumDiscretePositions(env, numSections) + discretePosition )

def getNumDiscreteObservations(env, numSections=NUM_DISCRETE_SECTIONS):
    return getDiscreteObservation(env, [env.observation_space.high[0], env.observation_space.high[1]], numSections)

def main():
    QTableName = "mouatingCarAI_Q-table"
    env = gym.make('MountainCar-v0')
    learnAndSave = True
    loadAndEvaluate = False

    if learnAndSave:
        # env.observation_space.low = [min_position, min_speed]
        # env.observation_space.high = [max_position, max_speed]
        #numDiscreteObservations = getNumDiscreteObservations(env)
        #Q = np.zeros( [ numDiscreteObservations, env.action_space.n ] )
        Q = loadModel(QTableName+".joblib")
        #learnQTable(env, Q, completionTries=300, alpha=.65, gamma=.8, epochs=240000, randomInfluence=3, visuaDisplay=False)
        learnQTable(env, Q, completionTries=300, alpha=.5, gamma=.7, epochs=5000, randomInfluence=0, visuaDisplay=True)
        saveModel(Q, QTableName+".joblib")

    if loadAndEvaluate:
        Q = loadModel(QTableName+".joblib")
        evaluateQTable(env, Q, numTests=15000, visuaDisplay=False)


if __name__ == "__main__":
    main()