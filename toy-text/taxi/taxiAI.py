#!/usr/bin/env python3
import gym, joblib, sys
import numpy as np 


def saveModel(model, modelPath):
    joblib.dump(model, modelPath) 
def loadModel(modelPath):
    return joblib.load(modelPath)

def learnQTable(env, Q, completionTries=100, alpha=.5, gamma=.5, epochs=5000, randomInfluence=1, visuaDisplay=False):
    rev_list = [] # rewards per epochs calculate
    # Q-learning Algorithm
    for i in range(epochs):
        s = env.reset()
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
            #Update Q-Table with new knowledge
            Q[s,a] = (1-alpha)*Q[s,a] + alpha*(reward + gamma*np.max(Q[s1,:]))
            rAll += reward
            s = s1
            if done == True:
                break
        rev_list.append(rAll)
        if visuaDisplay:
            env.render()
    print( "Reward Sum on all epochs " + str(sum(rev_list)/epochs) )
    print( "Final Values Q-Table" )
    #np.set_printoptions(threshold=sys.maxsize)
    print( Q )
    print( Q.shape )

def evaluateQTable(env, Q, numTests):
    totalOfRewards = 0
    for i in range(numTests):
        observation = env.reset()
        totalReward = 0 
        done = False
        while not done:
            action = np.argmax(Q[observation,:])
            observation, reward, done, _ = env.step(action)
            totalReward += reward
        totalOfRewards += totalReward
    print(totalOfRewards/numTests)


def main():
    QTableName = "taxiAI_Q-table"
    env = gym.make('Taxi-v2')
    learnAndSave = False
    loadAndEvaluate = True

    if learnAndSave:
        Q = np.zeros( [ env.observation_space.n, env.action_space.n ] )
        learnQTable(env, Q, completionTries=300, alpha=.65, gamma=.9, epochs=500000, randomInfluence=3)
        learnQTable(env, Q, completionTries=100, alpha=.25, gamma=.9, epochs=10000, randomInfluence=.5)
        saveModel(Q, QTableName+".joblib")

    if loadAndEvaluate:
        Q = loadModel(QTableName+".joblib")
        evaluateQTable(env, Q, numTests=500000)


if __name__ == "__main__":
    main()