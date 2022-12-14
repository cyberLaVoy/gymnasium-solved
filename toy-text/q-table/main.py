#!/usr/bin/env python3
import gymnasium as gym
import joblib
import numpy as np 


def saveModel(model, modelPath):
    joblib.dump(model, modelPath) 
def loadModel(modelPath):
    return joblib.load(modelPath)

def learnQTable(env, Q, alpha=.65, gamma=.9, epochs=100000):
    """
    Trains a Q-table using the Q-learning algorithm.
    Args:
        env: the game environment to learn the Q-table for
        Q: the initial Q-table to use for learning
        alpha: the learning rate, which determines the weight of new knowledge compared to existing knowledge
        gamma: the discount factor, which determines the importance of future rewards
        epochs: the number of iterations to run the learning algorithm for

    Returns:
        The trained Q-table.
    """
    # make a copy of the Q-table to use for training
    trainedQ = np.copy(Q)
    for episode in range(epochs):
        # Reset the environment and get the initial state
        currentState, _ = env.reset()
        # Set flags for whether the episode has terminated or been truncated
        terminated = False
        truncated = False
        # Run the Q-learning algorithm for this episode
        while not (terminated or truncated):
            # Choose an action using the Q-table, with a decreasing influence of randomness
            action = np.argmax(trainedQ[currentState,:] + np.random.randn(1,env.action_space.n)*(1/(episode+1))) 
            # Take the action and get the new state and reward
            newState,reward,terminated,truncated, _ = env.step(action)
            # Update Q-Table with new knowledge
            trainedQ[currentState, action] = (1-alpha)*trainedQ[currentState, action] + alpha*(reward + gamma*np.max(trainedQ[newState,:]))
            # Set the current state to the new state
            currentState = newState

    return trainedQ

def evaluateQTable(env, Q, numTests):
    for _ in range(numTests):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = np.argmax(Q[observation,:])
            observation,_,terminated,truncated, _ = env.step(action)

def main():
    # Game options
    games = [
        "CliffWalking-v0", # 0
        "FrozenLake-v1", # 1
        "Taxi-v3", # 2
        #"Blackjack-v1", # 3
    ]

    # Choose the game
    gameChoice = games[2]
    QTableName = gameChoice + "_Q-table"

    # Choose operations
    learnAndSave = True
    loadAndEvaluate = True

    if learnAndSave:
        env = gym.make(gameChoice)
        Q = np.zeros( [ env.observation_space.n, env.action_space.n ] )
        learnedQ = learnQTable(env, Q)
        saveModel(learnedQ, QTableName+".joblib")

    if loadAndEvaluate:
        env = gym.make(gameChoice, render_mode="human")
        Q = loadModel(QTableName+".joblib")
        evaluateQTable(env, Q, numTests=100)


if __name__ == "__main__":
    main()