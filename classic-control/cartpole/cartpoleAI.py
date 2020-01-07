import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def createNeuralNetwork():
    model = Sequential()
    model.add( Dense( 48, input_dim=4, activation="sigmoid" ) ) # input_dim defines how many input values
    model.add( Dense( 48, activation="sigmoid" ) )
    model.add( Dense( 48, activation="sigmoid" ) )
    model.add( Dense( 2, activation="softmax" ) )
    model.compile( loss="mean_squared_error", optimizer="adam" )
    return model

def trainNeuralNetwork(env, model, epochs, render=True):
    alpha = .6 # how much current reward counts, vs predicted award (after action)
    epsilon = .8 # starting probability of performing random action
    decay = .999 # decay per epoch, of epsilon probablity
    for _ in range(epochs):
        currState = np.reshape(env.reset(), [1, 4])
        epsilon *= decay
        done = False
        epochReward = 0
        while not done:
            if render:
                env.render()
            if np.random.random() <= epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax( model.predict(currState) )
            nextState, r, done, _ = env.step(a)
            nextState = np.reshape(nextState, [1, 4])

            # the predicted reward (before action)
            targetVector = model.predict(currState)
            targetVector[0][a] = alpha*r + (1-alpha)*np.max( model.predict(nextState) )
            # fit model to target reward
            model.fit(currState, targetVector, epochs=1, verbose=0)

            currState = nextState
            if done:
                print("Epoch reward:", epochReward, "Randomness:", epsilon)
            else:
                epochReward += 1
    env.close()


def testModel(model, env, trials, render=True):
    totalRewards = 0
    for _ in range( trials ):
        s = env.reset()
        done = False
        trialReward = 0
        while not done:
            if render:
                env.render()
            s = np.reshape(s, [1, 4]) #reshape the state vector to be correct for tensorflow's format
            a = np.argmax( model.predict(s) )
            s, r, done, _ = env.step(a)
            trialReward += r 
        totalRewards += trialReward
    env.close( )
    return totalRewards/trials

def main():
    env = gym.make('CartPole-v0')
    model = createNeuralNetwork()
    trainNeuralNetwork(env, model, 500, False)
    result = testModel(model, env, 100)
    print(result)

if __name__ == "__main__":
    main()
