import gym, math
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time


def saveModel(model, fileName):
    model.save(fileName + ".h5")

def loadModel(fileName):
    return load_model(fileName + ".h5")
    

def createNeuralNetwork():
    model = Sequential()
    model.add( Dense( 16, input_dim=4, activation="relu" ) ) # input_dim defines how many input values
    model.add( Dense( 64, activation="relu" ) )
    model.add( Dense( 32, activation="relu" ) )
    model.add( Dense( 32, activation="linear" ) )
    model.add( Dense( 2, activation="linear" ) )
    model.compile( loss="mean_squared_error", optimizer="adam" )
    return model

def trainNeuralNetwork(env, model, epochs, render=True):
    gamma = .95
    epsilon = .9 # starting probability of performing random action
    decay = .99 # decay per epoch, of epsilon probablity
    for i in range(1, epochs+1):
        print("Epoch", str(i)+'/'+str(epochs))
        currState = np.reshape(env.reset(), [1, 4])
        epsilon *= decay
        epochReward = 0
        done = False
        while True:
            if render:
                env.render()
            if np.random.random() <= epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax( model.predict(currState) )
            nextState, r, _, _ = env.step(a)

            # reward engineering 
            r = 0
            if abs(nextState[0]) < 1.6 and abs(nextState[2]) < 3*math.pi/180:
                r = 1
            if abs(nextState[0]) > 2.4 or abs(nextState[2]) > 18*math.pi/180:
                r = -1
                done = True

            nextState = np.reshape(nextState, [1, 4])

            # the predicted reward (before action)
            targetVector = model.predict(currState)
            if done:
                targetVector[0][a] = r
            else:
                # set the actual reward + a portial on the predicted reward (after action)
                targetVector[0][a] = r + gamma*np.max( model.predict(nextState) )
            # fit model to target reward
            model.fit(currState, targetVector, epochs=3, verbose=0)

            if done:
                break
            currState = nextState
            epochReward += 1
        print("Epoch reward:", epochReward, "Randomness:", epsilon)
    env.close()
    return model


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

def observeAgent(model, env):
    while True:
        s = env.reset()
        while True:
            env.render()
            s = np.reshape(s, [1, 4]) #reshape the state vector to be correct for tensorflow's format
            a = np.argmax( model.predict(s) )
            s, _, _, _ = env.step(a)
            if abs(s[0]) > 2.4 or abs(s[2]) > 18*math.pi/180:
                break
    env.close()

def main():
    env = gym.make('CartPole-v0')

    modelName = "cartPole-v0-nn"
    trainAndSave = False
    loadAndTest = True
    loadAndObserve = False

    if trainAndSave:
        model = createNeuralNetwork()
        trainNeuralNetwork(env, model, 500, False)
        saveModel(model, modelName)
    
    if loadAndTest:
        model = loadModel(modelName)
        result = testModel(model, env, 100, True)
        print(result)
    
    if loadAndObserve:
        model = loadModel(modelName)
        observeAgent(model, env)


if __name__ == "__main__":
    main()
