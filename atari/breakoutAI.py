import gym, time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

def saveModel(model, fileName):
    model.save(fileName + ".h5")
def loadModel(fileName):
    return load_model(fileName + ".h5")
  

def createNeuralNetwork():
    model = Sequential()
    model.add( Dense( 16, input_dim=25200, activation="relu" ) ) # input_dim defines how many input values
    model.add( Dense( 32, activation="relu" ) )
    model.add( Dense( 64, activation="linear" ) )
    model.add( Dense( 64, activation="linear" ) )
    model.add( Dense( 4, activation="linear" ) )
    model.compile( loss="mean_squared_error", optimizer="adam" )
    return model

def convertFrame(frame):
    frame = np.mean(frame, axis=2)
    frame = Image.fromarray(frame)
    frame = frame.resize((80, 105))
    #plt.imshow(frame, cmap="gray")
    #plt.show()
    return np.array(frame)

def gather3frames(env, a):
    frames = []
    for _ in range(3):
        frame, r, done, info = env.step(a)
        frame = convertFrame(frame)
        frames.append(frame)
    obs = np.reshape(frames, (1, 25200))
    return obs, r, done, info

    


def trainNeuralNetwork(env, model, epochs, render=True):
    gamma = .95
    epsilon = .5 # starting probability of performing random action
    decay = .99 # decay per epoch, of epsilon probablity
    for i in range(1, epochs+1):
        print("Epoch", str(i)+'/'+str(epochs))
        epsilon *= decay
        epochReward = 0
        env.reset()
        a = env.action_space.sample()
        currState = gather3frames(env, a)

        while True:
            if render:
                env.render()
            if np.random.random() <= epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax( model.predict(currState) )
            nextState, r, done, info = gather3frames(env, a)

            if info["ale.lives"] == 0:
                r = -1

            # the predicted reward (before action)
            targetVector = model.predict(currState)
            if done:
                targetVector[0][a] = r
            else:
                # set the actual reward + a portial on the predicted reward (after action)
                targetVector[0][a] = r + gamma*np.max( model.predict(nextState) )
            # fit model to target reward
            model.fit(currState, targetVector, verbose=0)

            if done:
                break
            currState = nextState
            epochReward += 1
        print("Epoch reward:", epochReward, "Randomness:", epsilon)
    env.close()
    return model

def observeAgent(model, env):
    while True:
        env.reset()
        a = env.action_space.sample()
        s = gather3frames(env, a)
        while True:
            env.render()
            a = np.argmax( model.predict(s) )
            print(model.predict(s))
            s, _, done, _ = gather3frames(env, a) 
            time.sleep(.01)
            if done:
                break
    env.close()

def main():
    env = gym.make("BreakoutNoFrameskip-v4")

    modelName = "breakoutnoframeskip-v4-nn"

    trainAndSave = True
    loadAndObserve = False

    if trainAndSave:
        model = createNeuralNetwork()
        trainNeuralNetwork(env, model, 50, True)
        saveModel(model, modelName)

    if loadAndObserve:
        model = loadModel(modelName)
        observeAgent(model, env)



main()