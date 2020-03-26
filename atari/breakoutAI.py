from memory import Top100, ReplayMemory
import gym, time, cv2, os, random
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.initializers import VarianceScaling
import tensorflow.keras as keras
import tensorflow as tf

# actions: 0 - do nothing; 1 - start game; 2 - move right; 3 - move left 

REPLAY_START = 8
BATCH_SIZE = 32
GAMMA = .99
TRAINING_ITER = 10000

def saveModel(model, fileName):
    model.save(fileName + ".h5")
def loadModel(fileName):
    return load_model(fileName + ".h5")

def getStartState(env):
    f0 = convertFrame( env.reset() )
    return np.array([f0])
  
def createModel():
    initializer = VarianceScaling()
    model = Sequential()
    model.add( Input((80,80,1)))
    model.add( Conv2D(16, 8, strides=4, activation="relu", kernel_initializer=initializer) )
    model.add( Conv2D(32, 4, strides=2, activation="relu", kernel_initializer=initializer) )
    model.add( Flatten() )
    model.add( Dense(64, activation="sigmoid", kernel_initializer=initializer) )
    model.add( Dense(64, activation="sigmoid", kernel_initializer=initializer) )
    model.add( Dense(4, activation="linear") )
    model.compile(optimizer="nadam", loss="huber_loss")
    return model

def fitBatch(model, startStates, actions, rewards, nextStates, isTerminal, gamma=GAMMA):
    nextQvalues = model.predict(nextStates)
    M = gamma*np.max(nextQvalues, axis=1)
    M[isTerminal] = 0
    nextQvalues[actions] = rewards + M
    model.fit(startStates, nextQvalues, batch_size=len(startStates), verbose=0)

def displayFrames(f):
    f = np.reshape(f, (80,80))
    f = f / 255
    plt.imshow(f, cmap="gray")
    plt.show()

def convertFrame(f):
    f = np.mean(f, axis=2).astype(np.uint8)
    f = f[35:195:, ::]
    f = cv2.resize(f, dsize=(80, 80), interpolation=cv2.INTER_NEAREST)
    f = np.reshape(f, (80,80,1))
    return f

def gatherFrames(env, a):
    f0, r0, done, info = env.step(a)
    f0 = convertFrame(f0)
    for _ in range(3):
        f1, r1, done, info = env.step(a)
        r0 += r1
        f0 = np.add(f0, convertFrame(f1))
        f0 = np.floor_divide(f0, 2)
    #displayFrames(f0)
    return np.array([f0]), np.sign(r0), done, info

def unpackSample(sample):
    startStates = np.array([ item[0] for item in sample])
    startStates = startStates / np.max(startStates)
    nextStates = np.array([ item[3] for item in sample])
    nextStates = nextStates / np.max(startStates)
    actions = np.array([ item[1] for item in sample])
    onehotActions = np.zeros((actions.size, 4)).astype(np.bool)
    onehotActions[np.arange(actions.size), actions] = 1
    rewards = np.array([ item[2] for item in sample])
    isTerminal = np.array([ item[4] for item in sample]).astype(np.bool)
    return startStates, onehotActions, rewards, nextStates, isTerminal

def learn(model, batch):
    startStates, actions, rewards, nextStates, isTerminal = unpackSample(batch)
    fitBatch(model, startStates, actions, rewards, nextStates, isTerminal)

def train(env, model, memory, top, trainingIter=TRAINING_ITER, replayStart=REPLAY_START, render=True):
    actionCount = 0
    episodes = replayStart + trainingIter
    epsilon = 1
    for eps in range(1, episodes+1):
        episodeReward, done = 0, False
        currState = getStartState(env)
        epsMem = []

        while not done:
            if render:
                env.render()

            ### step
            if np.random.random() <= epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax( model.predict( currState ) )
            nextState, r, done, _ = gatherFrames(env, a)
            actionCount += 1
            
            ### add to episode memory
            epsMem.append( (currState[0], a, r, nextState[0], done) )

            ### upkeep for next step 
            episodeReward += r
            currState = nextState

        ### add to global memory
        if episodeReward != 0:
            for mem in epsMem:
                memory.append( mem )
            top.append( (episodeReward, epsMem) )

        ### learn
        if eps >= replayStart:
            epsilon = max(.1, epsilon*.9995)

            _, episode = top.rand()
            batch = random.sample(episode, 32)
            learn(model, batch)

            batch = memory.sample(32) 
            learn(model, batch)

        print("Episode " + str(eps)+'/'+str(episodes) +" reward:", episodeReward, "Randomness:", epsilon)

    env.close()
    return model

def observeAgent(model, env):
    while True:
        env.reset()
        s = getStartState(env)
        while True:
            env.render()
            if np.random.random() <= .05:
                a = env.action_space.sample()
            else:
                a = np.argmax( model.predict( s ) )
            s, _, done, _ = gatherFrames(env, a) 
            time.sleep(.01)
            if done:
                break
    env.close()

def main():
    env = gym.make("BreakoutDeterministic-v4")
    modelName = "breakoutdeterministic-v4-nn"
    #env = gym.make("BreakoutNoFrameskip-v4")
    #modelName = "breakoutnoframeskip-v4-nn"

    memory = ReplayMemory(250000)
    top = Top100()

    trainAndSave = True
    loadAndObserve = False

    if trainAndSave:
        model = createModel()
        memory.load()
        print(model.summary())
        print("Training model...")
        train(env, model, memory, top, render=True)
        saveModel(model, modelName)

    if loadAndObserve:
        model = loadModel(modelName)
        observeAgent(model, env)

if __name__ == "__main__":
    main()