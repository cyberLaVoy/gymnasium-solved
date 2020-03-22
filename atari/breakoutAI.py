from memory import ReplayMemory
import gym, time, cv2, os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda
from tensorflow.keras.initializers import VarianceScaling
import tensorflow.keras as keras

# actions: 0 - do nothing; 1 - start game; 2 - move right; 3 - move left 

MEM_ITEMS_COUNT = 1000000
REPLAY_START = 5000
BATCH_SIZE = 32
GAMMA = .99
RANDOMNESS_CONVERGENCE_ITER = 50000
TRAINING_ITER = 2000
TARGET_UPDATE_ITER = 5000
MAX_DO_NOTHING = 30

class TargetNetwork():
    def __init__(self, model, name="target_network"):
        self.name = name + ".h5"
        if os.path.exists(self.name):
            self.model = load_model(self.name)
        else:
            self.model = model
    def __del__(self):
        print("Saving target netwok...")
        self.model.save(self.name)
    def update(self, newModel):
        newModel.save(self.name)
        self.model = load_model(self.name)
    def predict(self, state):
        return self.model.predict(state)


def saveModel(model, fileName):
    model.save(fileName + ".h5")
def loadModel(fileName):
    return load_model(fileName + ".h5")

def getStartState(env):
    f0 = convertFrame( env.reset() )
    startState = f0
    for _ in range(3):
        startState = np.add(startState, f0)
        startState = np.floor_divide(startState, 2)
    return np.array([startState])
  
def createModel():
    initializer = VarianceScaling(scale=2)
    framesIn = Input((80,80,1), name="frames")
    actionsIn = Input((4,), name="mask")

    norm = Lambda(lambda x: x / 255.0)(framesIn)
    conv1 = Conv2D(16, 8, strides=4, activation="relu", kernel_initializer=initializer)(norm)
    conv2 = Conv2D(8, 4, strides=2, activation="relu", kernel_initializer=initializer)(conv1)
    flat = Flatten()(conv2)
    h0 = Dense(32, activation="relu", kernel_initializer=initializer)(flat)
    h1 = Dense(32, activation="relu", kernel_initializer=initializer)(h0)
    h2 = Dense(32, activation="relu", kernel_initializer=initializer)(h1)
    out = Dense(4, activation="linear")(h2)

    filtered = keras.layers.multiply([out, actionsIn])
    model = Model(inputs=[framesIn, actionsIn], outputs=filtered)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss="mse")
    return model

def fitBatch(model, targetModel, startStates, actions, rewards, nextStates, isTerminal, gamma=GAMMA):
    # predict q-values for any action, on states after action
    nextQvalues = targetModel.predict([nextStates, np.ones(actions.shape)])
    # zero out predicted q-values for all terminal states
    nextQvalues[isTerminal] = 0
    # set expected q-values to be the actual rewards plus gamma percent of best predicted q-values
    expectedQvalues = rewards + gamma*np.max(nextQvalues, axis=1)
    expectedQvalues = actions*expectedQvalues[:, None]
    model.fit([startStates, actions], expectedQvalues, batch_size=len(startStates), verbose=0)

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
    action = 0
    for i in range(7):
        f1, r1, done, info = env.step(action)
        r0 += r1
        if i % 2 == 0:
            f0 = np.add(f0, convertFrame(f1))
            f0 = np.floor_divide(f0, 2)
        else:
            action = a
    return np.array([f0]), np.sign(r0), done, info

def getEpsilon(i, convergence=RANDOMNESS_CONVERGENCE_ITER):
    if i >= convergence:
        return .1
    else:
        return 1 - .9*i/convergence

def getAction(env, model, currState, actionCount):
    actionMask = np.ones((1,4))
    epsilon = getEpsilon(actionCount)
    if np.random.random() <= epsilon:
        a = env.action_space.sample()
    else:
        a = np.argmax( model.predict( [currState, actionMask] ) )
    return a

def unpackSample(sample):
    startStates = np.array([ item[0] for item in sample])
    actions = np.array([ item[1] for item in sample]).astype(np.uint8)
    rewards = np.array([ item[2] for item in sample])
    nextStates = np.array([ item[3] for item in sample])
    isTerminal = np.array([ item[4] for item in sample])
    onehotActions = np.zeros((actions.size, 4)).astype(np.uint8)
    onehotActions[np.arange(actions.size), actions] = 1
    return startStates, onehotActions, rewards, nextStates, isTerminal

def learn(model, targetModel, memory, actionCount, batchSize=BATCH_SIZE):
    batch = memory.sample(batchSize)
    startStates, actions, rewards, nextStates, isTerminal = unpackSample(batch)
    fitBatch(model, targetModel, startStates, actions, rewards, nextStates, isTerminal)

def buildReplay(env, memory, amount=REPLAY_START):
    iteration = 0
    while iteration < amount:
        currState = getStartState(env)
        done = False
        while not done:
            a = env.action_space.sample()
            nextState, r, done, _ = gatherFrames(env, a)
            if done:
                r = 0
            memory.append( (currState[0], a, r, nextState[0], done) )
            iteration += 1
        print("Iteration:", iteration, '/', amount)


def train(env, model, targetModel, memory, targetUpdate=TARGET_UPDATE_ITER, episodes=TRAINING_ITER, maxDoNothing=MAX_DO_NOTHING, render=True):
    actionCount = 0
    for i in range(1, episodes+1):
        print("Episode", str(i)+'/'+str(episodes))
        episodeReward = 0
        currState = getStartState(env)
        done = False
        doNothingCount = 0
        while not done:

            if render:
                env.render()

            a = getAction(env, model, currState, actionCount)
            if doNothingCount is not None and a == 0:
                doNothingCount += 1
                if doNothingCount >= maxDoNothing:
                    a = 1
                    doNothingCount = None
            actionCount += 1

            nextState, r, done, _ = gatherFrames(env, a)

            if done:
                r = -1

            currState = nextState

            episodeReward += r
            memory.append( (currState[0], a, r, nextState[0], done) )
            if actionCount % targetUpdate == 0:
                print("Updating target network...")
                targetModel.update(model)
            if actionCount % 4 == 0:
                learn(model, targetModel, memory, actionCount)
        print("Episode reward:", episodeReward, "Epsilon:", getEpsilon(actionCount))
    env.close()
    return model

def observeAgent(model, env):
    actionMask = np.ones((1,4))
    while True:
        env.reset()
        s,_,_,_ = gatherFrames(env, 1)
        while True:
            env.render()
            a = np.argmax( model.predict([s, actionMask]) )
            s, _, done, _ = gatherFrames(env, a) 
            time.sleep(.01)
            if done:
                break
    env.close()

def main():
    env = gym.make("BreakoutNoFrameskip-v4")
    modelName = "breakoutdeterministic-v4-nn"
    memory = ReplayMemory(MEM_ITEMS_COUNT)

    saveReplay = False
    trainAndSave = True
    loadAndObserve = False

    if saveReplay:
        buildReplay(env, memory)
        memory.save()

    if trainAndSave:
        model = createModel()
        targetModel = TargetNetwork(model)
        memory.load()
        print(model.summary())
        print("Training model...")
        train(env, model, targetModel, memory, render=True)
        saveModel(model, modelName)

    if loadAndObserve:
        model = loadModel(modelName)
        observeAgent(model, env)

if __name__ == "__main__":
    main()