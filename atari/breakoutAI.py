import gym, time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Lambda
import tensorflow.keras as keras
from random import randint, seed

MEM_ITEMS_COUNT = 1000000
REPLAY_START = 50000

class RingBuf:
    def __init__(self, size=MEM_ITEMS_COUNT):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class Memory:
    def __init__(self, capacity=REPLAY_START):
        self.data = {"start_states": RingBuf(), "actions" : RingBuf(), "rewards": RingBuf(), "next_states": RingBuf(), "is_terminal": RingBuf()}
        self.capactiy= capacity
    def sample(self, size):
        minimum = 0
        maximum = len(self.data["start_states"])
        indices = []
        for _ in range(size):
            indices.append(randint(minimum, maximum))
        collection = ( [self.data["start_states"][i] for i in indices], 
                       [self.data["actions"][i] for i in indices],
                       [self.data["rewards"][i] for i in indices],
                       [self.data["next_states"][i] for i in indices],
                       [self.data["is_terminal"][i] for i in indices] )
        return collection
    def add(self, startState, action, reward, nextState, isTerminal):
        self.data["start_states"].append(startState)
        self.data["actions"].append(action)
        self.data["rewards"].append(reward)
        self.data["next_states"].append(nextState)
        self.data["is_terminal"].append(isTerminal)
    def atCapacity(self): 
        return len(self.data["start_states"]) >= self.capactiy


# actions: 0 - do nothing; 1 - start game; 2 - move right; 3 - move left 

def saveModel(model, fileName):
    model.save(fileName + ".h5")
def loadModel(fileName):
    return load_model(fileName + ".h5")
  
def createModel():
    framesIn = Input((105,320,1), name="frames")
    actionsIn = Input((4,), name="mask")

    norm = Lambda(lambda x: x / 255.0)(framesIn)
    conv1 = Conv2D(16, 8, strides=4, activation="relu")(norm)
    conv2 = Conv2D(32, 4, strides=2, activation="relu")(conv1)
    flat = Flatten()(conv2)
    h0 = Dense(128, activation="relu")(flat)
    h1 = Dense(128, activation="relu")(h0)
    out = Dense(4, activation="softmax")(h1)

    filtered = keras.layers.multiply([out, actionsIn])
    model = Model(inputs=[framesIn, actionsIn], outputs=filtered)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss="mse")
    return model

def fitBatch(model, startStates, actions, rewards, nextStates, isTerminal):
    gamma = .99

    startStates = np.array(startStates)
    actions = np.array(actions)
    rewards = np.array(rewards)
    nextStates = np.array(nextStates)
    isTerminal = np.array(isTerminal)

    nextQvalues = model.predict([nextStates, np.ones(actions.shape)])
    nextQvalues[isTerminal] = 0
    qValues = rewards + gamma*np.max(nextQvalues, axis=1)
    model.fit([startStates, actions], actions*qValues[:, None], batch_size=len(startStates), verbose=0)

def convertFrame(f):
    f = np.mean(f, axis=2).astype(np.uint8)
    f = f[::2, ::2]
    f = np.reshape(f, (105,80,1))
    #plt.imshow(f, cmap="gray")
    #plt.show()
    return f

def gatherFrames(env, a, reset=False):
    if reset:
        f0 = convertFrame(env.reset())
        f1 = f0.copy()
        for _ in range(3):
            f0 = np.concatenate((f0, f1), axis=1)
        return np.array([f0])
    f0, r0, done, info = env.step(a)
    f0 = convertFrame(f0)
    for _ in range(3):
        f1, r1, done, info = env.step(a)
        r0 += r1
        f0 = np.concatenate((f0, convertFrame(f1)), axis=1)
    #plt.imshow(f0, cmap="gray")
    #plt.show()
    return np.array([f0]), np.sign(r0), done, info

def getEpsilon(i):
    e0 = 1 # starting probability of performing random action
    return max(e0-(.9*i/250), .1)

def learn(model, memory):
    batch = memory.sample(32)
    startStates, actions, rewards, nextStates, isTerminal = batch
    fitBatch(model, startStates, actions, rewards, nextStates, isTerminal)

def buildReplay(env, memory):
    iteration = 0
    while not memory.atCapacity():
        currState = gatherFrames(env, None, True)
        while True:
            a = env.action_space.sample()
            nextState, r, done, _ = gatherFrames(env, a)
            action = [0]*4
            action[a] = 1
            memory.add(currState[0], action, r, nextState[0], done)
            iteration += 1
            if done:
                break
        print(iteration)


def train(env, model, memory, epochs, render=True):
    actionMask = np.ones((1,4))
    for i in range(1, epochs+1):
        print("Epoch", str(i)+'/'+str(epochs))
        epochReward = 0
        currState = gatherFrames(env, None, True)
        epsilon = getEpsilon(i)
        while True:
            if render:
                env.render()

            if np.random.random() <= epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax( model.predict( [currState, actionMask] ) )
            nextState, r, done, _ = gatherFrames(env, a)

            action = [0]*4
            action[a] = 1
            memory.add(currState[0], action, r, nextState[0], done)
            epochReward += r
            if done:
                break
            currState = nextState
        if i % 4 == 0:
            learn(model, memory)
        print("Epoch reward:", epochReward, "Randomness:", epsilon)
    env.close()
    return model

def observeAgent(model, env):
    actionMask = np.ones((1,4))
    while True:
        s = gatherFrames(env, None, True)
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
    modelName = "breakoutnoframeskip-v4-nn"
    memory = Memory()

    trainAndSave = True
    loadAndObserve = False

    seed(69)

    if trainAndSave:
        model = createModel()
        print(model.summary())
        print("Building replay...")
        buildReplay(env, memory)
        print("Training model...")
        train(env, model, memory, 500, True)
        saveModel(model, modelName)

    if loadAndObserve:
        model = loadModel(modelName)
        observeAgent(model, env)


main()