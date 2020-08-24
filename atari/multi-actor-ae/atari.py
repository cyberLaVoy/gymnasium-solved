import gym, cv2, psutil
import numpy as np
from memory import RingBuffer

class Atari:
    def __init__(self, game):
        self.game = game
        self.env = gym.make(self.game)
        self.lives = None
        self.framesAfterDeath = None
        self.state = None
        self.score = None
        self.episode = None

    def _processFrame(self, frame):
        # grayscale
        frame = np.mean(frame, axis=2)
        # down sample
        frame = cv2.resize(frame, (84, 84))
        # reshape 
        frame = np.reshape(frame, (84, 84, 1))
        return frame.astype(np.uint8)

    def _getState(self):
        return np.dstack( [self.state[i] for i in range(4)] )

    def getStateChange(self):
        change = self.state[3]-self.state[2]
        change = np.where(change > 0, 1, 0)
        return np.reshape( change, (84,84,1) ).astype(np.uint8)

    def reset(self):
        frame = self.env.reset() 
        self.episode = RingBuffer(512)
        self.episode.append(frame)
        frame = self._processFrame( frame )
        self.state = RingBuffer(4)
        for _ in range(4):
            self.state.append(frame)
        self.lives = self.env.ale.lives()
        self.framesAfterDeath = 0
        self.score = 0
        return self._getState()

    def step(self, action):
        if None in [self.lives, self.framesAfterDeath, self.score, self.episode]:
            raise RuntimeError("step called before reset")

        if action >= self.env.action_space.n: 
            action = 0
        obs, reward, done, info = self.env.step(action)
        self.score += reward
        self.episode.append(obs)
        self.state.append( self._processFrame(obs) )

        # assume no life has been lost
        self.framesAfterDeath += 1
        info["life_lost"] = False
        # but if a life has been lost
        if info["ale.lives"] < self.lives:
            # update lives count, and frames after death
            self.lives = info["ale.lives"]
            self.framesAfterDeath = 0
            info["life_lost"] = True
        # but if lives have been reset
        if info["ale.lives"] > self.lives:
            # update lives count, and frames after death
            self.lives = info["ale.lives"]
            self.framesAfterDeath = 0

        return self._getState(), np.sign(reward), done, info
    
    def saveEpisode(self):
        if self.episode is None:
            raise RuntimeError("no episode to save")
        pathOut = self.game + "_best_episode.avi"
        fps = 35.0
        height, width, _ = self.episode[0].shape
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(pathOut, fourcc, fps, size)
        for i in range(len(self.episode)):
            frame = cv2.cvtColor(self.episode[i], cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()
        print("Episode saved.")

    def getFramesAfterDeath(self):
        if self.framesAfterDeath is None:
            raise RuntimeError("info request before reset")
        return self.framesAfterDeath
    def getScore(self):
        if self.score is None:
            raise RuntimeError("info request before reset")
        return int( self.score )
    def getActionSpace(self):
        return self.env.action_space.n
    def getActionMeanings(self):
        return self.env.unwrapped.get_action_meanings()

    def render(self):
        self.env.render()
    def close(self):
        self.env.close()
