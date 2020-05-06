import gym, cv2
import numpy as np
from memory import RingBuffer

class Atari:
    def __init__(self, game):
        self.game = game
        self.env = gym.make(self.game)
        self.lives = None
        self.framesAfterDeath = None
        self.state = None

    def _processFrame(self, frame):
        # grayscale
        frame = np.mean(frame, axis=2).astype(np.uint8)
        # down sample
        frame =cv2.resize(frame, (84, 84))
        return frame

    def _getState(self):
        return np.dstack( [self.state[i] for i in range(4)] )

    def reset(self):
        frame = self.env.reset() 
        frame = self._processFrame( frame )
        self.state = RingBuffer(4)
        for _ in range(4):
            self.state.append(frame)
        self.lives = self.env.ale.lives()
        self.framesAfterDeath = 0
        return self._getState()

    def step(self, action):
        if None in [self.lives, self.framesAfterDeath]:
            raise RuntimeError("step called before reset")

        obs, reward, done, info = self.env.step(action)
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
    
    def getFramesAfterDeath(self):
        if self.framesAfterDeath is None:
            raise RuntimeError("info request before reset")
        return self.framesAfterDeath

    def getActionSpace(self):
        return self.env.action_space.n
    def getActionMeanings(self):
        return self.env.unwrapped.get_action_meanings()
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()
