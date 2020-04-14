import gym
import numpy as np

class Atari:
    def __init__(self, game):
        self.env = gym.make(game)
        self.lives = None
        self.framesAfterDeath = None

    def reset(self):
        self.lives = self.env.ale.lives()
        self.framesAfterDeath = 0
        return self._processFrame( self.env.reset() )

    def step(self, action):
        if self.lives is None or self.framesAfterDeath is None:
            raise RuntimeError("step called before reset")

        state, reward, done, info = self.env.step(action)
        state = self._processFrame(state)

        self.framesAfterDeath += 1
        if info["ale.lives"] < self.lives:
            info["life_lost"] = True
            self.lives = info["ale.lives"]
            self.framesAfterDeath = 0
        else:
            info["life_lost"] = False

        return state, np.sign(reward), done, info
    
    def _processFrame(self, frame):
        # down sample
        frame = frame[::2, ::2]
        # grayscale
        frame = np.mean(frame, axis=2).astype(np.uint8)
        return frame

    def getFramesAfterDeath(self):
        if self.framesAfterDeath is None:
            raise RuntimeError("info request before reset")
        return self.framesAfterDeath

    def getActionSpace(self):
        return self.env.action_space.n
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()
