import random, cv2
import gymnasium as gym
import numpy as np

class Atari:
    def __init__(self, game, seed=69, render=False):
        self.game = game
        if render:
            render_mode = "human"
        else:
            render_mode = None
        self.env = gym.make(self.game, render_mode=render_mode)
        self.lives = None
        self.framesAfterDeath = None
        self.episode = None
        self.score = None
        random.seed( seed )

    def reset(self):
        frame, _ = self.env.reset() 
        self.lives = self.env.ale.lives()
        self.framesAfterDeath = 0
        self.score = 0
        self.episode = [frame]
        return self._processFrame( frame )

    def step(self, action):
        if None in [self.lives, self.framesAfterDeath, self.score, self.episode]:
            raise RuntimeError("step called before reset")

        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.score += reward

        self.episode.append(state)
        state = self._processFrame(state)

        # assume no life has been lost
        self.framesAfterDeath += 1
        info["life_lost"] = False
        # but if a life has been lost
        if info["lives"] < self.lives:
            # update lives count, and frames after death
            self.lives = info["lives"]
            self.framesAfterDeath = 0
            info["life_lost"] = True
        # but if lives have been reset
        if info["lives"] > self.lives:
            # update lives count, and frames after death
            self.lives = info["lives"]
            self.framesAfterDeath = 0

        return state, np.sign(reward), done, info
    
    def _processFrame(self, frame):
        # grayscale
        frame = np.mean(frame, axis=2).astype(np.uint8)
        # down sample
        frame =cv2.resize(frame, (84, 84))
        return frame

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
        return self.score
    def getActionSpace(self):
        return self.env.action_space.n
    def getActionMeanings(self):
        return self.env.unwrapped.get_action_meanings()
    def close(self):
        self.env.close()
