import numpy as np
import cv2
import gym


class Preprocess():
    def __init__(self, size=32, stacks=4, return_seq=False, RGB=False):
        self.channels=3 if RGB else 1
        self.hist=np.zeros((stacks,size,size,self.channels))
        self.size=size
        self.stacks=stacks
        self.return_seq=return_seq
        self.RGB=RGB

    def reset(self):
        self.hist=np.zeros((self.stacks,self.size,self.size,self.channels))

    def __call__(self, img):
        resized = cv2.resize(img, (self.size,self.size), interpolation = cv2.INTER_AREA)
        if not self.RGB:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            resized = np.reshape(resized, resized.shape + (1,))
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        resized     = np.divide(resized, 255)
        self.hist   = np.concatenate((self.hist[1:], [resized]), axis=0)
        
        if self.return_seq:
            return self.hist
        else:
            final_img = np.concatenate(self.hist, axis=-1)
            return final_img
            

class Pong():
    def __init__(self, img_size=32, stacks=4, skips=4, return_seq=False):
        self.env = gym.make('Pong-v0')
        self.preprocess = Preprocess(img_size, stacks, return_seq)
        self.skips=skips
        self.pong_action = { 0: 0, 1: 2, 2: 3 }
        self.action_space=self.env.action_space
        self.action_space.n=3
        self.observation_space=(img_size, img_size, stacks)

    def reset(self):
        self.preprocess.reset()
        s = self.env.reset()
        s = self.preprocess(s)
        return s

    def step(self, a):
        total_r=0
        for i in range(self.skips):
            self.env.render()

            n_s, r, done, info = self.env.step(self.pong_action[a])
            n_s = self.preprocess(n_s)
            total_r+=r

            if done: break
            
        return n_s, total_r, done, info
    
    
class Atari():
    def __init__(self, env, do_preprocess=False, img_size=64, stacks=4, skips=1, return_seq=False, RGB=False):
        self.env = gym.make(env)
        self.action_space      = self.env.action_space
        self.observation_space = self.env.observation_space.shape
        self.skips             = skips
        self.preprocess        = Preprocess(img_size, stacks, return_seq, RGB=RGB)
        self.do_preprocess     = do_preprocess
        if do_preprocess:
            self.observation_space =(img_size, img_size, stacks)
            self.skips             = skips

    def reset(self):
        s = self.env.reset()
        if self.do_preprocess:
            self.preprocess.reset()
            s = self.preprocess(s)
        return s

    def step(self, a):
        total_r=0
        for i in range(self.skips):
            self.env.render()

            n_s, r, done, info = self.env.step(a)
            total_r+=r

            if self.do_preprocess:
                n_s = self.preprocess(n_s)
            if done: break
        return n_s, total_r, done, info



class Mario():
    def __init__(self, img_size=32, stacks=4, skips=4, return_seq=False):
        
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = gym_super_mario_bros.make('SuperMarioBros-v2')
        
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.preprocess = Preprocess(img_size, stacks, return_seq)
        self.skips=skips
        self.action_space=self.env.action_space
        self.observation_space=(img_size, img_size, stacks)

    def reset(self):
        self.preprocess.reset()
        s = self.env.reset()
        s = self.preprocess(s)
        return s

    def step(self, a):
        total_r=0
        for i in range(self.skips):
            self.env.render()

            n_s, r, done, info = self.env.step(a)
            n_s = self.preprocess(n_s)
            total_r+=r

            if done: break
            
        return n_s, total_r, done, info


