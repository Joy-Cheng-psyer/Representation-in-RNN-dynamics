#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auditory tone detection task."""

import numpy as np
from gym import spaces

import neurogym as ngym
import pdb##设置断点

class ToneDetection(ngym.TrialEnv):
    '''
    By Ru-Yuan Zhang (ruyuanzhang@gmail.com)
    By Xiangbin Teng (xiangbin.teng@gmail.com) - 1) add pre-onset period; 2) make the noise masker more acoustics based; 3) Change of the task from indicating positions of tone into yes/no

    Human subjects are asked to report whether they hear a pure tone in a piece of white noise by pressing one of two buttons to indicate yes or no. 
    It is a yes-or-no task without requirement for indicating the position of the tone. 

    The tone lasts 30ms and is inserted either at the 500ms, 1000ms, or 1500ms of a 2-s white noise.

    Note in this version we did not consider the fixation period as we mainly aim to model human data. 
    
    

    Args:
        <dt>: milliseconds, delta time,
        <sigma>: float, input noise level, control the task difficulty
    '''

    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/jneuro/5/12/3261.full.pdf',
        'paper_name': 'Representation of Tones in Noise in the Responses of Auditory Nerve Fibers  in Cats',
        'tags': ['auditory', 'perceptual', 'supervised', 'decision']
    }

    def __init__(self, dt=10, sigma=0.1, timing=None, tone_mag=0):
        super().__init__(dt=dt)
        pdb.set_trace()

        self.sigma = sigma   # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'noresp': -0.1}  # need to change here

        self.timing = {
            'fixation': 500,
            'stimulus': 2000
            }
        if timing:
            self.timing.update(timing)
        
        self.toneTiming = [500, 1000, 1500] # ms, the onset times of a tone
        self.toneDur = 30 # ms, the duration of a tone

        self.toneTimingIdx = [int(i / self.dt) for i in self.toneTiming]
        self.stimArray = np.zeros(int(self.timing['stimulus']/self.dt))

        self.abort = False

        self.signals = np.linspace(0, 1, 5)[:-1] # signal strength
        self.conditions = [0, 1, 2, 3] # no tone, tone at position 1/2/3

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': 1}
        self.action_space = spaces.Discrete(2)
        self.act_dict = {'fixation': 0, 'choice': range(1, 5+1)}
        self.tone_mag = tone_mag



    def _new_trial(self, condition=None):
        '''
        <condition>: int (0/1/2/3), indicate no tone, tone at position 1/2/3
        '''
        if condition is None:
            condition = self.rng.choice(self.conditions)

        # Trial info
        

        truth = 1 if condition>0 else 0
        trial = {
            'ground_truth':   truth,
            'condition': condition
        }

        # generate tone stimulus
        stim = self.stimArray.copy()
        stim += 1
        if condition != 0:
            #stim[self.toneTimingIdx[condition-1]-1] = 0.4672 * 10**snr
            stim[self.toneTimingIdx[condition-1]] += 10**self.tone_mag
            #stim[self.toneTimingIdx[condition-1]+1] = 0.4672 * 10**snr

        ground_truth = trial['ground_truth']

        # Periods
        self.add_period(['fixation','stimulus'])

        # Observations
        
        # generate stim input
        # define stimulus
        stim = stim[:, np.newaxis] # stimulus must be at least two dimension with the 1st dimen as seq_len

        self.add_ob(stim, 'stimulus')
        self.add_randn(0, self.sigma, 'stimulus') # add input noise
        self.add_randn(0, 0.05, 'fixation') # add input noise
        
        # Ground truth
        self.set_groundtruth(ground_truth)

        return trial

    def _step(self, action):
        """
        In this tone detection task, no need to define reward step function, just output the final choice.
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # # observations
        # if self.in_period('stimulus'): # start a new trial once step into decision stage       
        #          new_trial = True
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = ToneDetection(dt=10, timing=None)
    ngym.utils.plot_env(env, num_steps=20, def_act=1)
    # env = PerceptualDecisionMakingDelayResponse()
    # ngym.utils.plot_env(env, num_steps=100, def_act=1)
