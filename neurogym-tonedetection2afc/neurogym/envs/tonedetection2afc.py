#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auditory tone detection task."""

import numpy as np
from gym import spaces
from scipy.stats import norm
from neurogym.envs.distribution.tone_distribution import*
import neurogym as ngym





class ToneDetection2AFC(ngym.TrialEnv):
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

    def __init__(self, dt=10, sigma=0.1, timing=None, tone_mag=0, p=0.5, distribution='uniform',onset_u1=1000,onset_sig1=100,onset_alpha=1,onset_beta=10,onset_u2=[500,1000],onset_sig2=[100,100],onset_u3=[500,1000,1500],onset_sig3=[100,100,100]):

        super().__init__(dt=dt)

        self.sigma = sigma   # Input noise
        self.p = p  # The ratio between period1 and period2
        
        self.distribution = distribution # distribution

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}  #changed


        self.timing = {
            'fixation': 500,
            'stimulus1': 2000,
            'delay': 500,
            'stimulus2': 2000
            }
        if timing:
            self.timing.update(timing)
            
        self.abort = False
        
        self.toneDur = 30 # ms, the duration of a tone
        
        ## Distribution
        ## Setting a distribution of stim onset 
        ### Getting timepoints for the two periods, respectively
        if self.distribution == 'uniform':
            tonelen = self.timing['stimulus1']/self.dt
            self.toneTimingIdx1, self.toneTimingIdx2 = uniform_distribution(tonelen)
            
        elif self.distribution == 'normal':
            tonelen = self.timing['stimulus1']/self.dt
            u1=onset_u1/self.dt
            sigma1=onset_sig1/self.dt
            self.toneTimingIdx1, self.toneTimingIdx2 = normal_distribution(tonelen,u1,sigma1)
            
        #elif self.distribution == 'gamma':
                    
        elif self.distribution == 'log normal':            
            tonelen = self.timing['stimulus1']/self.dt
            u1=onset_u1/self.dt
            sigma1=onset_sig1/self.dt
            self.toneTimingIdx1, self.toneTimingIdx2 = lognormal_distribution(tonelen,u1,sigma1)
            
        elif self.distribution == 'bi-modal':
            tonelen = self.timing['stimulus1']/self.dt
            u1,u2 = onset_u2[0]/self.dt,onset_u2[1]/self.dt
            sigma1,sigma2 = onset_sig2[0]/self.dt,onset_sig2[1]/self.dt
            self.toneTimingIdx1, self.toneTimingIdx2 = bimodal_distribution(tonelen,u1,sigma1,u2,sigma2)
            
        elif self.distribution == 'tri-modal':
            tonelen = self.timing['stimulus1']/self.dt
            u1,u2,u3 = onset_u3[0]/self.dt,onset_u3[1]/self.dt,onset_u3[2]/self.dt
            sigma1,sigma2,sigma3 = onset_sig3[0]/self.dt,onset_sig3[1]/self.dt,onset_sig3[2]/self.dt
            self.toneTimingIdx1, self.toneTimingIdx2 = trimodal_distribution(tonelen,u1,sigma1,u2,sigma2,u3,sigma3)
                
            
        
        # Defining the input sequence of two periods
        self.stimArray1 = np.zeros(int(self.timing['stimulus1']/self.dt))
        self.stimArray2 = np.zeros(int(self.timing['stimulus2']/self.dt))


        self.conditions = [1, 2] # tone at period1 or period2

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32)        
        self.ob_dict = {'fixation': 0, 'stimulus1': 1, 'stimulus2': 2}
        
        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0, 'choice':[1, 2]}
        self.tone_mag = tone_mag




    def _new_trial(self, condition=None):
        '''
        <condition>: int (0/1/2/3), indicate no tone, tone at position 1/2/3
        '''
        ##Adjusting the probability of the tone appearing at both periods
        if condition is None:
            prob1 = np.random.uniform()
            if prob1 < self.p:
                condition = self.conditions[0]
            else:
                condition = self.conditions[1]


        # Trial info
        
        truth0 = condition # trial condition
        truth1 = self.stimArray1.copy() # continuous trial condition
        truth2 = self.stimArray2.copy() # continuous trial condition
        if condition ==1:
            truth1[self.toneTimingIdx1:] += condition #eg. [0 0 0 0 0 1 1 1 1 1,,,]
        else:
            truth2[self.toneTimingIdx2:] += condition #eg. [0 0 0 0,,, 2 2 2 2 2]

            
        trial = {
            'ground_truth':   truth0,
            'condition': condition,
            'stim1': truth1,
            'stim2': truth2,
            
        }

        # generate tone stimulus
        stim1 = self.stimArray1.copy()
        stim1 += 1
        
        stim2 = self.stimArray2.copy()
        stim2 += 1
        
        #Setting a exact timepoint based on the condition
        if condition == 1:
            stim1[self.toneTimingIdx1] += 10**self.tone_mag## stim2 is all 0
        else:
            stim2[self.toneTimingIdx2] += 10**self.tone_mag## stim1 is all 0

        ground_truth = trial['ground_truth']

        # Periods
        self.add_period(['fixation','stimulus1','delay','stimulus2'])

        # Observations
        
        # generate stim input
        # define stimulus

        stim1 = stim1[:, np.newaxis] # stimulus must be at least two dimension with the 1st dimen as seq_len
        stim2 = stim2[:, np.newaxis]
        self.add_ob(stim1, 'stimulus1')
        self.add_ob(stim2, 'stimulus2')
        self.add_randn(0, self.sigma, 'stimulus1') # add input noise
        self.add_randn(0, self.sigma, 'stimulus2')
        self.add_randn(0, 0.05, 'fixation') # add input noise
        self.add_randn(0, 0.05, 'delay')



        
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
        ob = self.ob_now
        
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
                
        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}


    
    
        
        
    


if __name__ == '__main__':
    env = NewToneDetection(dt=10, timing=None)
    ngym.utils.plot_env(env, num_steps=20, def_act=1)
    # env = PerceptualDecisionMakingDelayResponse()
    # ngym.utils.plot_env(env, num_steps=100, def_act=1)
