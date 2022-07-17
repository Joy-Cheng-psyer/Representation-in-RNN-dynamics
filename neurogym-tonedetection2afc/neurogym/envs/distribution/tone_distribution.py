#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
from scipy.stats import lognorm

# Defining distribution function
    
def uniform_distribution(tonelen):
    #tonelen = self.timing['stimulus1']/self.dt # get the number of timepoints

    toneTimingIdx1 = int(np.random.uniform(0,tonelen))# get a timepoint subject to uniform distribution for the first period
    toneTimingIdx2 = int(np.random.uniform(0,tonelen))

    return toneTimingIdx1, toneTimingIdx2

def normal_distribution(tonelen,u,sigma):
    #tonelen = self.timing['stimulus1']/self.dt 
    #tonelen2 = self.timing['stimulus2']/self.dt

    #u = u/self.dt
    #sigma = sigma/self.dt

    norm_distribution = norm(loc = u, scale = sigma)

    x = np.arange(tonelen)

    discrete_pdf = norm_distribution.pdf(x)

    normalize_index = np.sum(discrete_pdf)
    normalized_discrete_pdf = discrete_pdf / normalize_index

    sample_list = []
    nsize= 10000
    for n in range(nsize):
        sample_list.append(np.random.choice(x,p=normalized_discrete_pdf))

    toneTimingIdx1 = int(np.random.choice(sample_list))
    toneTimingIdx2 = int(np.random.choice(sample_list))

    return toneTimingIdx1, toneTimingIdx2

def lognormal_distribution(tonelen, u, sigma):
    #tonelen = self.timing['stimulus1']/self.dt 
    #tonelen2 = self.timing['stimulus2']/self.dt

    #u = u/self.dt
    #sigma = sigma/self.dt

    lognorm_distribution = lognorm(s = sigma, loc = 0, scale=np.exp(u) )

    x = np.arange(tonelen)

    discrete_pdf = lognorm_distribution.pdf(x)

    normalize_index = np.sum(discrete_pdf)
    normalized_discrete_pdf = discrete_pdf / normalize_index

    sample_list = []
    nsize= 10000
    for n in range(nsize):
        sample_list.append(np.random.choice(x,p=normalized_discrete_pdf))

    toneTimingIdx1 = int(np.random.choice(sample_list))
    toneTimingIdx2 = int(np.random.choice(sample_list))

    return toneTimingIdx1, toneTimingIdx2
    

def bimodal_distribution(tonelen,u1,sigma1,u2,sigma2):
    #tonelen = self.timing['stimulus1']/self.dt 
    #tonelen2 = self.timing['stimulus2']/self.dt

    #u1,u2 = u2[0]/self.dt,u2[1]/self.dt
    #sigma1,sigma2 = sigma2[0]/self.dt,sigma2[1]/self.dt

    norm_distribution1 = norm(loc = u1, scale = sigma1)
    norm_distribution2 = norm(loc = u2, scale = sigma2)

    x = np.arange(tonelen)

    discrete_pdf1 = norm_distribution1.pdf(x)
    discrete_pdf2 = norm_distribution2.pdf(x)

    two_gaussian_distribution = discrete_pdf1 + discrete_pdf2
    normalize_index = np.sum(two_gaussian_distribution)
    normalized_two_gaussian_distribution = two_gaussian_distribution / normalize_index

    sample_list = []
    nsize= 10000
    for n in range(nsize):
        sample_list.append(np.random.choice(x,p=normalized_two_gaussian_distribution))

    toneTimingIdx1 = int(np.random.choice(sample_list))
    toneTimingIdx2 = int(np.random.choice(sample_list))

    return toneTimingIdx1, toneTimingIdx2

def trimodal_distribution(tonelen,u1,sigma1,u2,sigma2,u3,sigma3):
    #tonelen = self.timing['stimulus1']/self.dt 
    #tonelen2 = self.timing['stimulus2']/self.dt

    #u1,u2,u3 = u3[0]/self.dt,u3[1]/self.dt,u3[2]/self.dt
    #sigma1,sigma2,sigma3 = sigma3[0]/self.dt,sigma3[1]/self.dt,sigma3[2]/self.dt

    norm_distribution1 = norm(loc = u1, scale = sigma1)
    norm_distribution2 = norm(loc = u2, scale = sigma2)
    norm_distribution3 = norm(loc = u3, scale = sigma3)

    x = np.arange(tonelen)

    discrete_pdf1 = norm_distribution1.pdf(x)
    discrete_pdf2 = norm_distribution2.pdf(x)
    discrete_pdf3 = norm_distribution3.pdf(x)

    three_gaussian_distribution = discrete_pdf1 + discrete_pdf2 + discrete_pdf3
    normalize_index = np.sum(three_gaussian_distribution)
    normalized_three_gaussian_distribution = three_gaussian_distribution / normalize_index

    sample_list = []
    nsize= 10000
    for n in range(nsize):
        sample_list.append(np.random.choice(x,p=normalized_three_gaussian_distribution))

    toneTimingIdx1 = int(np.random.choice(sample_list))
    toneTimingIdx2 = int(np.random.choice(sample_list))

    return toneTimingIdx1, toneTimingIdx2