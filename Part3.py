# -*- coding: utf-8 -*-
"""
Created on Mon May  1 09:45:26 2023

@author: pinaj
"""

#                                         Monte Carlo Stochastic Simulation
trials = 10000
variable_samples = np.zeros((trials, len(mean_list)))
for col in range(len(mean_list)):
    s_mean = mean_list[col]
    s_std = std_list[col]
    variable_samples[:, col] = np.random.normal(s_mean, s_std, trials)
overall_yield = np.prod(variable_samples, axis=1)
overall_mean = overall_yield.mean()
overall_std = overall_yield.std()
true_overall_mean = np.prod(mean_list)
true_overall_std = (statistics.pstdev(mean_list))

print('Mean of Overall Yield:', true_overall_mean)
print('Standard Deviation of Overall Yield: ', true_overall_std)
print('Monte Carlo mean: ', overall_mean)
print('Monte Carlo std:', overall_std)