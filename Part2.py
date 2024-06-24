# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 20:56:33 2023

@author: pinaj
"""
#                                                Part 1
import pandas as pd
import numpy as np
import toleranceinterval as ti
import scipy.stats as stats
from scipy.stats import norm
import random 
from random import sample
from random import randrange
from numpy  import sqrt
from scipy.stats import chi2

#                                       Loads the WorkBook (Works)
excel_data_df = pd.read_excel('example-input-1.xlsx', sheet_name='Ex1 Yields', usecols='B:J', skiprows = 7)

#                                    Makes an Empty DataFrame (Works)
new_df = pd.DataFrame()

#                                  Calculates the Summary Statistics(Works)
new_df['Mean'] = excel_data_df.mean(axis='index')
new_df['Standard Deviation'] = excel_data_df.std(axis='index')
new_df['Count'] = excel_data_df.count(axis='index')

#                          Calculates the One-sided Lower Tolerance Interval (Works)
# bound1 = ti.oneside.normal(new_df, 0.9986501, 0.95)
# new_df['Lower Bound'] = bound1 
bound1_lower_tol = []
alpha = 0.95
prop1 = 0.9986501
for i, row in new_df.iterrows():
    one_mean = row['Mean']
    one_std = row['Standard Deviation']
    one_count = row['Count']
    dof_1 = one_count - 1
    zp = norm.isf((1-prop1))
    si =zp*sqrt(one_count)
    t = stats.t.ppf((1-alpha),dof_1,si)
    k1 = t / sqrt(one_count)      # Tolerance factor
    Tol_int = one_mean - (k1 * one_std)
    bound1_lower_tol.append(Tol_int)
new_df['Lower Tolerance Interval'] = bound1_lower_tol

#                        Calculates the 2-sided Tolerance Interval (Second Approach Works)

# -----(First Approach)------
# bound2 = ti.twoside.normal(new_df, 0.99730, 0.95) 
# new_df['2-sided Lower Bound'] = bound2[:,0]
# new_df['2-sided Upper Bound'] = bound2[:,1]

# ----(Second Approach)------
bound2_lower_tol = []
bound2_upper_tol = []
prop2 = 0.99730
for i, row in new_df.iterrows():
    row_mean = row['Mean']
    row_std = row['Standard Deviation']
    row_count = row['Count']
    dof = row_count - 1
    propinv = (1.0 - prop2) / 2.0
    z = norm.isf(propinv)
    chi = chi2.isf(q=(1-alpha), df=dof)
    k2 = sqrt((dof * (1 + (1/row_count)) * z**2)/ chi)
    lower_tol = row_mean - (k2 * row_std)
    upper_tol = row_mean + (k2 * row_std)
    bound2_lower_tol.append(lower_tol)
    bound2_upper_tol.append(upper_tol)
new_df['2-sided Lower Bound'] = bound2_lower_tol
new_df['2-sided Upper Bound'] = bound2_upper_tol
    


#                     Prints the Dataframe with the Summary Statistics and Tolerance Intervals(Works)
pd.options.display.max_columns = None
print(new_df)



# For the second set
excel_data2_df = pd.read_excel('example-input-2.xlsx', sheet_name='Ex2 Yields', usecols='B:J', skiprows = 7)
new2_df = pd.DataFrame()

new2_df['Mean'] = excel_data2_df.mean(axis='index')
new2_df['Standard Deviation'] = excel_data2_df.std(axis='index')
new2_df['Count'] = excel_data2_df.count(axis='index')

# bound1_2 = ti.oneside.normal(new2_df, 0.9986501, 0.95)
# bound2_2 = ti.twoside.normal(new2_df, 0.99730, 0.95)

bound1_2_lower_tol = []
for i, row in new2_df.iterrows():
    one2_mean = row['Mean']
    one2_std = row['Standard Deviation']
    one2_count = row['Count']
    dof_1_2 = one2_count - 1
    zp_2 = norm.isf((1-prop1))
    si_2 =zp_2*sqrt(one2_count)
    t2 = stats.t.ppf((1-alpha),dof_1_2,si_2)
    k1_2 = t2 / sqrt(one2_count)      # Tolerance factor
    Tol_int2 = one2_mean - (k1_2 * one2_std)
    bound1_2_lower_tol.append(Tol_int2)
new2_df['Lower Tolerance Interval'] = bound1_2_lower_tol

# new2_df['Lower Bound'] = bound1

bound2_2_lower_tol = []
bound2_2_upper_tol = []
for i, row in new2_df.iterrows():
    row2_mean = row['Mean']
    row2_std = row['Standard Deviation']
    row2_count = row['Count']
    dof = row2_count - 1
    propinv_2 = (1.0 - prop2) / 2.0
    z_2 = norm.isf(propinv_2)
    chi = chi2.isf(q=(1-alpha), df=dof)
    k2_2 = sqrt((dof * (1 + (1/row2_count)) * z_2**2)/ chi)
    lower_tol_2 = row2_mean - (k2_2 * row2_std)
    upper_tol_2 = row2_mean + (k2_2 * row2_std)
    bound2_2_lower_tol.append(lower_tol_2)
    bound2_2_upper_tol.append(upper_tol_2)
new2_df['2-sided Lower Bound'] = bound2_2_lower_tol
new2_df['2-sided Upper Bound'] = bound2_2_upper_tol



print(new2_df)



#End of Part 1

#                                                   Part 2
stochastic_data_df = pd.DataFrame()
stohastic_data = []
for i, rows in new_df.iterrows():
    stohastic_mean = row['Mean']
    stohastic_std = row['Standard Deviation']
    s = np.random.normal(stohastic_mean, stohastic_std, 10)
    stohastic_data.append(s)
stochastic_data_df['Stohastic Data'] = stohastic_data

print(stochastic_data_df)


