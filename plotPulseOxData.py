#! /usr/bin/env python

#
# Script name: plotPulseOxData.py
# Copyright 2021 Neal Patwari
#
# Purpose:
#   1. Load fake data from Figure 1 of Sjoding "Racial bias..." 2020 paper.
#   2. Build some detectors of hypoxemia (arterial oxygen saturation of <88%)
#      from pulse oximetry reading.
#   3. Calculate and plot the error performance
#
# Version History:
#   Version 1.0:  Initial Release.  11 Oct 2021.
#
# License: see LICENSE.md

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Section 1: Load and understand the data.

# These are commands I always use to format plots to have a larger font size and
# to refresh automatically as they're changed.
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
plt.ion()


# Our two hypotheses:
# H0: the "normal"
#     Arterial Oxygen Saturation is >= 88.0
# H1: the "abnormal", what we want to be alarmed about
#     Arterial Oxygen Saturation is < 88.0

# Load data: There are two files separated by race.
# I use _w and _b for the white and Black patient data, respectively
data_w = np.loadtxt("oxygenation_w.csv", delimiter=', ', comments='#')
data_b = np.loadtxt("oxygenation_b.csv", delimiter=', ', comments='#')

# The 0th column is the pulse ox value.
# The 1st column is the arterial oxygen saturation.
#   We take the arterial Ox Sat as the "truth" because it is the "gold standard"
#   for monitoring of oxygen saturation in the blood.
# Each row is one patient.
pulseOx_w = data_w[:, 0]
arterOx_w = data_w[:, 1]
pulseOx_b = data_b[:, 0]
arterOx_b = data_b[:, 1]

# Plot the data
plt.figure(1)
plt.clf()
# Subplot with 1 row, 2 columns, currently plotting into #1.
plt.subplot(1, 2, 1)
plt.plot(pulseOx_w, arterOx_w, 'rx', label="White", linewidth=2)
plt.grid('on')
plt.ylim([68, 100])  # Have a uniform y limits for both subplots.
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.ylabel('Arterial Ox Saturation (%)', fontsize=16)
plt.legend(fontsize=16)

# Subplot with 1 row, 2 columns, currently plotting into #2.
plt.subplot(1, 2, 2)
plt.plot(pulseOx_b, arterOx_b, 'bx', label="Black", linewidth=2)
plt.xlabel('Pulse Ox Meast (%)', fontsize=16)
plt.grid('on')
plt.ylim([68, 100])  # Have a uniform y limits for both subplots.
plt.legend(fontsize=16)


# Our two hypotheses:
# H0: the "normal"
#     Arterial Oxygen Saturation is >= 88.0
# H1: the "abnormal", what we want to be alarmed about
#     Arterial Oxygen Saturation is < 88.0
#
# As an example, let's find the probability that a white patient
# has arterial oxygen saturation < 88.0

# Here's a python way of finding the indices of the arterOx_w vector where its value < 88.0.
# the np.where() returns (strangely) a numpy array, length 1, with the first element being a list
# of the indices.  I take care of that by simply requesting the first element of the numpy array.
H1_w_indices = np.where(arterOx_w < 88.0)[0]

# We want the probability of the arterial ox sat measurement being < 88.0, ie., the proportion:
# I use *1.0 to make sure that the division is floating point.  This is not necessary in Python 3
# but I do this to be more backwards compatible.
prob_H1_w = len(H1_w_indices)*1.0 / len(arterOx_w)

print('The probability of H1 for white patients in this data set is ' + str(prob_H1_w))


# Section 2: Be able to calculate Prob[ Correct Detection ] and Prob[ False Alarm ].

# Prob[ Correct Detection ] = # of correct decisions in the data set / # of hypoxemia cases in the data set
numCorrectB = len(np.intersect1d(np.where(arterOx_b < 88.0)
                  [0], np.where(pulseOx_b < 91.5)))
numHypoxB = len(np.where(arterOx_b < 88.0)[0])
probCorrectB = numCorrectB / numHypoxB
print("probCorrectB = " + str(probCorrectB*100))

numCorrectW = len(np.intersect1d(np.where(arterOx_w < 88.0)
                  [0], np.where(pulseOx_w < 91.5)))
numHypoxW = len(np.where(arterOx_w < 88.0)[0])
probCorrectW = numCorrectW / numHypoxW
print("probCorrectW = " + str(probCorrectW*100))


# Prob[ False Alarm ] = # of false alarms in the data set / # of cases in the data set that are not hypoxemia
falseAlarmB = len(np.intersect1d(np.where(arterOx_b >= 88.0)
                  [0], np.where(pulseOx_b < 91.5)))
numNotHypoxB = len(np.where(arterOx_b >= 88.0)[0])
probFalseB = falseAlarmB / numNotHypoxB
print("probFalseB = " + str(probFalseB*100))

falseAlarmW = len(np.intersect1d(np.where(arterOx_w >= 88.0)
                  [0], np.where(pulseOx_w < 91.5)))
numNotHypoxW = len(np.where(arterOx_w >= 88.0)[0])
probFalseW = falseAlarmW / numNotHypoxW
print("probFalseW = " + str(probFalseW*100))


# Finally, calculate the probability of false alarm, and probability of detection over ALL patients.
numCorrect = numCorrectB + numCorrectW
numHypox = numHypoxB + numHypoxW
probCorrect = numCorrect / numHypox
print("overall correct detection prob = " + str(probCorrect*100))

falseAlarm = falseAlarmB + falseAlarmW
numNotHypox = numNotHypoxB + numNotHypoxW
probFalse = falseAlarm / numNotHypox
print("overall false alarm prob = " + str(probFalse*100))


# Section 3: Calculate and plot the results for all possible thresholds.
# 3a.
def calculate_false_alarm(threshold, arterOx_data, pulseOx_data):
    numfalseAlarm = len(np.intersect1d(np.where(arterOx_data >= 88.0)
                                       [0], np.where(pulseOx_data < threshold)))
    numNotHypox = len(np.where(arterOx_data >= 88.0)[0])
    return numfalseAlarm / numNotHypox


def calculate_correct_detection(threshold, arterOx_data, pulseOx_data):
    numCorrect = len(np.intersect1d(np.where(arterOx_data < 88.0)
                                    [0], np.where(pulseOx_data < threshold)))
    numHypox = len(np.where(arterOx_data < 88.0)[0])
    return numCorrect / numHypox


p_FA_w = np.empty(9)
p_FA_b = np.empty(9)
p_FA_all = np.empty(9)
p_CD_w = np.empty(9)
p_CD_b = np.empty(9)
p_CD_all = np.empty(9)

arterOx_all = np.concatenate([arterOx_w, arterOx_b])
pulseOx_all = np.concatenate([pulseOx_w, pulseOx_b])
threshold_list = [88.5, 89.5, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5]

for idx, threshold in enumerate(threshold_list):
    p_FA_w[idx] = calculate_false_alarm(threshold, arterOx_w, pulseOx_w)
    p_FA_b[idx] = calculate_false_alarm(threshold, arterOx_b, pulseOx_b)
    p_FA_all[idx] = calculate_false_alarm(threshold, arterOx_all, pulseOx_all)
    p_CD_w[idx] = calculate_correct_detection(threshold, arterOx_w, pulseOx_w)
    p_CD_b[idx] = calculate_correct_detection(threshold, arterOx_b, pulseOx_b)
    p_CD_all[idx] = calculate_correct_detection(
        threshold, arterOx_all, pulseOx_all)

# Plot the results
plt.figure()
plt.plot(p_FA_w, p_CD_w, 'rs', label="White", linewidth=2)
plt.plot(p_FA_b, p_CD_b, 'ko', label="Black", linewidth=2)
plt.plot(p_FA_all, p_CD_all, 'g.', label="All", linewidth=2)
plt.grid('on')
plt.xlabel('Probability of False Alarm', fontsize=16)
plt.ylabel('Probability of Correct Detection', fontsize=16)
plt.xticks(np.arange(0, 1.01, 0.1))
plt.yticks(np.arange(0, 1.01, 0.1))
plt.legend(fontsize=16)

for i, threshold in enumerate(threshold_list):
    # Put the threshold on each dot, connect the white/Black points for
    # that correspond to the same threshold.
    plt.text(p_FA_w[i], p_CD_w[i], str(threshold), horizontalalignment='right')
    plt.text(p_FA_b[i], p_CD_b[i], str(threshold), horizontalalignment='left')
    plt.plot([p_FA_b[i], p_FA_w[i]], [p_CD_b[i], p_CD_w[i]], 'b-', linewidth=2)

# 3b.
print("errors, all patients: ")
p_FN_all = 1 - p_CD_all
minSum = np.add(p_FA_all, p_FN_all)
print(minSum)

print("errors, white patients: ")
p_FN_w = 1 - p_CD_w
minSum_w = np.add(p_FA_w, p_FN_w)
print(minSum_w)

print("errors, black patients: ")
p_FN_b = 1 - p_CD_b
minSum_b = np.add(p_FA_b, p_FN_b)
print(minSum_b)


# Section 4:  Consider having a different threshold by race.

def calculate_total_error(threshold, arterOx, pulseOx):
    false_alarm = calculate_false_alarm(threshold, arterOx, pulseOx)
    false_neg = 1 - calculate_correct_detection(threshold, arterOx, pulseOx)
    return false_alarm + false_neg


print("threshold testing: ")
print(calculate_total_error(93.0, arterOx_b, pulseOx_b))
