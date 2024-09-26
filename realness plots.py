# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:33:29 2024

@author: YSK
"""
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

#loading data
df = pd.read_csv('data/Combined JATOS - Combined per stimuli.csv')
df_caps = pd.read_csv('data/data with caps.csv')
subjects = [str(i) for i in df_caps['participant_id']]

#group-level analysis - mean over realness score ~ CAPS score
plt.scatter(df_caps['Total_CAPS_Score'], df_caps['AVG realness'], facecolor='none', edgecolor='k', s=20)
b, a = np.polyfit(df_caps['Total_CAPS_Score'], df_caps['AVG realness'], deg=1)
xseq = np.linspace(0, 20, num=100)
plt.plot(xseq, a + b * xseq, color="darkred", lw=1.5)
res = stats.spearmanr(df_caps['Total_CAPS_Score'], df_caps['AVG realness'])
corr = f'rho = {res[0]:.3f}*\np = {res[1]:.5f}'
plt.text(16, 3.5, corr, c='darkred')

plt.xlim(-0.2, 20)
plt.ylim(0.8, 4.0)
plt.xlabel('Total CAPS score')
plt.ylabel('Mean realness [A.U.]')

plt.savefig(f'Res/average over trials.png')

#inidivdual-level analysis - realness scores
plt.clf()
df_plot = df.loc[:, subjects]
labels = df['image'].to_list()

df_bars = df_caps.sort_values(by=['Total_CAPS_Score'])
sorted_ids = [str(i) for i in df_bars['participant_id'].to_list()]
df_sorted = df_plot[sorted_ids].copy()



#adding the average realness report
'''
avg = df_caps['AVG realness'].to_list()
df_plot.loc[len(df_plot.index)] = avg 
labels.append("Avg. realness")
'''
sns.set_style('ticks')
ax = sns.heatmap(df_sorted, 
                 cmap='coolwarm', yticklabels=labels, xticklabels=True)
ax.tick_params(axis='both', which='major', width = 0.2,length=3, labelsize=3, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.ylabel('Stimuli', fontsize = 10)
plt.xlabel('Subjects', fontsize = 8)
plt.tight_layout()
plt.savefig(f'Res/heatmap_sorted.png', dpi=800)

#inidivdual-level analysis - CAPS scores

fig, ax = plt.subplots(1,1)
ax.bar(x = np.arange(1, len(subjects)+1),height = df_bars['Total_CAPS_Score'], facecolor='none', edgecolor='k')
ax.xaxis.set_ticks(np.arange(1, len(subjects)+1))
ax.set_xlim(0, 52)
ax.set_ylim(-0.5, 20)
plt.xticks(fontsize = 9, rotation=45)
plt.xlabel('Subjects')
plt.ylabel('Total CAPS score')
sns.despine()
plt.tight_layout()
plt.savefig(f'Res/CAPS.png', dpi=800)

#individual plots
fig = plt.figure(figsize=[12,8])
for s in subjects:
    plt.clf()
    xs = df['Visual Similarity']
    ys = df[s] #realness
    zs = 20+20*df_caps[df_caps['participant_id']==s]['Total_CAPS_Score'].item()
    plt.scatter(xs, ys, s=zs, label=s)
    plt.xlim(0.8, 1)
    plt.ylim(0.5,4.5)
    plt.legend()
    plt.xlabel('Visual similarity of pair of animals')
    plt.ylabel('Realness [A.U.]')
    
    plt.savefig(f'Res/subject_{s}.png')
