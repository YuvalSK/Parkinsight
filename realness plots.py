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

plt.savefig(f'Res/average-realness-CAPS.png')

#inidivdual-level analysis
## sort by participants
###options = 'Age', 'Total_CAPS_Score'

sort_by = 'Total_CAPS_Score'
df_plot = df.loc[:, subjects]
df_plot['image'] = df['image']
df_by_caps = df_caps.sort_values(by=[sort_by], ascending=True)
fig, ax = plt.subplots(1,1)
x_t = np.arange(0, len(df_by_caps)+1)

plt.clf()
plt.hist(df_by_caps[sort_by], 
         bins=10,
         label='Our data', 
         facecolor='lightgray', 
         edgecolor='k')

plt.xlim(0, 25)
plt.yticks(ticks=np.arange(0,21,5))

plt.ylim(0, 25)
plt.xticks(ticks=np.arange(0,21,5))

plt.xlabel('CAPS Score')
plt.ylabel('Participants [#]')
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig(f'Res/participants_by_{sort_by}.png', dpi=800)

## sort by images
sort_by_i = 'Log Phylogenetic distance'
###options = 'Clip Realness', 'Semantic Similarity', 'Visual Similarity', 'Phylogenetic distance', 'Log Phylogenetic distance'

df_sorted = pd.DataFrame()
sorted_ids = [str(i) for i in df_by_caps['participant_id'].to_list()]
sorted_ids.append('image')
df_sorted = df_plot[sorted_ids].copy()
#df_sorted['avg_realness_image'] = df_sorted.iloc[:,:-1].mean(axis=1)

df_sorted[sort_by_i] = df[sort_by_i]
df_sorted = df_sorted.sort_values(by=[sort_by_i],ascending=False) # see order matches relations of sorted_by and image realness

plt.scatter(df_sorted[sort_by_i], df_sorted['image'], facecolor='none', edgecolor='k', s=20)
plt.ylim(-1, 95)
plt.xlim(0.84, 1.0)
plt.yticks(fontsize = 4)
plt.ylabel('Stimuli')
plt.xlabel(f'{sort_by_i}')
sns.despine()
plt.tight_layout()
plt.show()

plt.savefig(f'Res/images_by_{sort_by_i}_sorted.png', dpi=800)



#adding the average realness report
'''
avg = df_caps['AVG realness'].to_list()
df_plot.loc[len(df_plot.index)] = avg 
labels.append("Avg. realness")
'''
sns.set_style('ticks')
labels = df_sorted['image'].to_list()

ax = sns.heatmap(df_sorted.iloc[:,:-2], 
                 cmap='coolwarm', cbar_kws={'label': 'Realness score'},
                 yticklabels=labels, xticklabels=True)

ax.tick_params(axis='both', which='major', width = 0.2,length=3, labelsize=3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.ylabel(f'Stimuli ordered by phyl. dist.', fontsize = 10)
plt.xlabel(f'Subjects oredered by CAPS score', fontsize = 8)
plt.tight_layout()
#plt.show()

plt.savefig(f'Res/heatmap_sorted_{sort_by}-{sort_by_i}2.png', dpi=800)


#order data to run mixed effect model (in R/JASP; less flexible in Python)
df_long = pd.melt(df_sorted, id_vars=['image'])
df_long['variable'] = df_long['variable'].astype(str)
df_by_caps['participant_id'] = df_by_caps['participant_id'].astype(str)
df_long = df_long.merge(df_by_caps[['participant_id', sort_by]], how='left', left_on='variable', right_on='participant_id')
df_long = df_long.merge(df[['image', sort_by_i]], how='left', on='image')
df_long.to_csv(f'data/long_reg_{sort_by}-{sort_by_i}.csv',index=False)

