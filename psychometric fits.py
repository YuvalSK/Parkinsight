# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:19:36 2023
@author: Yuvi
"""
import numpy as np
np.seterr(divide = 'ignore') 
import pandas as pd
from scipy import stats
from pingouin import bayesfactor_ttest
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import psignifit as ps
options = {'sigmoidName': 'norm',
           'expType': 'YesNo'}
"""
for psignifit, the data should be in the format:
    [x, nCorrect, total]. 
"""

def spearmanr_statistic(x, y):
    return stats.spearmanr(x, y)[0]

def pearsonsr_statistic(x, y):
    return stats.pearsonr(x, y)[0]

#import data
conf_df = pd.read_csv('data/animals-conf.csv')
real_df = pd.read_csv('data/animals-real.csv')
times_df = pd.read_csv('data/animals-rt.csv')

participants = conf_df.columns[1:-2].values

o = 'phy-dist'
marker_size = 8
fig, ax1 = plt.subplots()

'''
Confidence analysis
'''
y1 = pd.pivot_table(conf_df,
                    values=participants, 
                    index=[o],
                    aggfunc='mean')

c1 = pd.pivot_table(conf_df,
                    values=participants[0], 
                    index=[o],
                    aggfunc='count')

e1 = y1.std(ddof=1, axis=1)/np.sqrt(len(participants))

x = y1.index 
ax1.scatter(x, np.mean(y1, axis=1), 
            facecolor= 'none',
            edgecolor='grey',
            s=marker_size*c1,
            linewidths=1
            )

#plt.errorbar(x, np.mean(y1, axis=1), yerr=e1,
#             ecolor ='k', 
#             fmt='none',
#             elinewidth=1)
z = np.polyfit(x, np.mean(y1, axis=1), 1)
p = np.poly1d(z)
res = stats.linregress(x, np.mean(y1, axis=1))
df = len(y1)-2
tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
ts = tinv(0.05, df)

l = f"Confidence"
print(f"Confidence: R² = {res.rvalue**2:.3f}\nβ1 = {res.slope:.3f}, ci(95%) = [{res.slope-ts*res.stderr:.3f},{res.slope+ts*res.stderr:.3f}], p(β1) = {res.pvalue:.5f}\nβ0: {res.intercept:.4f}, ci(95%) = [{res.intercept-ts*res.intercept_stderr:.4f},{res.intercept+ts*res.intercept_stderr:.4f}]")
ax1.plot(x, p(x), c='grey', linestyle='--', label=l)

'''
RTs analysis
'''
y3 = pd.pivot_table(times_df,
                    values=participants, 
                    index=[o],
                    aggfunc='mean')

ax2 = ax1.twinx()
ax2.scatter(x, np.mean(y3, axis=1),
            facecolor = 'none',
            edgecolor='r',
            s=marker_size*c1,
            linewidths=1
            )

exclude = 10000
y3 = y3[y3[participants]<exclude]
z3 = np.polyfit(x, np.mean(y3, axis=1), 1)
p3 = np.poly1d(z3)
res = stats.linregress(x, np.mean(y3, axis=1))

df = len(y3)-2
tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
ts = tinv(0.05, df)
l = f"RT"
print(f"RT: R² = {res.rvalue**2:.3f}\nβ1: {res.slope:.3f}, ci(95%) = [{res.slope-ts*res.stderr:.3f},{res.slope+ts*res.stderr:.3f}], p-val(β1): {res.pvalue:.5f}\nβ0: {res.intercept:.3f}, ci(95%) = [{res.intercept-ts*res.intercept_stderr:.3f},{res.intercept+ts*res.intercept_stderr:.3f}]")

ax2.plot(x, p3(x), c='r', linestyle='-', label=l)
ax1.plot(x, p3(x), c='r', linestyle='-', label=l)
ax1.set_ylim(bottom=0.9, top=4.1)
ax2.set_ylim(bottom=1000, top=3800)


'''
Realness analysis
'''
y2 = pd.pivot_table(real_df,
                    values=participants, 
                    index=[o],
                    aggfunc='mean')


e2 = y2.std(ddof=1, axis=1)/np.sqrt(len(participants))
x = y2.index 

ax1.scatter(x, np.mean(y2, axis=1),
            facecolor = 'none',
            edgecolor='k',
            s=marker_size*c1,
            linewidths=1
            )

#plt.errorbar(x, np.mean(y2, axis=1), yerr=e2,
#             ecolor ='k', 
#             fmt='none',
#             elinewidth=1)

z2 = np.polyfit(x, np.mean(y2, axis=1), 1)
p2 = np.poly1d(z2)
res = stats.linregress(x, np.mean(y2, axis=1))

df = len(y2)-2
tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
ts = tinv(0.05, df)
l = f"Real"
print(f"Realness: R² = {res.rvalue**2:.3f}\nβ1: {res.slope:.3f}, ci(95%) = [{res.slope-ts*res.stderr:.3f},{res.slope+ts*res.stderr:.3f}], p-val(β1): {res.pvalue:.5f}\nβ0: {res.intercept:.3f}, ci(95%) = [{res.intercept-ts*res.intercept_stderr:.3f},{res.intercept+ts*res.intercept_stderr:.3f}]")
ax1.plot(x, p2(x), c='k', linestyle='-', label=l)

#vals = real_df[participants[0]]
#plt.xticks(x, conf_df['image'], rotation=90)
#plt.xticks(ticks= conf_df['image'].values)
#plt.yticks(ticks= conf_df[o])


ax1.set_ylabel('Response')
ax1.set_xlabel(f'{o} of stimuli')
ax2.set_ylabel('RT[ms]')
ax1.legend(loc='center right')
plt.tight_layout()
#plt.show()

plt.savefig(f'Res/{o}.png', dpi=800)

#individual slopes?
slopes_r = []
slopes_rt = []
intercepts_r = []
intercepts_rts = []

for s in participants:
    print(f"analyzing participant: {s}/{len(participants)}")
    plt.clf()
    y1 = pd.pivot_table(times_df,
                        values=s, 
                        index=[o],
                        aggfunc='mean')

    c1 = pd.pivot_table(times_df,
                        values=s, 
                        index=[o],
                        aggfunc='count')

    x = y1.index 
    plt.scatter(x, y1[s], 
                facecolor= 'none',
                edgecolor='r',
                s=marker_size*c1,
                linewidths=1
                )
    
    z = np.polyfit(x, y1[s],1)
    p = np.poly1d(z)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y1[s])
    slopes_rt.append(slope)
    intercepts_rts.append(slope)
    l = f"RT: β0: {intercept:.4f}, β1: {slope:.4f}"
    plt.plot(x, p(x), c='r', linestyle='-', label=l)
    
    #for analysis with caps
    
    y2 = pd.pivot_table(real_df,
                        values=s, 
                        index=[o],
                        aggfunc='mean')

    plt.scatter(x, y2[s],
                facecolor = 'none',
                edgecolor='k',
                s=marker_size*c1,
                linewidths=1
                )

    z2 = np.polyfit(x, y2[s],1)
    p2 = np.poly1d(z2)
    r_slope, r_intercept, r_r_value, r_p_value, r_std_err = stats.linregress(x, y2[s])
    l_r = f"Realness: β0: {r_intercept:.4f}, β1: {r_slope:.4f}"
    slopes_r.append(r_slope)
    intercepts_r.append(r_intercept)
    
    plt.plot(x, p2(x), c='k', linestyle='-', label=l_r)
    plt.legend(loc='center right')
    plt.title(f"Subject: {s}")
    plt.savefig(f'Res/participants/{s}_{o}_real_RT.png', dpi=800)


caps = pd.read_csv('data/data with caps.csv')

#exploration
y = slopes_r
print(f"Realness mean: β1({o}): {np.mean(slopes_r):.3f} ± {stats.sem(slopes_r, ddof=1):.3f}")
t, p = stats.ttest_1samp(slopes_r, 0)
bf = bayesfactor_ttest(t, np.size(x), np.size(y), 
                       paired=False, 
                       alternative='two-sided',
                       r=.5)
print(f"ttest vs. zero: pval = {p:.10f}, t({len(slopes_r)-1}) = {t:.3f}, BF10 = {bf:.3f}")


y = slopes_rt
print(f"mean RT β1({o}): {np.mean(slopes_rt):.3f} ± {stats.sem(slopes_rt, ddof=1):.3f}")
t, p = stats.ttest_1samp(slopes_rt, 0)
bf = bayesfactor_ttest(t, np.size(x), np.size(y), 
                       paired=False, 
                       alternative='two-sided',
                       r=.5)
print(f"ttest vs. zero: pval = {p:.10f}, t({len(slopes_rt)-1}) = {t:.3f}, , BF10 = {bf:.3f}")

y = intercepts_r
print(f"Realness mean: β0({o}): {np.mean(y):.3f} ± {stats.sem(y, ddof=1):.3f}")
t, p = stats.ttest_1samp(y, 0)
bf = bayesfactor_ttest(t, np.size(x), np.size(y), 
                       paired=False, 
                       alternative='two-sided',
                       r=.5)
print(f"ttest vs. zero: pval = {p}, t({len(y)-1}) = {t:.3f}, BF10 = {bf:.3f}")

#analysis by id of slopes
ax = plt.subplot()
var = "participant_id"
x = caps[var]
y = slopes_r
fig, ax = plt.subplots()
ax.scatter(np.arange(1, len(x)+1,1), y,
            facecolor='none',
            edgecolors='k')
ax.axhline(y=0, linestyle='--', color='k')

ax.set_ylabel(f"β1({o})")
ax.set_xlabel(f"{var}")
ax.spines[['right', 'top']].set_visible(False)
#plt.show()

plt.savefig(f"Res/{o} by {var} - β0 realism.png", dpi=800)

#analysis by age of intercepts
var = "Age"
x = caps[var]
y = intercepts_r

r, p = stats.pearsonr(x, y)
print(f"Pearson's r: r = {r:.3f}, p = {p:.3f}")

res = stats.permutation_test((x, y), pearsonsr_statistic, vectorized=False,
                       n_resamples = 9999, 
                       alternative='two-sided')
print(f"r = {res.statistic:.3f}, p(prem.) = {res.pvalue:.4f}")

fig, ax = plt.subplots()
plt.scatter(x, y, 
            facecolor='none',
            edgecolors='gray')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
res = stats.linregress(x, y)

plt.plot(x, p(x), c='k', linestyle='-')

plt.ylabel(f"β0 realism")
plt.xlabel(f"{var}")
plt.xlim(left=18, right=88)
plt.xticks(np.arange(18, 89, 10))

sns.despine(ax=ax, top = True, right = True)

plt.tight_layout()
#plt.legend()
#plt.show()

plt.savefig(f"Res/{o} by {var} - β0 realism.png", dpi=800)

#analysis by caps of intercepts
var = "Total_CAPS_Score"
x = caps[var]
r, p = stats.pearsonr(x, y)
print(f"Pearson's r: r = {r:.3f}, p = {p:.3f}")

res = stats.permutation_test((x, y), pearsonsr_statistic, vectorized=False,
                       n_resamples = 9999, 
                       alternative='two-sided')
print(f"r = {res.statistic:.3f}, p(prem.) = {res.pvalue:.4f}")

fig, ax = plt.subplots()
plt.scatter(x, y, 
            facecolor='none',
            edgecolors='gray')

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
res = stats.linregress(x, y)

plt.plot(x, p(x), c='k', linestyle='-')

plt.ylabel(f"β0 realism")
plt.xlabel(f"{var}")
plt.xlim(left=0, right=20)
sns.despine(ax=ax, top = True, right = True)

plt.tight_layout()
#plt.legend()
#plt.show()

plt.savefig(f"Res/{o} by {var} - β0 realism.png", dpi=800)


#phychophysics
o_stimuli = ['phy-dist', 'vis-sim']
#n_subplots = int(len(participants))


#logistic fits per participant
for o in o_stimuli:
    print(f"calculating by {o}...")
    for c, participant in enumerate(participants):
        fig, ax = plt.subplots(figsize=(20, 9))

        print(f"-{c+1}/{len(participants)}: {participant[-6:]}")
        ## count how many times mean above 2
        conf_sort = conf_df.sort_values(o, ascending=False)
    
        nCorrect = pd.pivot_table(conf_sort,
                                  values=participant, 
                                  index=[o],
                                  aggfunc='mean')
                                #aggfunc=lambda x:(x>4).count())
        total = pd.pivot_table(conf_sort,
                               values=participant, 
                               index=[o],
                               aggfunc="count")
        
        #more than a single trial
        total = total.loc[(total[participant] > 1)]
        
        nCorrect = nCorrect.loc[total.index]
        #unique stimuli
        unique_s = total.value_counts()
        # parse unique stimuli into desired range for psychophysics
        x = np.arange(0.001, 0.01, 0.01/(len(unique_s)+1)) 
    
        data = []
        for i in range(len(unique_s)):
            val = [x[i], total.iloc[i,0] * (nCorrect.iloc[i,0]-1)/3, total.iloc[i,0]]
            data.append(val)
            #if val[1] != val[2]:
                #print(f"difference! {val[1]} vs. {val[2]}")
        
        psy_data = np.array(data) 
        #print(psy_data)
        
        res = ps.psignifit(psy_data, options)
        ps.psigniplot.plotPsych(res, 
                                xLabel=f'Stimulus by {o}', 
                                yLabel='High conf. ratio',
                                legendLabel='NT',
                                dataColor='k',lineColor='k', 
                                axisHandle=ax, showImediate=False)
     
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #plt.rcParams.update({'font.size': 8})
        #plt.tight_layout()
        
        plt.savefig(f'Res/participants/{o}-{participant}.png', dpi=600)




"""
del res1['Posterior']
del res1['weight']
del res2['Posterior']
del res2['weight']
 Without the del fields above we wont be able to use the 2D Bayesian plots
 anymore and the equalityTest will fail. All other functions work without
 it.
"""