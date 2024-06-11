# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 21:20:39 2023

@author: nn
"""

import hddm
import patsy
from patsy import dmatrices
import numpy as np
import matplotlib
import arviz as az
# matplotlib.use('agg')
import pandas as pd
import matplotlib.pyplot as plt
import kabuki
import pymc
from kabuki.analyze import gelman_rubin
from kabuki.analyze import check_geweke
from hddm.models import HDDMBase
from kabuki.hierarchical import Knode
from kabuki.utils import HalfCauchy
import warnings
warnings.filterwarnings('ignore')

#======================= load data ==============================================
# load data 
dataALL=hddm.load_csv('ALL.csv')

#======================= fitting ===============================================
# fit regression model for full variables. this part is the formal analysis
# all results come from this model.
# here saved model fittings, parameters, contrasts and convergency
AESlist=[]
for i in range(5):
    AESregressor= hddm.HDDMRegressor(dataALL, {'v~AES+ANT+CON+C(group)+C(F)+R+E','a~AES+ANT+CON+C(group)+C(F)+R+E'}, p_outlier=0.05, group_only_regressors=(False),keep_regressor_trace=True)
    AESregressor.find_starting_values()
    AESregressor.sample(10000,burn=5000,thin=2,dbname='tracesAES_group.db',db='pickle') # MCMC 
    AESlist.append(AESregressor)
    
# chcek convergency   
AES_convergence=gelman_rubin(AESlist)
np.save('AES_group_convergency.npy',AES_convergence)
#readALLconvergency=np.load('AESconvergency.npy').item()

# concat models
AESmodel=kabuki.utils.concat_models(AESlist, concat_traces=True)

# check convergence by visually
AESmodel.plot_posteriors()
AESmodel.plot_posterior_predictive()
AESregressor.plot_posterior_predictive()

# save result
AESmodel.save('AES_group_results')
AESstats = AESmodel.print_stats(fname='AES_group_stats.csv') # 
AESmodel=hddm.load('AES_group_results') #load results

# PPC compare
ppc_data=hddm.utils.post_pred_gen(AESregressor)
hddm.utils.post_pred_stats(dataALL, ppc_data)
ppc_data.head(10)
ppc_compare = hddm.utils.post_pred_stats(dataALL, ppc_data)
print(ppc_compare)
hddm.save_csv(ppc_compare, 'PPCcompare.csv')

#========================= extract data ==============================================
# extract drift rate and statistic compare

v1_Inter,v1_G,v1_AES,v1_ANT,v1_CON,v1_F,v1_R,v1_E = AESmodel.nodes_db.loc[['v_Intercept','v_C(group)[T.PA]','v_AES','v_ANT','v_CON','v_C(F)[T.win]','v_R','v_E'],'node']
                                            
a1_Inter,a1_G,a1_AES,a1_ANT,a1_CON,a1_F,a1_R,a1_E = AESmodel.nodes_db.loc[['a_Intercept','a_C(group)[T.PA]','a_AES','a_ANT','a_CON','a_C(F)[T.win]','a_R','a_E'],'node']
     
#-----------------------------contrast variables----------------------------------

v2_inter  =pd.Series(np.array((v1_Inter.trace()<0).mean()))
v2_AES    =pd.Series(np.array((v1_AES.trace()<0).mean()))# AES effect
v2_ANT    =pd.Series(np.array((v1_ANT.trace()<0).mean()))# anticipatory effect
v2_CON    =pd.Series(np.array((v1_CON.trace()<0).mean()))# consumatory effect
v2_G      =pd.Series(np.array((v1_G.trace()<0).mean()))# group effect
v2_F      =pd.Series(np.array((v1_F.trace()<0).mean()))# framework effect
v2_R      =pd.Series(np.array((v1_R.trace()<0).mean()))# reward effect
v2_E      =pd.Series(np.array((v1_E.trace()<0).mean()))# effort effect

v2_ER_split=pd.Series(np.array((v1_E.trace()<v1_R.trace()).mean()))#compare E R separately
v2_AES_ANT=pd.Series(np.array((v1_AES.trace()<v1_ANT.trace()).mean()))#compare AES/anti
v2_AES_CON=pd.Series(np.array((v1_AES.trace()<v1_CON.trace()).mean())) #compare AES/consum
v2_ANT_CON=pd.Series(np.array((v1_ANT.trace()<v1_CON.trace()).mean()))#compare anti/consum

a2_inter  =pd.Series(np.array((a1_Inter.trace()<0).mean()))
a2_AES    =pd.Series(np.array((a1_AES.trace()<0).mean()))# AES effect
a2_ANT    =pd.Series(np.array((a1_ANT.trace()<0).mean()))# anticipatory effect
a2_CON    =pd.Series(np.array((a1_CON.trace()<0).mean()))# consumatory effect
a2_F      =pd.Series(np.array((a1_F.trace()<0).mean()))# framework effect
a2_R      =pd.Series(np.array((a1_R.trace()<0).mean()))# reward effect
a2_E      =pd.Series(np.array((a1_E.trace()<0).mean()))# effort effect
a2_G      =pd.Series(np.array((a1_G.trace()<0).mean()))# group effect

a2_ER_split =pd.Series(np.array((a1_E.trace()<a1_R.trace()).mean()))#compare E R separately
a2_AES_ANT  =pd.Series(np.array((a1_AES.trace()<a1_ANT.trace()).mean()))#compare AES/anti
a2_AES_CON  =pd.Series(np.array((a1_AES.trace()<a1_CON.trace()).mean())) #compare AES/consum
a2_ANT_CON  =pd.Series(np.array((a1_ANT.trace()<a1_CON.trace()).mean()))#compare anti/consum

#-----------------------save contrast results------------------------------------
v2_contrast = pd.concat([v2_G,v2_inter,v2_AES,v2_ANT,v2_CON,v2_F,v2_R,v2_E,
                         v2_ER_split,v2_AES_ANT,v2_AES_CON,v2_ANT_CON],
                          keys=['v2_group','v2_inter','v2_AES','v2_anti','v2_consum','v2_F','v2_R','v2_E',
                                'v2_ER','v2_AESANT','v2_AESCON','v2_ANTCON'])
a2_contrast = pd.concat([a2_G,a2_inter,a2_AES,a2_ANT,a2_CON,a2_F,a2_R,a2_E,
                         a2_ER_split,a2_AES_ANT,a2_AES_CON,a2_ANT_CON],
                          keys=['a2_group','a2_inter','a2_AES','a2_anti','a2_consum','a2_F','a2_R','a2_E',
                               'a2_ER','a2_AESANT','a2_AESCON','a2_ANTCON' ])


hddm.save_csv(v2_contrast, 'v2_contrast.csv')
hddm.save_csv(a2_contrast, 'a2_contrast.csv')

#=====================plot posterial distribution of drift rate===========================

font={'family':'Times New Roman','weight':'bold','size':'16'}

# AES, ANT, CON effect
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['r','dodgerblue','g'])")
hddm.analyze.plot_posterior_nodes([v1_AES,v1_ANT,v1_CON],bins=18)
plt.xlabel('clinical severity effect on drift rate',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(loc='upper left',frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('v1_AES_ANT_CON.png',dpi=600)#red is AES, blue is ANT,green is CON

# group difference
hddm.analyze.plot_posterior_nodes([v1_G],bins=7)
plt.xlabel('group effect in drift rate')
plt.ylabel('posterior probability')
plt.legend(fancybox=False,labels=['HC-PA'])
plt.savefig('v1_G.png',dpi=300)

# effort 
hddm.analyze.plot_posterior_nodes([v1_E],bins=7)
plt.xlabel('effort effect in drift rate')
plt.ylabel('posterior probability')
plt.legend(fancybox=False,labels=['effort'])
plt.savefig('v1_E.png',dpi=300)

# reward
hddm.analyze.plot_posterior_nodes([v1_R],bins=7)
plt.xlabel('reward effect in drift rate')
plt.ylabel('posterior probability')
plt.legend(fancybox=False,labels=['reward'])
plt.savefig('v1_R.png',dpi=300)

# framework
hddm.analyze.plot_posterior_nodes([v1_F],bins=7)
plt.xlabel('framework effect in drift rate')
plt.ylabel('posterior probability')
plt.legend(fancybox=False,labels=['win-loss'])
plt.savefig('v1_F.png',dpi=300)

#---------------plot threshold-----------------------------
# AES, ANT, CON
hddm.analyze.plot_posterior_nodes([a1_AES,a1_ANT,a1_CON],bins=15)
plt.xlabel('clinical effect in threshold',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(loc='upper left',frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('a_AES_anti_consum.png',dpi=600)

# group difference
hddm.analyze.plot_posterior_nodes([a1_G],bins=10)
plt.xlabel('group effect in threshold')
plt.ylabel('posterior probability')
plt.legend(fancybox=False,labels=['HC-PA'])
plt.savefig('a1_G.png',dpi=300)

# reward
hddm.analyze.plot_posterior_nodes([a1_R],bins=8)
plt.xlabel('reward effect in threshold')
plt.ylabel('posterior probability')
plt.legend(fancybox=False,labels=['reward'])
plt.savefig('a1_R.png',dpi=300)

# effort
hddm.analyze.plot_posterior_nodes([a1_E],bins=8)
plt.xlabel('effort effect in threshold')
plt.ylabel('posterior probability')
plt.legend(fancybox=False,labels=['effort'])
plt.savefig('a1_E.png',dpi=300)

# framework
hddm.analyze.plot_posterior_nodes([a1_F],bins=10)
plt.xlabel('framework effect in threshold')
plt.ylabel('posterior probability')
plt.legend(fancybox=False,labels=['win-loss'])
plt.savefig('a1_F.png',dpi=300)

#========================================================================
#==========================================================================
# ==========run task model via depends_on for ploting======================
# Group
Gmodel=hddm.HDDM(dataALL,depends_on={'v':'group','a':'group'},p_outlier=0.05)
Gmodel.find_starting_values()
Gmodel.sample(10000,burn=5000,thin=2,dbname='tracesG.db',db='pickle')
Gstats = Gmodel.print_stats(fname='G_stats.csv') #
Gmodel.save('G_results')
Gstats = Gmodel.print_stats(fname='G_stats.csv') # 
Gmodel=hddm.load('G_results') #load results

# framework 
WLmodel=hddm.HDDM(dataALL,depends_on={'v':'F','a':'F'},p_outlier=0.05)
WLmodel.find_starting_values()
WLmodel.sample(10000,burn=5000,thin=2,dbname='tracesF.db',db='pickle')
WLstats = WLmodel.print_stats(fname='WL_stats.csv') #
WLmodel.save('WL_results')
WLstats = WLmodel.print_stats(fname='WL_stats.csv') # 
Fmodel=hddm.load('WL_results') #load results

# effort
Emodel=hddm.HDDM(dataALL,depends_on={'v':'E','a':'E'},p_outlier=0.05)
Emodel.find_starting_values()
Emodel.sample(10000,burn=5000,thin=2,dbname='tracesE.db',db='pickle')
Estats = Emodel.print_stats(fname='E_stats.csv')
Emodel.save('E_results')
Estats = Emodel.print_stats(fname='E_stats.csv') 
Emodel=hddm.load('E_results') #load results

# reward 
Rmodel=hddm.HDDM(dataALL,depends_on={'v':'R','a':'R'},p_outlier=0.05)
Rmodel.find_starting_values()
Rmodel.sample(10000,burn=5000,thin=2,dbname='tracesR.db',db='pickle')
Rstats = Rmodel.print_stats(fname='R_stats.csv')
Rmodel.save('R_results')
Rstats = WLmodel.print_stats(fname='R_stats.csv') # 
Rmodel=hddm.load('R_results') #load results

vr1,vr2,vr3,vr4,vr5,ar1,ar2,ar3,ar4,ar5= Rmodel.nodes_db.loc[['v(5)','v(10)','v(15)','v(20)','v(25)','a(5)','a(10)','a(15)','a(20)','a(25)'],'node']

ve1,ve2,ve3,ve4,ve5,ae1,ae2,ae3,ae4,ae5= Emodel.nodes_db.loc[['v(2)','v(4)','v(6)','v(8)','v(10)','a(2)','a(4)','a(6)','a(8)','a(10)'],'node']

aL,aW,vL,vW= Fmodel.nodes_db.loc[['a(loss)','a(win)','v(loss)','v(win)'],'node']

aHC,aPA,vHC,vPA=Gmodel.nodes_db.loc[['a(PA)','a(HC)','v(PA)','v(HC)'],'node']

#-------------drift rate-----------------------------------

# group difference 
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['red','green'])")
hddm.analyze.plot_posterior_nodes([vHC,vPA],bins=15)
plt.xlabel('group effect in drift rate',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('v_G.png',dpi=600)

#  reward level on drift rate
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['g','dodgerblue','orange','darkorchid','red'])")
hddm.analyze.plot_posterior_nodes([vr1,vr2,vr3,vr4,vr5])
plt.xlabel('reward effect in drift rate',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(bbox_to_anchor=(0.41,0.4),frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('v_R.png',dpi=600)

#  effort level
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['g','dodgerblue','orange','darkorchid','red'])")
hddm.analyze.plot_posterior_nodes([ve1,ve2,ve3,ve4,ve5])
plt.xlabel('effort effect in drift rate',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(bbox_to_anchor=(0.72,0.4),frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('v_E.png',dpi=600)

# win-loss
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['green','red'])")
hddm.analyze.plot_posterior_nodes([vL,vW],bins=20)
plt.xlabel('framework effect in drift rate',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('v_WL.png',dpi=600)

#----------------threshold------------------------------------
# group on threshold 
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['red','green'])")
hddm.analyze.plot_posterior_nodes([aHC,aPA],bins=15)
plt.xlabel('group effect in threshold',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('a_G.png',dpi=600)

# threshold and reward
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['g','dodgerblue','orange','darkorchid','red'])")
hddm.analyze.plot_posterior_nodes([ar1,ar2,ar3,ar4,ar5])
plt.xlabel('reward effect in threshold',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(bbox_to_anchor=(0.7,0.4),frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('a_R.png',dpi=600)

# threshold and effort
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['g','dodgerblue','orange','darkorchid','red'])")
hddm.analyze.plot_posterior_nodes([ae1,ae2,ae3,ae4,ae5])
plt.xlabel('effort effect in threshold',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(bbox_to_anchor=(0.7,0.4),frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('a_E.png',dpi=600)

#threshold on win-loss
plt.rcParams['axes.prop_cycle'] = ("cycler('color', ['green','red'])")
hddm.analyze.plot_posterior_nodes([aL,aW],bins=10)
plt.xlabel('framework effect in threshold',font)
plt.ylabel('posterior probability',font)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(bbox_to_anchor=(0.68,0.7),frameon=False,prop={'family':'Times New Roman','size':'16'})
plt.savefig('a_WL.png',dpi=600)
