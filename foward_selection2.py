import numpy as np
import pandas as pd
import missingno as msno
import math as math
import sklearn.linear_model
import sklearn.metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
        print(selected, candidate, round(current_score,4), round(best_new_score,4))
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

def scatter_corr(data,corr_mat,x,y):
    plt.figure()
    ax=sns.scatterplot(data=data,x=x,y=y)
    ax.set_title('Correlation between {} and {}: {}'.format(x,y,round(corr_mat[x][y],2)))
    ax.set(xlabel=x, ylabel=y)
    plt.plot()
    plt.savefig('corr_{}_{}.png'.format(x,y))
    return

def export_summ(summary,filename):
    for i in range(3):
        summ_prelim_html=summary.tables[i].as_html()
        pd.read_html(summ_prelim_html, header=0, index_col=0)[0].to_csv('{}{}.csv'.format(filename, i))
    return

raw=pd.read_excel('DATA_GPA modeling.xlsx')
#
Y=raw[['api00','api99','growth']]
#predictor
X=raw[['dnum', 'yr_rnd', 'meals', 'ell', 'mobility', 'acs_k3', 'acs_46', 'enroll', #school attributes
       'not_hsg', 'hsg', 'some_col', 'avg_ed', #family education
       'full', 'emer']] # teacher attribute
i=raw['snum'] #unique ID
len(set(i))==len((i))

sns.set_style()

#preview
dat_corr=raw.drop(['snum','dnum'],axis=1)
sns.pairplot(dat_corr)
corr_mat=dat_corr.corr()
sns.heatmap(corr_mat,cmap='coolwarm')

scatter_corr(dat_corr,corr_mat,'api99','api00')
scatter_corr(dat_corr,corr_mat,'avg_ed','api00')
scatter_corr(dat_corr,corr_mat,'meals','api00')
scatter_corr(dat_corr,corr_mat,'ell','api00')
scatter_corr(dat_corr,corr_mat,'not_hsg','api00')
scatter_corr(dat_corr,corr_mat,'emer','api00')

#api99 api00 hihgly correlated
#api00 +ve with avg_ed, -ve with meals, ell,not_hsg, emer

sns.pairplot(dat_corr[['some_col','avg_ed','not_hsg','hsg']])
plt.savefig('pairplot.png')

scatter_corr(dat_corr,corr_mat,'enroll','ell')
scatter_corr(dat_corr,corr_mat,'full','emer')
scatter_corr(dat_corr,corr_mat,'avg_ed','ell')
#meals +ve ell, not_hsg, -ve avg_ed
#ell +ve not_hsg, avg_ed
#yr_rnd +ve enroll
#avg_ed +ve hsg
#full -ve emer


for i, col in enumerate(raw.columns.drop(['dnum','yr_rnd'])):
    plt.figure(i)
    sns.boxplot(x='dnum', y=col,data=raw)
    # all acs_k3 in dnum 140 is negative, it is likely to be erroneous input, flip it back


plt.figure()
sns.boxplot(x='dnum', y='api00',data=raw)
plt.xticks(
    rotation=90, 
    fontweight='light'
)
plt.plot()
plt.savefig('dnum_vs_api00.png')
    # some districts are better in api, some are worse (better:98, 166. 316, worse: 41, 253, 716)


dat_adjusted=raw.copy()
dat_adjusted['acs_k3']=  abs(dat_adjusted['acs_k3'])

sns.boxplot(x='dnum', y='acs_k3',data=dat_adjusted)

pd.crosstab(dat_adjusted['dnum'],dat_adjusted['yr_rnd'],normalize='index')
#district contains a lot of in year_rnd, a lot of districts are predominantly not year round or otherwise

sns.scatterplot(data=raw,x='avg_ed',y='not_hsg')
sns.scatterplot(data=raw,x='avg_ed',y='hsg')
#hsg and not_hsg dont sum to 100%, one must be erroneous
#high hsg tends to have avg avg_ed
#doesn't make sense, no reasonable interpretation. could be erroroneous, omit this variable

dat_adjusted.drop('hsg',axis=1,inplace=True)

dat_adjusted['ell_pct']=dat_adjusted['ell']/dat_adjusted['enroll']

scatter_corr(dat_adjusted,dat_adjusted.corr(),'avg_ed','ell_pct')


#missing data
plt.figure()
msno.bar(dat_adjusted) #meals contains a lot of missing data, other not, can be droped
plt.plot()
plt.savefig('missing_dat.png')

na_rows_exc_meals=dat_adjusted.drop(['meals','avg_ed'],axis=1).isnull().any(axis=1)
dat_nona=dat_adjusted.loc[~na_rows_exc_meals].copy()
dat_nona['missing_meals']=pd.isna(dat_nona['meals']).astype(int)
for i, col in enumerate(dat_nona.columns.drop(['meals','missing_meals'])):
    plt.figure()
    sns.boxplot(x='missing_meals', y=col,data=dat_nona)
    plt.plot()
    plt.savefig('missing_meals_{}.png'.format(col))

#NA in meal predicts a better result (api00,api99)
#add this variable to the model
#lower ell_pct (eng learner), higher full (full time teacher pct), lower emergency teacher (emer) tends to miss, 
#higher parents education level (avg.ed)  tends to miss,
pd.crosstab(dat_nona['missing_meals'],dat_nona['yr_rnd'],normalize='index')
#almost all missing meal are not year around school

#Fairly evident that missing meal is Missing at Random, its missingness depends on other random variables. Use multiple imputation to estimate missing values

dat_unimputed=dat_nona.drop(['snum','api00','api99','growth','ell','missing_meals'],axis=1)
dat_unimputed=dat_unimputed.join(pd.get_dummies(dat_unimputed['dnum'],prefix='dnum',drop_first=True)).drop('dnum',axis=1)

imputer=IterativeImputer()
imputer.fit(dat_unimputed)
dat_imputed=imputer.transform(dat_unimputed)
dat_imputed=pd.DataFrame(dat_imputed,columns=dat_unimputed.columns, index=dat_nona.index)
dat_imputed=dat_imputed.join(dat_nona['missing_meals'])

#measure academic performance with average of api00 and api99
api_avg=dat_nona['api00']+dat_nona['api99']
api_avg.name='api_avg'
api_avg.index=pd.Index(range(len(api_avg.index)))
temp=dat_imputed.astype(float)
model_prelim = sm.OLS(api_avg.to_numpy(), sm.add_constant(dat_imputed)).fit()
print(model_prelim.summary())
export_summ(model_prelim.summary(), 'prelim')

(model_prelim.pvalues>0.05).sum()

#VIF
df_VIF=pd.DataFrame(index=dat_imputed.columns)
for var in dat_imputed.columns:
    model_VIF=sm.OLS(dat_imputed[var], dat_imputed.drop(var,axis=1)).fit()
    df_VIF.loc[var,'VIF']=1/(1-model_VIF.rsquared**2)
    #multi-collin comes from meals, acs_k3, acs_46, full, avg_ed, dnum_401
df_VIF.to_csv('VIF_prelim.csv')
#    
dat_standardized=pd.DataFrame(StandardScaler().fit_transform(dat_imputed),columns=dat_imputed.columns)

pca = PCA()
pca.fit(dat_standardized)

pc_all=pd.DataFrame(pca.components_,columns=dat_imputed.columns, index=["PC"+str(i) for i in range(1,len(dat_standardized.columns)+1)])
pct_explained=np.cumsum(pca.explained_variance_ratio_)

plt.figure()
ax=sns.lineplot(x=list(range(1,len(dat_imputed.columns)+1)), y=pct_explained)
ax.set_title('% variation explained by first n PCs')
ax.set(xlabel='n', ylabel='%')
plt.savefig('var_explained.png')

n_90pct=np.argmax(pct_explained>.9)+1

#after PCA, multicollinearity gone
X_pca_all=pd.DataFrame(pca.fit_transform(dat_standardized),columns=["PC"+str(i) for i in range(1,len(dat_standardized.columns)+1)])

df_VIF_pca=pd.DataFrame(index=X_pca_all.columns)
for var in X_pca_all.columns:
    model_VIF=sm.OLS(X_pca_all[var], X_pca_all.drop(var,axis=1)).fit()
    df_VIF_pca.loc[var,'VIF']=1/(1-model_VIF.rsquared**2)


#35 attributes needed to explain 90% variance 
pc6=pc_all.loc[['PC1','PC2','PC3', 'PC40','PC43','PC46']].transpose()
pc6.to_csv('pc6.csv')


#model after PCA
model_pca = sm.OLS(api_avg.to_numpy(), sm.add_constant(X_pca_all)).fit()
(model_pca.pvalues>0.05).sum()
export_summ(model_pca.summary(), 'summ_pca')

#model with forward selction
model_opt=forward_selected(X_pca_all.join(api_avg),'api_avg')
factor_selected=list(model_opt.params.index)
factor_selected.remove('Intercept')
model_opt.summary()
export_summ(model_opt.summary(), 'summ_forw')
(model_opt.pvalues<0.05).sum()

#No heteroscedasticity
plt.figure()
stats.probplot(model_opt.resid, dist="norm", plot= plt)
plt.savefig('ResidualQQ.png')

plt.figure()
ax=sns.scatterplot(x=model_opt.fittedvalues,y=model_opt.resid)
ax.set(xlabel='fitted_value', ylabel='residual', title='Residual Plot')
plt.savefig('ResidualPlot.png')


#LM p-value > 5%. therefore reject n0, therefore het exist

model_HW = sm.OLS(api_avg.to_numpy(),  sm.add_constant(X_pca_all[factor_selected])).fit(cov_type='HC1',use_t=None)
print(model_HW.summary())
export_summ(model_HW.summary(),'summ_HW')

df_BPTest=pd.DataFrame(index=['Lagrange Multiplier Statistic', 'LM p-value','F-Statistic', 'F p-value'])
df_BPTest['constantError']=sms.het_breuschpagan(model_opt.resid,model_opt.model.exog)
df_BPTest['HWError']=sms.het_breuschpagan(model_HW.resid,model_HW.model.exog)

(model_HW.pvalues<0.05).sum()

#perform Durbin-Watson test
durbin_watson(model_opt.resid)

