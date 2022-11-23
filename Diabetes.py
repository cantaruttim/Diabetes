#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


diabetes = pd.read_csv("diabetes.rwrite1.txt", encoding= "utf_8", delimiter=" ")
diabetes


# In[3]:


diabetes.info()


# In[4]:


diabetes.mean(), np.std(diabetes)


# In[5]:


sns.countplot(diabetes['age'])


# In[6]:


sns.countplot(diabetes['tc'])


# In[7]:


sns.countplot(diabetes['ldl'])


# In[8]:


sns.countplot(diabetes['hdl'])


# In[9]:


sns.countplot(diabetes['glu'])


# In[10]:


sns.set_theme(style="ticks")
sns.pairplot(diabetes, hue = "age")


# In[11]:


# Observando correlação entre as variáveis do dataframe

sns.set_theme(style="whitegrid")

corr_mat = diabetes.corr().stack().reset_index(name="correlation")

g = sns.relplot(
    data=corr_mat,
    x="level_0", y="level_1", hue="correlation", size="correlation",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(50, 250), size_norm=(-.2, .8),
)


g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
for artist in g.legend.legendHandles:
    artist.set_edgecolor(".7")


# In[12]:


diabetes.shape


# In[13]:


# selecionamos as colunas de 1:4 porém a coluna 4 não entrará nessa variável.
X = diabetes.iloc[:, 0:10].values
X


# In[14]:


y = diabetes.iloc[:, 10].values
y


# In[15]:


# Treinando os modelos
from sklearn.linear_model import LinearRegression


# In[16]:


reg =LinearRegression().fit(X, y)


# In[23]:


reg.score(X, y), reg.coef_, reg.intercept_


# In[22]:


reg.predict(X)


# In[29]:


sns.set_theme(style="ticks")


# In[54]:


sns.lmplot(data=diabetes, x='tc', y='y', height=5)


# In[38]:


from sklearn.linear_model import Ridge


# In[40]:


rng = np.random.random(0)
y = y
X = X


# In[41]:


clf = Ridge(alpha=1.0)
clf.fit(X, y)


# In[43]:


clf.predict(X) 


# In[46]:


clf.score(X,y)


# In[47]:


from sklearn.linear_model import RidgeCV


# In[49]:


X = X
y = y 
clf_cv = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
clf_cv.score(X, y)


# In[76]:


from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[78]:


rng_sdg = np.random.RandomState(0)
X = X
y = y 
reg_sdg = make_pipeline(StandardScaler(),
                       SGDRegressor(max_iter=1000, tol=1e-3))
reg_sdg.fit(X, y)


# In[79]:


reg_sdg.predict(X)


# In[80]:


reg_sdg.score(X, y)

