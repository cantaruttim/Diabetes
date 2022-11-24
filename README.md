# Diabetes - Modelos de Regressão Linear

Este projeto teve como objetivo aplicar modelos de machine learning, utilizando a biblioteca scikit-learn, carregando o banco: Diabetes Database (https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
 
### Linguagens:

<div>
  <img align="center" alt="Canta-Python" height="30" width="40" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" />
  <img align="center" alt="Canta-Jupyter" height="30" width="40" src="https://devicons.railway.app/i/jupyter.svg" />
<div />

### Procedimentos:

Para realizar as primeiras análises foram utilizadas as seguintes bibliotecas

 ```
 
 import pandas as pd
 import seaborn as sns
 %matplotlib inline
 import numpy as np
 
 ```
 
 Em diabetes.info() observamos as caracteristicas desse dataset contento as variáveis aplicadas, utiilizando o Z-score
 
 ```
 >>> <class 'pandas.core.frame.DataFrame'>
RangeIndex: 442 entries, 0 to 441
Data columns (total 11 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   age     442 non-null    float64
 1   sex     442 non-null    float64
 2   bmi     442 non-null    float64
 3   map     442 non-null    float64
 4   tc      442 non-null    float64
 5   ldl     442 non-null    float64
 6   hdl     442 non-null    float64
 7   tch     442 non-null    float64
 8   ltg     442 non-null    float64
 9   glu     442 non-null    float64
 10  y       442 non-null    int64  
dtypes: float64(10), int64(1)
memory usage: 38.1 KB
```

Ao observar as correlações entre as variáveis, obtive esse gráfico

![image](https://user-images.githubusercontent.com/81988636/203664010-48e25f02-f49e-4647-b9ec-7756fc61604a.png)

As variáveis que me chamaram mais atenção ao realizar a análise das correlações, foram as idade (`age`), colesterol total (`tc`), o valor de LDL (`ldl`), o valor de HDL (`hdl`) e o índice de glicose plasmática (`glu`)

A descrição de cada variáveis segue conforme a documentação:

- age age in years

- sex

- bmi body mass index

- bp average blood pressure

- s1 tc, total serum cholesterol

- s2 ldl, low-density lipoproteins

- s3 hdl, high-density lipoproteins

- s4 tch, total cholesterol / HDL

- s5 ltg, possibly log of serum triglycerides level

- s6 glu, blood sugar level

### Gráficos das variávies :

- [ ] Para idade

![image](https://user-images.githubusercontent.com/81988636/203664841-9a2152b3-d1eb-4d37-822e-38bbd9ca1785.png)

- [ ] Para o valor total de colesterol 

![image](https://user-images.githubusercontent.com/81988636/203664939-a5d5c4f0-a386-49d5-a247-6083cada8e42.png)

- [ ] Para o valor de LDL (Colesterol Ruim :confused: ) 

![image](https://user-images.githubusercontent.com/81988636/203664986-72ada936-b1da-4f45-86c9-90da3da3888f.png)

- [ ] Para o valor de HDL (Colesterol Bom :scream: ) <br />
_Uma observação importante para esse gráfico é a sua relação inversa com os gráfico anteriores_ Pacientes que apresentaram HDL alto tiveram menos chances de apresentar níveis de glicose elevados [Valores de `y` dentro dos valores normais]

![image](https://user-images.githubusercontent.com/81988636/203665220-25495128-40a8-4862-bb12-5488845dea62.png)

- [ ] Para os níveis de glicose plasmáticas

![image](https://user-images.githubusercontent.com/81988636/203665436-c5731c68-1b94-4dd4-a533-e362ae81cfae.png)

### Prepando as variáveis para treinar os modelos

Observe o gráfico a seguir: 

`ns.lmplot(data=diabetes, x='tc', y='y', height=5)`

![image](https://user-images.githubusercontent.com/81988636/203665682-464986b7-2d14-456b-8dba-2ff9c084655e.png)

_Esse gráfico mostra uma relação linear entre as variáveis `tc` com a variável `y`_

E olhe que interessante:

`ns.lmplot(data=diabetes, x='hdl', y='y', height=5)`

![image](https://user-images.githubusercontent.com/81988636/203666012-2ba95485-ca45-437c-a9b9-6851cea336be.png)

_Esse gráfico mostra uma relação inversa com os os valores de `hdl` e os valores de `y`_


Esses gráficos ilustram muito bem qual o tipo de problema que estamos enfrentando. Um problema de correlação e como podemos prever utilizando algoritmos de _Regressão Linear_

A variável X obteve os valores : `X = diabetes.iloc[:, 0:10].values` do DataSet
A variável y obeteve apenas o valore da coluna `y` :`y = diabetes.iloc[:, 10].values`


#### Modelos

```
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

### Resultados

Para cada modelo, obteve-se os valores de:

- Linear Regression: reg.score(X, y) = 0.5177494254132934
- Ridge : clf.score(X,y) = 0.45123139467990536
- RidgeCV : clf_cv.score(X, y) = 0.5166287840315835
- SGDRegressor : reg_sdg.score(X, y) = 0.5138156128090378
