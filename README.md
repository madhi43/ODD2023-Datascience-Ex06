# ODD2023-Datascience-Ex06

# AIM 

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM

STEP 1: Read the given Data

STEP 2: Clean the Data Set using Data Cleaning Process

STEP 3: Apply Feature Transformation techniques to all the features of the data set

STEP 4: Print the transformed features



# Program
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/d0a1db73-66c5-4e2e-8f1f-7ba5f4ce77b0)

```
df.head()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/0d338a48-af55-4a2d-af12-e3ccd6cc4baa)

```
df.isnull().sum()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/1ca5d263-87c4-4ef4-971d-9b5b31f2dc67)

```
df.info()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/504c7303-c8eb-4473-aad2-003637b1e79c)


```
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()
```

```
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()
```

```
df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
```

```
df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
```

```
df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
```

```
df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()
```

```
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
```

```
from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))
```

```
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
```
```
df2=df.copy()
```
```
df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
```
