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
df.describe()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/c06b618b-fb44-4e2e-b223-095f50b9a725)


```
df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/aaffd845-b9fe-4704-b3ae-7b4f3f5c8d03)

```
sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/e68528c8-6c78-42b2-9068-cdc9f1e3c647)


```
sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/097979c0-803e-4618-b6ff-93cd9fe4cf04)

```
df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/79b1262a-e785-4a19-9bac-3f71f1728220)

```
df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/df956f1e-c9b7-4894-bef1-693fba72b777)

```
df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/f169c9d0-22c2-4647-b41b-ba52b393bce2)

```
df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/163cfc21-d009-47f8-954d-398107ea472f)

```
from sklearn.preprocessing import PowerTransformer
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/b8610549-adb5-4cbc-9bfc-1bbd08420b28)

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
![image](https://github.com/madhi43/ODD2023-Datascience-Ex06/assets/103943383/7a478387-3797-4c60-98fb-3d1524c56e53)

```

# RESULT

Thus feature transformation is done for the given set
