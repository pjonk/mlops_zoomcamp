#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('uv pip show scikit-learn')


# In[7]:


get_ipython().system('python --version')


# In[27]:


import pickle
import pandas as pd
from pathlib import Path


# In[11]:


with open('artifacts/model.pkl', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[12]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# In[14]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[15]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[20]:


print(f'The standard dev of the predictions is: {y_pred.std()}')


# ## Question 2

# In[41]:


year = 2023
month = 3

output_folder = Path('data')
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / f'{year:04d}-{month:02d}_predictions.parquet' 

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df['predictions'] = y_pred

df_result = df[['ride_id', 'predictions']]

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[43]:


get_ipython().system('ls -lh data')


# ## Question 3

# In[ ]:




