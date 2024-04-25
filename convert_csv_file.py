import pandas as pd

df = pd.read_csv("/scratch/lg3490/tfv/count.csv",header=0)
df.iloc[:,1:].astype(float)
df.iloc[:,1:] = df.iloc[:,1:].apply(lambda x: (x/x.sum()), axis=1).copy()
df.to_csv("/scratch/lg3490/tfv/count_parsed.csv")