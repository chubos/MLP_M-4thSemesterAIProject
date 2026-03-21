import pandas as pd

df = pd.read_csv('house-votes-84.data', header=None)

# Mapowanie wartości dla całej tabeli
df = df.replace({'y': 1, 'n': -1, '?': 0})

# Mapowanie nazw partii w pierwszej kolumnie
df[0] = df[0].replace({'democrat': 0, 'republican': 1})

df.to_csv('transformed_votes.csv', index=False, header=False)
