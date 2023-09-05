import pandas as pd
import numpy as np
from os import system

def clear():
    system('clear')


df = pd.read_csv('./eurusd-5m.csv', sep=';')

closes = df['close'].to_numpy(dtype=np.float32)

up = [0]
down = [0]
same = [0]

for i in range(len(closes)):
    if i == 0:
        continue
    else:
        if closes[i] > closes[i-1]:
            up.append(1)
            down.append(0)
            same.append(0)
        elif closes[i] < closes[i-1]:
            up.append(0)
            down.append(1)
            same.append(0)
        else:
            up.append(0)
            down.append(0)
            same.append(0)

    print(f"{i}/{len(closes)}")

df['up'] = pd.Series(up)
df['down'] = pd.Series(down)
df['same'] = pd.Series(same)
df.to_csv('./new_data.csv', sep=';')
