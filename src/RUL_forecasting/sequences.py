import pandas as pd
import numpy as np

def make_sequences(df, sensors, target, seq_len):
    X, y = [], []
    for engine in df['engine_id'].unique():
        eng = df[df['engine_id'] == engine].sort_values('cycle')
        vals = eng[sensors].values
        ruls = eng[target].values
        for i in range(len(vals) - seq_len + 1):
            X.append(vals[i : i + seq_len])
            y.append(ruls[i + seq_len - 1])
    
    return np.array(X), np.array(y)
