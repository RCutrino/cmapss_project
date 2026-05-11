import pandas as pd

def add_rul(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    max_cycle = (df.groupby("engine_id")["cycle"].max().reset_index())
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    return df
