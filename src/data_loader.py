import pandas as pd

COLUMNS = (["engine_id", "cycle"]
            + [f"setting_{i}" for i in range(1, 4)]
            + [f"sensor_{i}" for i in range(1, 22)])

def load_train_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path,
                    sep=r"\s+",
                    header=None)

    df.columns = COLUMNS

    return df
