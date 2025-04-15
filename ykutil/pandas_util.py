import pandas as pd


def expand_json_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df = pd.concat([df, pd.json_normalize(df[col])], axis=1)
        df = df.drop(columns=[col])
    return df
