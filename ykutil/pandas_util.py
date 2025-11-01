import pandas as pd
from pathlib import Path


def expand_json_cols(df: pd.DataFrame, cols: list[str] | str) -> pd.DataFrame:
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        df = pd.concat([df, pd.json_normalize(df[col])], axis=1)
        df = df.drop(columns=[col])
    return df


def df_to_csv(filename: str):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffixes = [s.lower() for s in path.suffixes]

    # Load dataframe based on file extension(s)
    if ".tsv" in suffixes:
        df = pd.read_csv(path, sep="\t")
    elif ".csv" in suffixes:
        df = pd.read_csv(path)
    elif ".parquet" in suffixes or ".pq" in suffixes:
        df = pd.read_parquet(path)
    elif ".feather" in suffixes:
        df = pd.read_feather(path)
    elif ".pkl" in suffixes or ".pickle" in suffixes:
        df = pd.read_pickle(path)
    elif ".jsonl" in suffixes or ".ndjson" in suffixes:
        df = pd.read_json(path, lines=True)
    elif ".json" in suffixes:
        df = pd.read_json(path)
    elif ".xlsx" in suffixes or ".xls" in suffixes:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension for {path.name}")

    # Build output path by stripping all suffixes and appending .csv
    suffix_str = "".join(suffixes)
    base_name = path.name[: -len(suffix_str)] if suffix_str else path.name
    target_csv = path.with_name(base_name + ".csv")

    df.to_csv(target_csv, index=False)

    return str(target_csv)
