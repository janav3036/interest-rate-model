import pandas as pd
from pathlib import Path

_DEFAULT_CSV = Path(__file__).parent / "gsec_yields.csv"

TENORS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
TENOR_COLS = ["tenor_3m", "tenor_6m", "tenor_1y", "tenor_2y", "tenor_5y", "tenor_10y"]

def load_cached_yields(filepath: str = None) -> pd.DataFrame:
    path = Path(filepath) if filepath else _DEFAULT_CSV
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def get_yield_snapshot(df: pd.DataFrame, date: str = None) -> pd.Series:
    if date is None:
        row = df.iloc[-1]
    else:
        row = df[df["date"] == pd.Timestamp(date)].iloc[0]
    return row[TENOR_COLS]

