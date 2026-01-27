from pathlib import Path
import pandas as pd
import math
from datetime import datetime

EXCEL_PATH = Path("UEFA Champions League 2016-2022 Data 3.xlsx")
OUTPUT_DIR = Path("csv_output")  
TABLE_ORDER = [
    "stadiums",
    "teams",
    "players",
    "managers",
    "matches",
    "goals",
]

def normalize_value(v, col_name=None):
    """Нормализует значения, сохраняет формат дат/NaT/NaN"""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NaT:
        return None

    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    
    if col_name and ('DATE' in col_name.upper() or 'DOB' in col_name.upper()):
        if isinstance(v, str):
            try:
                return pd.to_datetime(v, format='%d-%b-%y %I.%M.%S.%f %p', errors='coerce')
            except:
                try:
                    return pd.to_datetime(v, errors='coerce')
                except:
                    return None
    return v

def sheet_to_dataframe(sheet_name: str) -> pd.DataFrame:
    """Считывает лист Excel и нормализует значения"""
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)

    for col in df.columns:
        df[col] = df[col].apply(lambda v: normalize_value(v, col))
    
    return df

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for table in TABLE_ORDER:
        df = sheet_to_dataframe(table)
        out_path = OUTPUT_DIR / f"{table}.csv"

        df.to_csv(out_path, index=False)
        print(f"[OK] {table} -> {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
