import pandas as pd
import math

def normalize_value(v, col_name=None):
    """Нормализует значения для вставки в БД"""
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


def get_data(excel_path: str) -> pd.DataFrame:

    goals = pd.read_excel(excel_path, sheet_name="goals")
    matches = pd.read_excel(excel_path, sheet_name="matches")
    players = pd.read_excel(excel_path, sheet_name="players")

    df = goals.merge(matches, on="MATCH_ID", how="left")
    df = df.merge(players, left_on="PID", right_on="PLAYER_ID", how="left")

    df["DATE_TIME"] = df["DATE_TIME"].apply(lambda x: normalize_value(x, "DATE_TIME"))
    df["DOB"] = df["DOB"].apply(lambda x: normalize_value(x, "DOB"))

    df["IS_PENALTY"] = df["GOAL_DESC"].str.contains("penalty", case=False, na=False).astype(int)
    df["FIRST_HALF_GOAL"] = (df["DURATION"] <= 45).astype(int)
    df["PLAYER_AGE"] = (df["DATE_TIME"] - df["DOB"]).dt.days / 365.25
    df["TOTAL_MATCH_GOALS"] = df["HOME_TEAM_SCORE"] + df["AWAY_TEAM_SCORE"]
    df["HOME_GOAL"] = (df["TEAM"] == df["HOME_TEAM"]).astype(int)

    return df
