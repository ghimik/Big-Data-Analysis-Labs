import traceback
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from joblib import load
from datetime import datetime
from typing import List, Optional
from pycaret.regression import setup, compare_models, finalize_model, save_model, load_model, pull, get_logs, predict_model


app = FastAPI(title="UEFA Champions League Data API")

players = pd.read_csv("csv_output/players.csv")
goals = pd.read_csv("csv_output/goals.csv")
matches = pd.read_csv("csv_output/matches.csv")
managers = pd.read_csv("csv_output/managers.csv")
stadiums = pd.read_csv("csv_output/stadiums.csv")
teams = pd.read_csv("csv_output/teams.csv")

try:
    model = load("predict_attendace_on_season_hometeam_awaytem.pkl")
except FileNotFoundError:
    model = None



class MatchPredictionRequest(BaseModel):
    SEASON: str
    HOME_TEAM: str
    AWAY_TEAM: str
    STADIUM: Optional[str] = None

class AutoMLTrainRequest(BaseModel):
    target: str = "ATTENDANCE"
    categorical_features: list[str] = ["SEASON","HOME_TEAM","AWAY_TEAM","STADIUM"]
    normalize: bool = True

class AutoMLPredictRequest(BaseModel):
    SEASON: str
    HOME_TEAM: str
    AWAY_TEAM: str
    STADIUM: Optional[str] = None
    model_name: str = "attendance_automl_model"


def df_to_json(df: pd.DataFrame):
    df = df.replace({pd.NA: None, float("nan"): None})
    return df.to_dict(orient="records")



@app.post("/train_automl")
def train_automl(request: AutoMLTrainRequest):
    try:
        df = pd.read_csv("ucl_matches_parsed.csv")
        df = df[request.categorical_features + [request.target]]
        for col in request.categorical_features:
            df[col] = df[col].astype(str)

        exp = setup(
            data=df,
            target=request.target,
            categorical_features=request.categorical_features,
            normalize=request.normalize,
        )

        best_model = compare_models()
        final_model = finalize_model(best_model)
        model_name: str = f"attendance_automl_model_{str(uuid.uuid4())}"

        save_model(final_model, model_name)

        metrics = pull().to_dict(orient="records")
        logs = get_logs().to_dict(orient="records")

        return {
            "status": "success",
            "model_name": model_name,
            "metrics": metrics,
            "logs": logs
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_automl")
def predict_automl(request: AutoMLPredictRequest):
    try:
        model = load_model(request.model_name)

        input_df = pd.DataFrame([{
            "SEASON": request.SEASON,
            "HOME_TEAM": request.HOME_TEAM,
            "AWAY_TEAM": request.AWAY_TEAM,
            "STADIUM": request.STADIUM
        }])

        pred = predict_model(model, data=input_df)
        predicted_value = float(pred["prediction_label"][0])

        return {"predicted_attendance": int(predicted_value)}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/players")
def get_players():
    return df_to_json(players)

@app.get("/goals")
def get_goals():
    return df_to_json(goals)

@app.get("/matches")
def get_matches():
    return df_to_json(matches)

@app.get("/managers")
def get_managers():
    return df_to_json(managers)

@app.get("/stadiums")
def get_stadiums():
    return df_to_json(stadiums)

@app.get("/teams")
def get_teams():
    return df_to_json(teams)

@app.get("/eda/top_teams")
def top_teams_goals(limit: int = 10):
    df = goals.merge(players, left_on="PID", right_on="PLAYER_ID")
    result = df.groupby("TEAM").agg(goals_scored=("GOAL_ID", "count")).reset_index()
    result = result.sort_values("goals_scored", ascending=False).head(limit)
    return result.to_dict(orient="records")

@app.get("/eda/top_teams_season")
def top_teams_season_goals(limit: int = 10):
    df = goals.merge(players, left_on="PID", right_on="PLAYER_ID") \
              .merge(matches, on="MATCH_ID")
    result = df.groupby(["SEASON", "TEAM"]).agg(goals=("GOAL_ID", "count")).reset_index()
    result = result.sort_values("goals", ascending=False).head(limit)
    return result.to_dict(orient="records")

@app.get("/eda/player_stats/{player_id}")
def player_stats(player_id: str):
    player_info = players[players["PLAYER_ID"] == player_id]
    if player_info.empty:
        raise HTTPException(status_code=404, detail="Player not found")
    player_goals = goals[goals["PID"] == player_id]
    full_name = f"{player_info.iloc[0]['FIRST_NAME']} {player_info.iloc[0]['LAST_NAME']}"
    goals_scored = len(player_goals)
    avg_minute = (
        float(player_goals["DURATION"].mean())
        if not player_goals.empty else 0.0
    )
    return {
        "player_id": player_id,
        "full_name": full_name,
        "goals_scored": goals_scored,
        "avg_goal_minute": avg_minute
    }

@app.post("/predict")
def predict_attendance(request: MatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    input_df = pd.DataFrame([request.dict()])
    predicted = model.predict(input_df)[0]
    return {"predicted_attendance": int(predicted)}

@app.get("/eda/total_goals")
def total_goals():
    df = goals.merge(players, left_on="PID", right_on="PLAYER_ID")
    return {
        "total_goals": int(len(df))
    }


@app.get("/eda/penalty_avg_age")
def penalty_avg_age():
    penalty_goals = goals[
        goals["GOAL_DESC"].str.contains("penalty", case=False, na=False)
    ]

    if penalty_goals.empty:
        return {"avg_age": None, "message": "No penalty goals"}

    penalty_players = players.merge(
        penalty_goals, left_on="PLAYER_ID", right_on="PID"
    )

    penalty_players["DOB"] = pd.to_datetime(
        penalty_players["DOB"], errors="coerce"
    )

    penalty_players = penalty_players.dropna(subset=["DOB"])

    if penalty_players.empty:
        return {"avg_age": None, "message": "No valid birth dates"}

    penalty_players["AGE"] = (
        (pd.Timestamp.today() - penalty_players["DOB"]).dt.days / 365.25
    )

    avg_age = float(penalty_players["AGE"].mean())

    return {
        "avg_age": round(avg_age, 1)
    }

@app.get("/eda/max_weight_by_position")
def max_weight_by_position():
    result = (
        players.groupby("POSITION")
        .agg(max_weight=("WEIGHT", "max"))
        .reset_index()
        .sort_values("max_weight", ascending=False)
    )

    return df_to_json(result)


@app.get("/eda/team_avg_goals")
def team_avg_goals(limit: int = 10):
    home = (
        matches.groupby("HOME_TEAM")
        .agg(home_goals=("HOME_TEAM_SCORE", "mean"))
        .reset_index()
        .rename(columns={"HOME_TEAM": "TEAM"})
    )

    away = (
        matches.groupby("AWAY_TEAM")
        .agg(away_goals=("AWAY_TEAM_SCORE", "mean"))
        .reset_index()
        .rename(columns={"AWAY_TEAM": "TEAM"})
    )

    df = home.merge(away, on="TEAM", how="outer").fillna(0)
    df["avg_goals"] = df["home_goals"] + df["away_goals"]

    df = df.sort_values("avg_goals", ascending=False).head(limit)

    return df_to_json(df[["TEAM", "avg_goals"]])
