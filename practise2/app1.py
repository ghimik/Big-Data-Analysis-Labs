import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import load
from datetime import datetime
from PIL import Image
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    players = pd.read_csv("csv_output/players.csv")
    goals = pd.read_csv("csv_output/goals.csv")
    matches = pd.read_csv("csv_output/matches.csv")
    managers = pd.read_csv("csv_output/managers.csv")
    stadiums = pd.read_csv("csv_output/stadiums.csv")
    teams = pd.read_csv("csv_output/teams.csv")
    return players, goals, matches, managers, stadiums, teams

players, goals, matches, managers, stadiums, teams = load_data()

st.sidebar.title("Меню")
page = st.sidebar.selectbox("Выберите страницу:", ["Главная", "EDA", "ML"])

if page == "Главная":
    st.title("Анализ данных UEFA Champions League")
    st.write("""
    Это проектная работа по анализу данных UEFA Champions League.
    Здесь представлены таблицы, графики и статистика игроков и команд.
    """)

    st.subheader("Статистика данных")
    st.write("Количество игроков:", len(players))
    st.write("Количество матчей:", len(matches))
    st.write("Количество голов:", len(goals))
    st.write("Количество менеджеров:", len(managers))
    st.write("Количество стадионов:", len(stadiums))
    st.write("Количество команд:", len(teams))

    st.subheader("Таблицы проекта")
    st.write("### Игроки")
    st.dataframe(players)

    st.write("### Голы")
    st.dataframe(goals)

    st.write("### Матчи")
    st.dataframe(matches)

    st.write("### Менеджеры")
    st.dataframe(managers)

    st.write("### Стадионы")
    st.dataframe(stadiums)

    st.write("### Команды")
    st.dataframe(teams)


elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    default_player_id = 'ply1479'
    player_id = st.text_input("Введите PLAYER_ID для анализа:", value=default_player_id)

    st.subheader("Топ команд по забитым голам")
    team_goals = goals.merge(players, left_on="PID", right_on="PLAYER_ID")
    team_goals_count = team_goals.groupby("TEAM").agg(goals_scored=("GOAL_ID", "count")).reset_index()
    team_goals_count = team_goals_count.sort_values("goals_scored", ascending=False).head(10)
    st.dataframe(team_goals_count)

    st.subheader("Топ команд по сезонам по количеству голов")
    season_team_goals = goals.merge(players, left_on="PID", right_on="PLAYER_ID") \
                              .merge(matches, left_on="MATCH_ID", right_on="MATCH_ID")
    season_team_count = season_team_goals.groupby(["SEASON", "TEAM"]).agg(goals=("GOAL_ID", "count")).reset_index()
    season_team_count = season_team_count.sort_values("goals", ascending=False).head(10)
    st.dataframe(season_team_count)


    st.subheader(f"Статистика по игроку {player_id}")
    player_goals = goals[goals["PID"] == player_id]
    player_info = players[players["PLAYER_ID"] == player_id]
    if not player_info.empty:
        full_name = f"{player_info.iloc[0]['FIRST_NAME']} {player_info.iloc[0]['LAST_NAME']}"
        goals_scored = len(player_goals)
        avg_minute = player_goals["DURATION"].mean() if not player_goals.empty else 0
        st.write(f"Игрок: {full_name}")
        st.write(f"Забито голов: {goals_scored}")
        st.write(f"Средняя минута гола: {avg_minute:.2f}")
    else:
        st.warning("Игрок с таким PLAYER_ID не найден.")

    st.subheader("Количество голов по матчам и игрокам")
    total_goals = goals.merge(players, left_on="PID", right_on="PLAYER_ID")
    st.write("Количество голов:", len(total_goals))

    st.subheader("Средний возраст игроков, забивших пенальти")
    penalty_goals = goals[goals["GOAL_DESC"].str.contains("penalty", case=False, na=False)]
    penalty_players = players.merge(penalty_goals, left_on="PLAYER_ID", right_on="PID")
    penalty_players["DOB"] = pd.to_datetime(penalty_players["DOB"], errors='coerce')
    if not penalty_players.empty:
        penalty_players["AGE"] = (pd.Timestamp.today() - penalty_players["DOB"]).dt.days / 365.25
        avg_age = penalty_players["AGE"].mean()
        st.write(f"Средний возраст: {avg_age:.1f} лет")
    else:
        st.write("Данных нет.")

    st.subheader("Максимальный вес игроков по позициям")
    max_weight = players.groupby("POSITION").agg(max_weight=("WEIGHT", "max")).reset_index()
    max_weight = max_weight.sort_values("max_weight", ascending=False)
    st.dataframe(max_weight)

    st.subheader("Среднее количество голов за матч по командам")
    team_avg_goals = matches.groupby("HOME_TEAM").agg(avg_goals=("HOME_TEAM_SCORE", "mean")).reset_index()
    team_avg_goals["avg_goals"] += matches.groupby("AWAY_TEAM")["AWAY_TEAM_SCORE"].mean().values
    team_avg_goals = team_avg_goals.sort_values("avg_goals", ascending=False).head(10)
    st.dataframe(team_avg_goals)



elif page == "ML":
    st.subheader("Обучающая и тестовая выборка (для демонстрации ML)")

    st.write(f"Размер обучающей выборки: {int(2279 * 0.8)}")
    st.write(f"Размер тестовой выборки: {int(2279 * 0.2)}")

    st.write("### Метрики модели KNN (attendance prediction)")
    st.write("""
        - MAE: 8300.6039
        - MSE: 177964266.4950
        - RMSE: 13340.3248
        - MAPE: 7417.345%
        - R2: 0.6979
    """)


    st.write("### Визуализация работы модели")
    try:
        img = Image.open("model_info.png")
        st.image(img, caption="Статистика модели KNN")
    except FileNotFoundError:
        st.warning("Файл model_info.png не найден в папке проекта.")

    st.write("### Предсказание посещаемости матча")

    with open("predict_attendace_on_season_hometeam_awaytem.pkl", "rb") as f:
        model = load(f)

    season = st.text_input("Сезон", value="2020-2021")
    home_team = st.text_input("Домашняя команда", value="Borussia Dortmund")
    away_team = st.text_input("Гостевая команда", value="Manchester City")
    stadium = st.text_input("Стадион", value="Signal Iduna Park")

    if st.button("Предсказать посещаемость"):
        if model is not None:
            input_df = pd.DataFrame([{
                "SEASON": season,
                "HOME_TEAM": home_team,
                "AWAY_TEAM": away_team,
                "STADIUM": stadium
            }])
            predicted_attendance = model.predict(input_df)[0]
            st.success(f"Ожидаемая посещаемость: {int(predicted_attendance)} человек")
        else:
            st.error("Модель не загружена, предсказание невозможно.")


