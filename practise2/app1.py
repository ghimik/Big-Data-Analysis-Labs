import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from datetime import datetime
from PIL import Image

st.set_page_config(page_title="UEFA Champions League Analysis", layout="wide")

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

tab1, tab2, tab3 = st.tabs(["Главная", "EDA", "ML"])

with tab1:
    st.title("Анализ данных UEFA Champions League")
    st.write("""
    Это проектная работа по анализу данных UEFA Champions League.
    Здесь представлены таблицы, графики и статистика игроков и команд.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Игроки", len(players))
        st.metric("Матчи", len(matches))
    with col2:
        st.metric("Голы", len(goals))
        st.metric("Менеджеры", len(managers))
    with col3:
        st.metric("Стадионы", len(stadiums))
        st.metric("Команды", len(teams))

    with st.expander("Таблицы проекта"):
        tab1_1, tab1_2, tab1_3, tab1_4, tab1_5, tab1_6 = st.tabs(["Игроки", "Голы", "Матчи", "Менеджеры", "Стадионы", "Команды"])
        with tab1_1:
            st.dataframe(players)
        with tab1_2:
            st.dataframe(goals)
        with tab1_3:
            st.dataframe(matches)
        with tab1_4:
            st.dataframe(managers)
        with tab1_5:
            st.dataframe(stadiums)
        with tab1_6:
            st.dataframe(teams)

with tab2:
    st.title("Exploratory Data Analysis")
    
    eda_tabs = st.tabs(["Общая статистика", "По игроку", "По командам", "Дополнительно"])
    
    with eda_tabs[0]:
        st.subheader("Количество голов по матчам и игрокам")
        total_goals = goals.merge(players, left_on="PID", right_on="PLAYER_ID")
        st.write("Общее количество голов:", len(total_goals))
        
        st.subheader("Топ команд по забитым голам")
        team_goals = goals.merge(players, left_on="PID", right_on="PLAYER_ID")
        team_goals_count = team_goals.groupby("TEAM").agg(goals_scored=("GOAL_ID", "count")).reset_index()
        team_goals_count = team_goals_count.sort_values("goals_scored", ascending=False).head(10)
        st.dataframe(team_goals_count)
        
        st.subheader("Топ команд по сезонам")
        season_team_goals = goals.merge(players, left_on="PID", right_on="PLAYER_ID") \
                                  .merge(matches, left_on="MATCH_ID", right_on="MATCH_ID")
        season_team_count = season_team_goals.groupby(["SEASON", "TEAM"]).agg(goals=("GOAL_ID", "count")).reset_index()
        season_team_count = season_team_count.sort_values("goals", ascending=False).head(10)
        st.dataframe(season_team_count)
    
    with eda_tabs[1]:
        default_player_id = 'ply1479'
        player_id = st.text_input("Введите PLAYER_ID для анализа:", value=default_player_id)
        
        player_goals = goals[goals["PID"] == player_id]
        player_info = players[players["PLAYER_ID"] == player_id]
        
        if not player_info.empty:
            col1, col2, col3 = st.columns(3)
            full_name = f"{player_info.iloc[0]['FIRST_NAME']} {player_info.iloc[0]['LAST_NAME']}"
            goals_scored = len(player_goals)
            avg_minute = player_goals["DURATION"].mean() if not player_goals.empty else 0
            
            with col1:
                st.metric("Игрок", full_name)
            with col2:
                st.metric("Забито голов", goals_scored)
            with col3:
                st.metric("Средняя минута гола", f"{avg_minute:.2f}")
        else:
            st.warning("Игрок с таким PLAYER_ID не найден.")
    
    with eda_tabs[2]:
        st.subheader("Среднее количество голов за матч по командам")
        home_avg = matches.groupby("HOME_TEAM")["HOME_TEAM_SCORE"].mean().reset_index()
        away_avg = matches.groupby("AWAY_TEAM")["AWAY_TEAM_SCORE"].mean().reset_index()
        team_avg_goals = home_avg.merge(away_avg, left_on="HOME_TEAM", right_on="AWAY_TEAM")
        team_avg_goals["avg_goals"] = (team_avg_goals["HOME_TEAM_SCORE"] + team_avg_goals["AWAY_TEAM_SCORE"]) / 2
        team_avg_goals = team_avg_goals[["HOME_TEAM", "avg_goals"]].sort_values("avg_goals", ascending=False).head(10)
        st.dataframe(team_avg_goals)
    
    with eda_tabs[3]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Средний возраст игроков, забивших пенальти")
            penalty_goals = goals[goals["GOAL_DESC"].str.contains("penalty", case=False, na=False)]
            penalty_players = players.merge(penalty_goals, left_on="PLAYER_ID", right_on="PID")
            penalty_players["DOB"] = pd.to_datetime(penalty_players["DOB"], errors='coerce')
            if not penalty_players.empty:
                penalty_players["AGE"] = (pd.Timestamp.today() - penalty_players["DOB"]).dt.days / 365.25
                avg_age = penalty_players["AGE"].mean()
                st.metric("Средний возраст", f"{avg_age:.1f} лет")
            else:
                st.write("Данных нет.")
        
        with col2:
            st.subheader("Максимальный вес игроков по позициям")
            max_weight = players.groupby("POSITION").agg(max_weight=("WEIGHT", "max")).reset_index()
            max_weight = max_weight.sort_values("max_weight", ascending=False)
            st.dataframe(max_weight)

with tab3:
    st.title("Machine Learning")
    
    ml_tabs = st.tabs(["Датасет", "Модель", "Предсказание"])
    
    with ml_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Обучающая выборка", int(2279 * 0.8))
        with col2:
            st.metric("Тестовая выборка", int(2279 * 0.2))
    
    with ml_tabs[1]:
        st.subheader("Метрики модели KNN")
        
        metrics_df = pd.DataFrame({
            "Метрика": ["MAE", "MSE", "RMSE", "MAPE", "R2"],
            "Значение": ["8300.60", "177964266.50", "13340.32", "7417.35%", "0.70"]
        })
        st.dataframe(metrics_df)
        
        try:
            img = Image.open("model_info.png")
            st.image(img, caption="Статистика модели KNN")
        except FileNotFoundError:
            st.info("Файл model_info.png не найден в папке проекта.")
    
    with ml_tabs[2]:
        st.subheader("Предсказание посещаемости матча")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                season = st.text_input("Сезон", value="2020-2021")
                home_team = st.text_input("Домашняя команда", value="Borussia Dortmund")
            with col2:
                away_team = st.text_input("Гостевая команда", value="Manchester City")
                stadium = st.text_input("Стадион", value="Signal Iduna Park")
            
            submitted = st.form_submit_button("Предсказать посещаемость")
            
            if submitted:
                try:
                    with open("predict_attendace_on_season_hometeam_awaytem.pkl", "rb") as f:
                        model = load(f)
                    
                    input_df = pd.DataFrame([{
                        "SEASON": season,
                        "HOME_TEAM": home_team,
                        "AWAY_TEAM": away_team,
                        "STADIUM": stadium
                    }])
                    predicted_attendance = model.predict(input_df)[0]
                    st.success(f"Ожидаемая посещаемость: {int(predicted_attendance)} человек")
                except:
                    st.error("Ошибка при загрузке модели или предсказании.")