import streamlit as st
import requests
import pandas as pd
from PIL import Image

API_URL = "http://localhost:8000" 

def fetch_data(endpoint):
    try:
        response = requests.get(f"{API_URL}/{endpoint}")
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return pd.DataFrame()

def fetch_eda_top_teams(limit=10):
    try:
        response = requests.get(f"{API_URL}/eda/top_teams", params={"limit": limit})
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except:
        return pd.DataFrame()

def fetch_eda_top_teams_season(limit=10):
    try:
        response = requests.get(f"{API_URL}/eda/top_teams_season", params={"limit": limit})
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except:
        return pd.DataFrame()

def fetch_player_stats(player_id):
    try:
        response = requests.get(f"{API_URL}/eda/player_stats/{player_id}")
        response.raise_for_status()
        return response.json()
    except:
        return None

def fetch_total_goals():
    try:
        r = requests.get(f"{API_URL}/eda/total_goals")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка загрузки total_goals: {e}")
        return None


def fetch_penalty_avg_age():
    try:
        r = requests.get(f"{API_URL}/eda/penalty_avg_age")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка загрузки penalty_avg_age: {e}")
        return None


def fetch_max_weight_by_position():
    try:
        r = requests.get(f"{API_URL}/eda/max_weight_by_position")
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.error(f"Ошибка загрузки max_weight_by_position: {e}")
        return pd.DataFrame()


def fetch_team_avg_goals(limit=10):
    try:
        r = requests.get(
            f"{API_URL}/eda/team_avg_goals",
            params={"limit": limit}
        )
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.error(f"Ошибка загрузки team_avg_goals: {e}")
        return pd.DataFrame()


def predict_attendance(season, home_team, away_team, stadium):
    try:
        data = {
            "SEASON": season,
            "HOME_TEAM": home_team,
            "AWAY_TEAM": away_team,
            "STADIUM": stadium
        }
        response = requests.post(f"{API_URL}/predict", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")
        return None

tab1, tab2, tab3 = st.tabs(["Главная", "EDA", "ML"])
st.set_page_config(page_title="UEFA Champions League", layout="wide")

with tab1:
    st.title("Анализ данных UEFA Champions League")
    st.write("Клиентское приложение — все данные с FastAPI сервера")
    
    players_df = fetch_data("players")
    goals_df = fetch_data("goals")
    matches_df = fetch_data("matches")
    managers_df = fetch_data("managers")
    stadiums_df = fetch_data("stadiums")
    teams_df = fetch_data("teams")
    
    st.subheader("Статистика данных")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Игроки", len(players_df))
        st.metric("Матчи", len(matches_df))
    with col2:
        st.metric("Голы", len(goals_df))
        st.metric("Менеджеры", len(managers_df))
    with col3:
        st.metric("Стадионы", len(stadiums_df))
        st.metric("Команды", len(teams_df))
    
    st.subheader("Таблицы")
    tab_names = ["Игроки", "Голы", "Матчи", "Менеджеры", "Стадионы", "Команды"]
    tabs = st.tabs(tab_names)
    data_frames = [players_df, goals_df, matches_df, managers_df, stadiums_df, teams_df]
    
    for tab, df, name in zip(tabs, data_frames, tab_names):
        with tab:
            if not df.empty:
                st.dataframe(df)
            else:
                st.warning(f"Не удалось загрузить {name}")

with tab2:
    st.title("Exploratory Data Analysis")
    
    eda_tabs = st.tabs(["Команды", "Игроки", "Общая статистика"])
    
    with eda_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Топ команд по забитым голам")
            top_teams = fetch_eda_top_teams(10)
            if not top_teams.empty:
                st.dataframe(top_teams)
                st.bar_chart(top_teams.set_index("TEAM")["goals_scored"])
            else:
                st.warning("Не удалось загрузить данные")
        
        with col2:
            st.subheader("Топ команд по сезонам")
            top_season = fetch_eda_top_teams_season(10)
            if not top_season.empty:
                st.dataframe(top_season)
            else:
                st.warning("Не удалось загрузить данные")
        
        st.subheader("Среднее количество голов за матч по командам")
        team_avg_goals_df = fetch_team_avg_goals(limit=10)
        if not team_avg_goals_df.empty:
            st.dataframe(team_avg_goals_df)
            st.bar_chart(team_avg_goals_df.set_index("TEAM")["avg_goals"])
        else:
            st.warning("Не удалось загрузить данные")
    
    with eda_tabs[1]:
        st.subheader("Статистика игрока")
        
        player_id = st.text_input("Введите PLAYER_ID:", value="ply1479")
        
        if st.button("Получить статистику"):
            stats = fetch_player_stats(player_id)
            if stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Игрок", stats.get("full_name", ""))
                with col2:
                    st.metric("Голов", stats.get("goals_scored", 0))
                with col3:
                    avg_minute = stats.get("avg_minute", 0)
                    st.metric("Средняя минута", f"{avg_minute:.2f}" if avg_minute else "—")
            else:
                st.warning("Игрок не найден или ошибка сервера")
    
    with eda_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Общее количество голов")
            total_goals = fetch_total_goals()
            if total_goals:
                st.metric("Всего голов", total_goals["total_goals"])
        
        with col2:
            st.subheader("Средний возраст игроков, забивших пенальти")
            penalty_age = fetch_penalty_avg_age()
            if penalty_age:
                if penalty_age.get("avg_age") is not None:
                    st.metric("Средний возраст", f"{penalty_age['avg_age']} лет")
                else:
                    st.warning(penalty_age.get("message", "Нет данных"))

with tab3:
    st.title("Machine Learning")
    
    ml_tabs = st.tabs(["Модель", "Предсказание"])
    
    with ml_tabs[0]:
        st.subheader("Метрики модели KNN")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", "8300.60")
            st.metric("MSE", "177964266.50")
            st.metric("RMSE", "13340.32")
        with col2:
            st.metric("MAPE", "7417.35%")
            st.metric("R2", "0.70")
        
        st.subheader("Визуализация модели")
        try:
            img = Image.open("model_info.png")
            st.image(img, caption="Статистика модели KNN")
        except:
            st.info("Файл model_info.png не найден")
    
    with ml_tabs[1]:
        st.subheader("Предсказание посещаемости матча")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                season = st.text_input("Сезон", value="2020-2021")
                home_team = st.text_input("Домашняя команда", value="Borussia Dortmund")
            with col2:
                away_team = st.text_input("Гостевая команда", value="Manchester City")
                stadium = st.text_input("Стадион", value="Signal Iduna Park")
            
            submitted = st.form_submit_button("Предсказать")
            
            if submitted:
                result = predict_attendance(season, home_team, away_team, stadium)
                if result:
                    st.success(f"Ожидаемая посещаемость: {result['predicted_attendance']} человек")
                else:
                    st.error("Ошибка при получении предсказания")