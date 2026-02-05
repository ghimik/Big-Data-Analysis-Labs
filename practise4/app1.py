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

st.sidebar.title("Меню")
page = st.sidebar.selectbox("Выберите страницу:", ["Главная", "EDA", "ML"])

if page == "Главная":
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

elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    
    st.subheader("Топ команд по забитым голам")
    top_teams = fetch_eda_top_teams(10)
    if not top_teams.empty:
        st.dataframe(top_teams)
        st.bar_chart(top_teams.set_index("TEAM")["goals_scored"])
    else:
        st.warning("Не удалось загрузить данные")
    
    st.subheader("Топ команд по сезонам")
    top_season = fetch_eda_top_teams_season(10)
    if not top_season.empty:
        st.dataframe(top_season)
    
    st.subheader("Статистика игрока")
    player_id = st.text_input("Введите PLAYER_ID:", value="ply1479")
    if st.button("Получить статистику"):
        stats = fetch_player_stats(player_id)
        if stats:
            st.json(stats)  
        else:
            st.warning("Игрок не найден или ошибка сервера")

    st.subheader("Общее количество голов")
    total_goals = fetch_total_goals()
    if total_goals:
        st.metric("Всего голов", total_goals["total_goals"])

    st.subheader("Средний возраст игроков, забивших пенальти")
    penalty_age = fetch_penalty_avg_age()

    if penalty_age:
        if penalty_age.get("avg_age") is not None:
            st.metric("Средний возраст", f"{penalty_age['avg_age']} лет")
        else:
            st.warning(penalty_age.get("message", "Нет данных"))


    st.subheader("Среднее количество голов за матч по командам")
    team_avg_goals_df = fetch_team_avg_goals(limit=10)

    if not team_avg_goals_df.empty:
        st.dataframe(team_avg_goals_df)
        st.bar_chart(
            team_avg_goals_df.set_index("TEAM")["avg_goals"]
        )
    else:
        st.warning("Не удалось загрузить данные")

    


elif page == "ML":

    st.title("Machine Learning")

    st.subheader("Метрики модели")
    st.write("""
        - MAE: 8300.6039
        - MSE: 177964266.4950
        - RMSE: 13340.3248
        - MAPE: 7417.345%
        - R2: 0.6979
    """)

    st.subheader("Визуализация модели")
    try:
        img = Image.open("model_info.png")
        st.image(img, caption="Статистика модели KNN")
    except:
        st.warning("Файл не найден")

    st.subheader("Предсказание посещаемости")
    season = st.text_input("Сезон", value="2020-2021")
    home_team = st.text_input("Домашняя команда", value="Borussia Dortmund")
    away_team = st.text_input("Гостевая команда", value="Manchester City")
    stadium = st.text_input("Стадион", value="Signal Iduna Park")

    if st.button("Предсказать"):
        result = predict_attendance(season, home_team, away_team, stadium)
        if result:
            st.success(f"Ожидаемая посещаемость: {result['predicted_attendance']} человек")


    st.subheader("AutoML с PyCaret")

    st.write("Вы можете обучить модель PyCaret или сделать предсказание с существующей AutoML-моделью.")

    with st.expander("Обучить новую AutoML модель"):
        train_model_name = st.text_input("Имя для модели (оставьте пустым для авто):", value="")
        if st.button("Обучить AutoML"):
            payload = {
                "target": "ATTENDANCE",
                "categorical_features": ["SEASON","HOME_TEAM","AWAY_TEAM","STADIUM"],
                "normalize": True
            }
            response = requests.post(f"{API_URL}/train_automl", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.success(f"Модель обучена и сохранена под именем: {data['model_name']}")
                st.write("Метрики моделей:")
                st.dataframe(pd.DataFrame(data['metrics']))
                st.write("Логи обучения:")
                st.dataframe(pd.DataFrame(data['logs']))
            else:
                st.error(f"Ошибка обучения: {response.text}")

    with st.expander("Предсказать посещаемость с AutoML моделью"):
        auto_model_name = st.text_input("Имя модели для предсказания:", value="attendance_automl_model")
        if st.button("Предсказать с AutoML"):
            payload = {
                "SEASON": season,
                "HOME_TEAM": home_team,
                "AWAY_TEAM": away_team,
                "STADIUM": stadium,
                "model_name": auto_model_name
            }
            response = requests.post(f"{API_URL}/predict_automl", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Ожидаемая посещаемость: {result['predicted_attendance']} человек")
            else:
                st.error(f"Ошибка предсказания: {response.text}")
