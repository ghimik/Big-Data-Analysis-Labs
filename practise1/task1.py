import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from config import db_config 


def run_queries(engine):
    with engine.connect() as conn:
        # join 2 таблицы: игроки и голы, агрегация по команде
        print("Топ команд по забитым голам:")
        query = text("""
            SELECT p."TEAM" AS team_name, COUNT(g."GOAL_ID") AS goals_scored
            FROM "players" p
            JOIN "goals" g ON p."PLAYER_ID" = g."PID"
            GROUP BY p."TEAM"
            ORDER BY goals_scored DESC
            LIMIT 10;
        """)
        result = conn.execute(query)
        for row in result:
            print(row)

        # join 3 таблицы: матчи, команды, голы
        print("\nТоп команд по сезонам по количеству голов:")
        query = text("""
            SELECT m."SEASON", p."TEAM" AS team_name, COUNT(g."GOAL_ID") AS goals
            FROM "matches" m
            JOIN "goals" g ON m."MATCH_ID" = g."MATCH_ID"
            JOIN "players" p ON g."PID" = p."PLAYER_ID"
            GROUP BY m."SEASON", p."TEAM"
            ORDER BY goals DESC
            LIMIT 10;
        """)
        result = conn.execute(query)
        for row in result:
            print(row)

        # выборка по конкретному игроку
        print("\nСтатистика по игроку ply1479:")
        query = text("""
            SELECT p."FIRST_NAME" || ' ' || p."LAST_NAME" AS full_name,
                   COUNT(g."GOAL_ID") AS goals_scored,
                   AVG(g."DURATION") AS avg_minute
            FROM "players" p
            LEFT JOIN "goals" g ON p."PLAYER_ID" = g."PID"
            WHERE p."PLAYER_ID" = 'ply1479'
            GROUP BY full_name;
        """)
        result = conn.execute(query)
        for row in result:
            print(row)

        # количество строк по совмещенным данным двух таблиц
        print("\nКоличество голов по матчам и игрокам:")
        query = text("""
            SELECT COUNT(*) 
            FROM "goals" g
            JOIN "players" p ON g."PID" = p."PLAYER_ID";
        """)
        result = conn.execute(query)
        print(result.scalar())

        # дополнительные сложные SELECT запросы
        print("\nСредний возраст игроков, забивших пенальти:")
        query = text("""
            SELECT AVG(EXTRACT(YEAR FROM AGE(CURRENT_DATE, p."DOB"))) AS avg_age
            FROM "players" p
            JOIN "goals" g ON p."PLAYER_ID" = g."PID"
            WHERE g."GOAL_DESC" ILIKE '%penalty%';
        """)
        result = conn.execute(query)
        print(result.scalar())

        print("\nМаксимальный вес игроков по позициям:")
        query = text("""
            SELECT "POSITION", MAX("WEIGHT") AS max_weight
            FROM "players"
            GROUP BY "POSITION"
            ORDER BY max_weight DESC;
        """)
        result = conn.execute(query)
        for row in result:
            print(row)

        print("\nСреднее количество голов за матч по командам:")
        query = text("""
            SELECT p."TEAM", AVG(m."HOME_TEAM_SCORE" + m."AWAY_TEAM_SCORE") AS avg_goals
            FROM "matches" m
            JOIN "players" p ON m."HOME_TEAM" = p."TEAM"
            GROUP BY p."TEAM"
            ORDER BY avg_goals DESC
            LIMIT 10;
        """)
        result = conn.execute(query)
        for row in result:
            print(row)


def main():
    engine = create_engine(db_config.alembic_url)
    run_queries(engine)

if __name__ == "__main__":
    main()
