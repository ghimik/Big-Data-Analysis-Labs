from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, concat_ws, max as spark_max, expr
from pyspark.sql.functions import year, current_date


DATA_PATH = "csv_output" 

def main():
    spark = SparkSession.builder \
        .appName("UEFA Champions League PySpark Analysis") \
        .getOrCreate()

    players = spark.read.option("header", True).csv(f"{DATA_PATH}/players.csv")
    teams = spark.read.option("header", True).csv(f"{DATA_PATH}/teams.csv")
    managers = spark.read.option("header", True).csv(f"{DATA_PATH}/managers.csv")
    matches = spark.read.option("header", True).csv(f"{DATA_PATH}/matches.csv")
    goals = spark.read.option("header", True).csv(f"{DATA_PATH}/goals.csv")

    players = players.withColumn("JERSEY_NUMBER", expr("try_cast(JERSEY_NUMBER as int)")) \
                     .withColumn("HEIGHT", expr("try_cast(HEIGHT as int)")) \
                     .withColumn("WEIGHT", expr("try_cast(WEIGHT as int)")) \
                     .withColumn("DOB", expr("try_cast(DOB as date)"))

    matches = matches.withColumn("HOME_TEAM_SCORE", expr("try_cast(HOME_TEAM_SCORE as int)")) \
                     .withColumn("AWAY_TEAM_SCORE", expr("try_cast(AWAY_TEAM_SCORE as int)")) \
                     .withColumn("ATTENDANCE", expr("try_cast(ATTENDANCE as int)")) \
                     .withColumn("PENALTY_SHOOT_OUT", expr("try_cast(PENALTY_SHOOT_OUT as int)")) \
                     .withColumn("DATE_TIME", expr("try_cast(DATE_TIME as timestamp)"))

    goals = goals.withColumn("DURATION", expr("try_cast(DURATION as int)"))

    print("Players Schema")
    players.printSchema()
    print("Matches Schema")
    matches.printSchema()
    print("Goals Schema")
    goals.printSchema()

    goals_per_team = players.join(goals, players.PLAYER_ID == goals.PID) \
                            .groupBy(players.TEAM) \
                            .agg(count(goals.GOAL_ID).alias("goals_scored")) \
                            .orderBy(col("goals_scored").desc())
    print("Топ команд по забитым голам")
    goals_per_team.show(10, truncate=False)

    goals_per_season_team = matches.join(goals, matches.MATCH_ID == goals.MATCH_ID) \
                                   .join(players, goals.PID == players.PLAYER_ID) \
                                   .groupBy(matches.SEASON, players.TEAM) \
                                   .agg(count(goals.GOAL_ID).alias("goals")) \
                                   .orderBy(col("goals").desc())
    print("Топ команд по сезонам по количеству голов")
    goals_per_season_team.show(10, truncate=False)

    player_stats = players.join(goals, players.PLAYER_ID == goals.PID, how="left") \
                          .filter(players.PLAYER_ID == "ply1479") \
                          .groupBy(concat_ws(" ", players.FIRST_NAME, players.LAST_NAME).alias("full_name")) \
                          .agg(
                              count(goals.GOAL_ID).alias("goals_scored"),
                              avg(goals.DURATION).alias("avg_minute")
                          )
    print("Cтатистика по игроку ply1479")
    player_stats.show(truncate=False)

    total_goals = goals.join(players, goals.PID == players.PLAYER_ID) \
                       .count()
    print("Количество голов по матчам и игрокам")
    print(total_goals)

    penalty_avg_age = players.join(goals, players.PLAYER_ID == goals.PID) \
                             .filter(goals.GOAL_DESC.contains("penalty")) \
                             .withColumn("age", year(current_date()) - year(players.DOB)) \
                             .agg(avg("age").alias("avg_age"))
    print("Средний возраст игроков, забивших пенальти")
    penalty_avg_age.show()

    max_weight = players.groupBy("POSITION") \
                        .agg(spark_max("WEIGHT").alias("max_weight")) \
                        .orderBy(col("max_weight").desc())
    print("Максимальный вес игроков по позициям")
    max_weight.show(truncate=True)

    avg_goals_per_team = matches.join(players, matches.HOME_TEAM == players.TEAM) \
                                .withColumn("match_goals", col("HOME_TEAM_SCORE") + col("AWAY_TEAM_SCORE")) \
                                .groupBy(players.TEAM) \
                                .agg(avg("match_goals").alias("avg_goals")) \
                                .orderBy(col("avg_goals").desc())
    print("Среднее количество голов за матч по командам")
    avg_goals_per_team.show(10, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
