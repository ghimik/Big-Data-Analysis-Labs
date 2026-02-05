import psycopg2
import pandas as pd
from pathlib import Path
from datetime import datetime

CSV_DIR = Path("csv_output")
TABLE_ORDER = ["stadiums", "teams", "players", "managers", "matches", "goals"]

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "football_db",
    "user": "admin",
    "password": "admin"
}

def parse_value(v):
    """Обрабатываем значения, чтобы None и даты корректно вставлялись"""
    if pd.isna(v):
        return None
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, datetime):
        return v
    return v

def insert_csv_to_db(conn, table_name, csv_file):
    df = pd.read_csv(csv_file)
    df = df.applymap(parse_value)
    columns = list(df.columns)
    
    placeholders = ", ".join(["%s"] * len(columns))
    cols = ", ".join(columns)
    query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"

    cur = conn.cursor()
    for row in df.itertuples(index=False, name=None):
        cur.execute(query, row)
    conn.commit()
    cur.close()
    print(f"[OK] Inserted {len(df)} rows into {table_name}")

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    for table in TABLE_ORDER:
        csv_file = CSV_DIR / f"{table}.csv"
        if csv_file.exists():
            insert_csv_to_db(conn, table, csv_file)
        else:
            print(f"[WARN] CSV not found: {csv_file}")
    conn.close()
    print("[DONE] All CSV files imported.")

if __name__ == "__main__":
    main()
