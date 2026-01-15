from pathlib import Path
import pandas as pd
import math
from datetime import datetime

EXCEL_PATH = Path("UEFA Champions League 2016-2022 Data 3.xlsx")
ALEMBIC_VERSIONS = Path("migrations/versions")

TABLE_ORDER = [
    "stadiums",
    "teams",
    "players",
    "managers",
    "matches",
    "goals",
]

REVISION = datetime.now().strftime("%Y%m%d_%H%M%S")
FILENAME = f"{REVISION}_initial_data.py"


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
    
    # КТО ТАК ВРЕМЯ ЗАПИСЫВАЕТ ИДИОТЫ
    if col_name and 'DATE' in col_name.upper() or 'DOB' in col_name.upper():
        if isinstance(v, str):
            try:
                return pd.to_datetime(v, format='%d-%b-%y %I.%M.%S.%f %p', errors='coerce')
            except:
                try:
                    return pd.to_datetime(v, errors='coerce')
                except:
                    return None
    return v


def sheet_to_rows(sheet_name: str):
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)

    rows = []
    for _, row in df.iterrows():
        clean = {k: normalize_value(v, k) for k, v in row.items()}
        rows.append(clean)

    return rows


def main():
    ALEMBIC_VERSIONS.mkdir(exist_ok=True)
    out = ALEMBIC_VERSIONS / FILENAME

    with open(out, "w", encoding="utf-8") as f:
        f.write(
f'''from alembic import op
import sqlalchemy as sa

revision = "{REVISION}"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
'''
        )

        for table in TABLE_ORDER:
            rows = sheet_to_rows(table)
            if not rows:
                continue

            cols = rows[0].keys()

            f.write(f"\n    op.bulk_insert(\n")
            f.write(f"        sa.table('{table}',\n")

            for c in cols:
                f.write(f"            sa.column('{c}'),\n")

            f.write("        ),\n")
            f.write("        [\n")

            for r in rows:
                for k, v in r.items():
                    if isinstance(v, datetime):
                        r[k] = v.isoformat()
                f.write(f"            {r},\n")

            f.write("        ]\n")
            f.write("    )\n")

        f.write(
'''
def downgrade():
    pass
'''
        )

    print(f"[OK] migration created: {out}")


if __name__ == "__main__":
    main()
