import pandas as pd
from pycaret.regression import *

# df = pd.read_csv("ucl_matches_parsed.csv")

# df = df[["SEASON", "HOME_TEAM", "AWAY_TEAM", "STADIUM", "ATTENDANCE"]]

# for col in ["SEASON", "HOME_TEAM", "AWAY_TEAM", "STADIUM"]:
#     df[col] = df[col].astype(str)

# # print(df.isna().sum())

# exp = setup(
#     data=df,
#     target="ATTENDANCE",
#     categorical_features=["SEASON","HOME_TEAM","AWAY_TEAM","STADIUM"],
#     normalize=True,
# )

# best_model = compare_models()

# final_model = finalize_model(best_model)

# save_model(final_model, "attendance_automl_model")

model = load_model("attendance_automl_model")

new_match = pd.DataFrame([{
    "SEASON": "2021-2022",
    "HOME_TEAM": "Borussia Dortmund",
    "AWAY_TEAM": "Manchester City",
    "STADIUM": "Signal Iduna Park"
}])

pred = predict_model(model, data=new_match)
print(f"exp: {pred['prediction_label'][0]}")