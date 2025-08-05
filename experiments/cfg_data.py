import pandas as pd

import pandas as pd

df = pd.read_csv("cum_cfg.csv")
df.columns = ["step", "neg_score", "x", "y"]

df["cum_sum"] = df["neg_score"] * (df["step"] + 1)
df["actual"] = df["cum_sum"].diff().fillna(df["cum_sum"])
df["actual"] = (df["actual"] > 0.5)
df.to_csv("cum_cfg_actual.csv", index=False) 