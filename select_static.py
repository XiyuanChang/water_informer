import pandas as pd

columns = ["STAID", "NITR_APP_KG_SQKM", "DRAIN_SQKM", "PHOS_APP_KG_SQKM", "RAW_DIS_NEAREST_DAM", "ELEV_MEAN_M_BASIN"]

df = pd.read_csv("climate_new/static_filtered.csv")
new_df = df[columns]
new_df.to_csv("climate_new/static_filtered_5.csv", index=False)