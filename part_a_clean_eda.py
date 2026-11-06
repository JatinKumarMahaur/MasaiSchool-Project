import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

# read and set file
raw = pd.read_csv("data/raw_shipments.csv")
raw_dedup= raw.drop_duplicates()
hubs = pd.read_csv("data/hubs.csv")

# calling the functions
clean, medians= clean_shipments(raw)
table= build_features(clean, hubs)

# add target
table['is_delayed'] = (clean['delivery_hours'] > table['promised_hours']).astype(int)

# Find common column, if any and merging
common_cols = clean.columns.intersection(table.columns).tolist()
modelling_table = clean.merge(table, on=common_cols)

# write the result
modelling_table.to_csv("data/modelling_table.csv", index=False)

weight_before= raw_dedup['parcel_weight_kg']

h = modelling_table["delivery_hours"]
q1, q3 = h.quantile(0.25), h.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
dv = modelling_table["declared_value_inr"]
ldv = modelling_table["log_declared_value"]

# Checks file
metrics = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(raw_dedup)),
    ("rows_after_dedup", len(raw_dedup)),
    ("missing_distance_km_before", int(raw_dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(weight_before.isna().sum())),
    ("sentinel_weight_rows", int((weight_before > 100).sum())),
    ("missing_declared_value_before", int(raw_dedup['declared_value'].isnull().sum())),
    ("missing_payment_mode_before", int(raw_dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", len(raw_dedup['service_type'].unique())),
    ("service_type_spellings_after", len(modelling_table['service_type'].unique())),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(modelling_table.isna().sum().sum())),
    ("delayed_count", int(table["is_delayed"].sum())),
    ("delay_rate", table["is_delayed"].mean()),
    ("hours_q1", q1), ("hours_q3", q3),
    ("hours_lower_bound", lo), ("hours_upper_bound", hi),
    ("hours_outlier_count", int(((h < lo) | (h > hi)).sum())),
    ("declared_value_mean_inr", dv.mean()),
    ("declared_value_median_inr", dv.median()),
    ("log_declared_value_mean", ldv.mean()),
    ("log_declared_value_median", ldv.median()),
]

checks = pd.DataFrame(metrics, columns=["metric", "value"])
checks["value"] = checks["value"].astype(float).round(4)
checks.to_csv("outputs/part_a_checks.csv", index=False)

# delay by group
modelling_table["distance_band"]= pd.cut(modelling_table['distance_km'], 
                               bins=[0,200,400,600,2000], 
                               labels=["0-200", "201-400", "401-600", "601+"], 
                               right= True)

rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = modelling_table.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"), 
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"), 
        mean_hours=("delivery_hours", "mean"))
    
    for grp, r in g.iterrows():
        rows.append([dim, str(grp), int(r["shipments"]), int(r["delayed"]), r["delay_rate"], r["mean_hours"]])

delay_col = ['dimension', 'group', 'shipments', 'delayed', 'delay_rate', 'mean_hours']
part_a_delay_by_group = pd.DataFrame(rows, columns=delay_col)
part_a_delay_by_group[["delay_rate", "mean_hours"]] = part_a_delay_by_group[["delay_rate", "mean_hours"]].round(4)
part_a_delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlations of numeric columns
corr = modelling_table.drop(columns=["distance_band"]).select_dtypes(include=[np.number]).corr().round(5)
corr.to_csv("outputs/part_a_correlations.csv")