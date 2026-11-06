import numpy as np
import pandas as pd

# read and set df
path= r'data\raw_shipments.csv'
df= pd.read_csv(path)

hubs_path = r'data\hubs.csv'
hubs = pd.read_csv(hubs_path)

def clean_shipments(raw, medians= None):
    clean_df = raw.copy()

    # 1. Drop exact duplicate rows
    clean_df.drop_duplicates(inplace= True)

    # 2. Clean service_type
    clean_df['service_type'] = (
        clean_df['service_type'].
        str.strip().
        str.title()
        )
    
    # 3. Replace missing payment_mode values
    clean_df['payment_mode'] = clean_df['payment_mode'].fillna('Unknown')
    
    # 4. Clean declared_value and convert it to numeric INR
    clean_df['declared_value_inr'] = (
        clean_df['declared_value'].
        str.strip().
        str.replace('₹','', regex= False).
        str.replace(',','', regex= False).
        astype('float')
        )
    clean_df.drop(columns=['declared_value'], inplace= True)

    # 5. Convert dispatch_time to datetime
    clean_df['dispatch_time'] = pd.to_datetime(clean_df['dispatch_time'], errors= 'coerce')

    # 6. Treat scanner fault
    clean_df.loc[clean_df['parcel_weight_kg']>100, 'parcel_weight_kg'] = np.nan

    # 7. assign missing median
    values = ['distance_km','parcel_weight_kg','declared_value_inr']
    if medians is None:
        medians = {c: float(clean_df[c].median()) for c in values}
    for c in values:
        clean_df[c] = clean_df[c].fillna(medians[c])

    return clean_df, medians

def build_features(clean, hubs):
    PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
    # assigning promised_hours
    clean['promised_hours'] = clean['service_type'].map(PROMISED_HOURS).astype(int)

    clean['is_cod'] = clean['payment_mode'].eq('COD').astype(int)
    clean['is_weekend'] = (clean['dispatch_time'].dt.dayofweek >= 5).astype(int)
    clean['dispatch_hour'] = (clean['dispatch_time'].dt.hour).astype(int)
    clean['log_declared_value'] = np.log(clean['declared_value_inr']) 
    # delivery_hours
    
    features = ['hub_id','promised_hours','is_cod','is_weekend','dispatch_hour','log_declared_value']
    mergedf = clean[features].merge(
        hubs[['hub_id','hub_city','region','hub_tier','hub_daily_capacity']], 
        on='hub_id', 
        how= 'left')

    return mergedf
