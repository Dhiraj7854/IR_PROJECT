"""
preprocess.py
Data preprocessing module for Solar Power Generation and Weather Sensor data.
Handles timestamp parsing, inverter aggregation, missing data imputation,
cyclic feature engineering, chronological train/val/test splitting, and leak-free scaling.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_clean_plant_data(gen_path, weather_path, is_plant1=True):
    """
    Loads generation and weather data, parses timestamps, aggregates inverters,
    and merges onto a continuous 15-minute grid.
    """
    df_gen = pd.read_csv(gen_path)
    df_weather = pd.read_csv(weather_path)

    # Date parsing
    if is_plant1:
        df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], format='%d-%m-%Y %H:%M')
    else:
        df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'])

    df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'])

    # Aggregate generation across inverters
    agg_gen = df_gen.groupby('DATE_TIME').agg({
        'AC_POWER': 'sum',
        'DC_POWER': 'sum',
        'DAILY_YIELD': 'sum',
        'TOTAL_YIELD': 'sum'
    }).reset_index()

    # Aggregate weather (typically 1 sensor, but groupby ensures unique timestamps)
    agg_weather = df_weather.groupby('DATE_TIME').agg({
        'AMBIENT_TEMPERATURE': 'mean',
        'MODULE_TEMPERATURE': 'mean',
        'IRRADIATION': 'mean'
    }).reset_index()

    # Create continuous 15-minute grid
    start_time = max(agg_gen['DATE_TIME'].min(), agg_weather['DATE_TIME'].min())
    end_time = min(agg_gen['DATE_TIME'].max(), agg_weather['DATE_TIME'].max())
    full_idx = pd.date_range(start=start_time, end=end_time, freq='15min')
    grid_df = pd.DataFrame({'DATE_TIME': full_idx})

    # Merge
    merged = pd.merge(grid_df, agg_gen, on='DATE_TIME', how='left')
    merged = pd.merge(merged, agg_weather, on='DATE_TIME', how='left')

    # Impute missing values with physical domain rules
    # Nighttime (hour < 5 or hour >= 19) solar power and irradiation are 0
    night_mask = (merged['DATE_TIME'].dt.hour < 5) | (merged['DATE_TIME'].dt.hour >= 19)
    merged.loc[night_mask, 'AC_POWER'] = merged.loc[night_mask, 'AC_POWER'].fillna(0.0)
    merged.loc[night_mask, 'DC_POWER'] = merged.loc[night_mask, 'DC_POWER'].fillna(0.0)
    merged.loc[night_mask, 'IRRADIATION'] = merged.loc[night_mask, 'IRRADIATION'].fillna(0.0)

    # Linear interpolation for short missing daytime gaps, then forward/backward fill
    merged['AC_POWER'] = merged['AC_POWER'].interpolate(method='linear').clip(lower=0.0).bfill().ffill()
    merged['DC_POWER'] = merged['DC_POWER'].interpolate(method='linear').clip(lower=0.0).bfill().ffill()
    merged['IRRADIATION'] = merged['IRRADIATION'].interpolate(method='linear').clip(lower=0.0).bfill().ffill()
    merged['AMBIENT_TEMPERATURE'] = merged['AMBIENT_TEMPERATURE'].interpolate(method='linear').bfill().ffill()
    merged['MODULE_TEMPERATURE'] = merged['MODULE_TEMPERATURE'].interpolate(method='linear').bfill().ffill()

    # Feature engineering: Cyclic time-of-day and day-of-week
    time_of_day = merged['DATE_TIME'].dt.hour + merged['DATE_TIME'].dt.minute / 60.0
    merged['sin_tod'] = np.sin(2 * np.pi * time_of_day / 24.0)
    merged['cos_tod'] = np.cos(2 * np.pi * time_of_day / 24.0)
    
    day_of_week = merged['DATE_TIME'].dt.dayofweek
    merged['sin_dow'] = np.sin(2 * np.pi * day_of_week / 7.0)
    merged['cos_dow'] = np.cos(2 * np.pi * day_of_week / 7.0)

    return merged


def create_sliding_windows(df, feature_cols, target_col='AC_POWER', history_len=192, horizon_len=96, step=1):
    """
    Builds sliding window tensors:
    X: shape (num_samples, history_len, num_features)
    y: shape (num_samples, horizon_len)
    timestamps_y: timestamp range corresponding to target horizon
    """
    features = df[feature_cols].values
    target = df[target_col].values
    dates = df['DATE_TIME'].values

    X_list = []
    y_list = []
    target_dates = []

    total_len = len(df)
    for i in range(0, total_len - history_len - horizon_len + 1, step):
        x_window = features[i : i + history_len]
        y_window = target[i + history_len : i + history_len + horizon_len]
        dates_window = dates[i + history_len : i + history_len + horizon_len]
        
        X_list.append(x_window)
        y_list.append(y_window)
        target_dates.append(dates_window)

    return np.array(X_list), np.array(y_list), np.array(target_dates)


def make_future_covariates(target_dates, latitude, longitude, timezone='Asia/Kolkata'):
    """Return known time-of-day and pvlib clear-sky features for each forecast step."""
    import pvlib

    dates = pd.DatetimeIndex(np.asarray(target_dates).reshape(-1))
    if dates.tz is None:
        dates = dates.tz_localize(timezone)
    else:
        dates = dates.tz_convert(timezone)

    location = pvlib.location.Location(latitude, longitude, tz=timezone)
    clear_sky_ghi = location.get_clearsky(dates, model='ineichen')['ghi'].to_numpy()
    fractional_hour = dates.hour + dates.minute / 60.0
    features = np.column_stack((
        np.sin(2 * np.pi * fractional_hour / 24.0),
        np.cos(2 * np.pi * fractional_hour / 24.0),
        clear_sky_ghi / 1000.0,
    ))
    return features.reshape(len(target_dates), target_dates.shape[1], 3).astype(np.float32)


def prepare_chronological_splits(df, feature_cols, target_col='AC_POWER', history_len=192, horizon_len=96,
                                  train_ratio=0.70, val_ratio=0.15):
    """
    Splits continuous time series chronologically to prevent future information leakage:
    1. Splits dataframe into train, val, and test partitions.
    2. Fits StandardScaler on training features and target ONLY.
    3. Transforms val and test using the fitted scalers.
    4. Creates sliding window tensors for each partition.
    """
    total_len = len(df)
    timestamps = pd.DatetimeIndex(df['DATE_TIME'])
    midnight_indices = np.flatnonzero((timestamps.hour == 0) & (timestamps.minute == 0))

    def next_midnight(index):
        candidates = midnight_indices[midnight_indices >= index]
        return int(candidates[0]) if len(candidates) else total_len

    train_end = next_midnight(int(total_len * train_ratio))
    val_end = next_midnight(int(total_len * (train_ratio + val_ratio)))
    if val_end <= train_end:
        raise ValueError("Chronological split ratios must leave at least one validation day")

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    # Include history_len buffer from preceding split so test/val can predict their respective starting points
    val_start = train_end - history_len
    val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_start = val_end - history_len
    test_df = df.iloc[test_start:].copy().reset_index(drop=True)

    # Fit scalers on train_df ONLY
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_cols].values)

    target_scaler = StandardScaler()
    target_scaler.fit(train_df[[target_col]].values)

    def scale_and_window(partition_df):
        scaled_features = feature_scaler.transform(partition_df[feature_cols].values)
        scaled_target = target_scaler.transform(partition_df[[target_col]].values).flatten()

        part_copy = partition_df.copy()
        part_copy[feature_cols] = scaled_features
        part_copy[target_col] = scaled_target

        X, y, dates = create_sliding_windows(
            part_copy, feature_cols, target_col=target_col,
            history_len=history_len, horizon_len=horizon_len
        )
        return X, y, dates

    X_train, y_train, dates_train = scale_and_window(train_df)
    X_val, y_val, dates_val = scale_and_window(val_df)
    X_test, y_test, dates_test = scale_and_window(test_df)

    scaler_dict = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'target_mean': target_scaler.mean_[0],
        'target_std': target_scaler.scale_[0],
        'rated_capacity': df[target_col].max()
    }

    return {
        'train': (X_train, y_train, dates_train),
        'val': (X_val, y_val, dates_val),
        'test': (X_test, y_test, dates_test),
        'scalers': scaler_dict,
        'raw_df': df,
        'feature_cols': feature_cols
    }


if __name__ == '__main__':
    print("Testing preprocess module...")
    df_p1 = load_and_clean_plant_data(
        'data/raw/Plant_1_Generation_Data.csv',
        'data/raw/Plant_1_Weather_Sensor_Data.csv',
        is_plant1=True,
    )
    print(f"Loaded Plant 1: {len(df_p1)} intervals ({df_p1['DATE_TIME'].min()} to {df_p1['DATE_TIME'].max()})")
    
    feature_cols = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
                    'sin_tod', 'cos_tod', 'sin_dow', 'cos_dow']
    data_dict = prepare_chronological_splits(df_p1, feature_cols)
    print("Train X shape:", data_dict['train'][0].shape, "y shape:", data_dict['train'][1].shape)
    print("Val X shape:", data_dict['val'][0].shape, "y shape:", data_dict['val'][1].shape)
    print("Test X shape:", data_dict['test'][0].shape, "y shape:", data_dict['test'][1].shape)
    print("Preprocessor test passed!")
