import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


HOLIDAYS_DEFAULT = ['2024-12-25', '2025-01-01', '2025-09-01']
BASE_TEMP_DEFAULT = 65


@dataclass
class AnalysisConfig:
    facility_name: str = "Unknown Facility"
    timezone: str | None = None              # e.g. "America/Toronto"
    holidays: list[str] | None = None        # override default holidays
    base_temp: float = BASE_TEMP_DEFAULT
    run_ml_anomalies: bool = True            # toggle LSTM block
    recent_window_days: int = 30             # "last N days" window


def _detect_meter_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Return (kwh_col, kw_col) from a raw interval DataFrame.

    Priority:
      1) 'Aggregate - kWh' / 'Aggregate - kW'
      2) Any '<something> kWh - kWh' and '<something> kWh - kW' pair.
    """
    orig_cols = list(df.columns)
    norm_map = {c.lower().strip(): c for c in orig_cols}

    # 1) Aggregate pair
    agg_kwh_key = 'aggregate - kwh'
    agg_kw_key = 'aggregate - kw'
    if agg_kwh_key in norm_map and agg_kw_key in norm_map:
        return norm_map[agg_kwh_key], norm_map[agg_kw_key]

    # 2) Numbered meter columns
    kwh_candidates = []
    kw_candidates = []
    for c in orig_cols:
        cl = c.lower()
        if 'kwh - kwh' in cl:
            kwh_candidates.append(c)
        if 'kwh - kw' in cl:
            kw_candidates.append(c)

    if kwh_candidates and kw_candidates:
        return kwh_candidates[0], kw_candidates[0]

    raise ValueError("Could not automatically detect kWh/kW columns in the interval file.")


def _build_anomaly_model(features_scaled: np.ndarray, timesteps: int = 24):
    """
    Train LSTM on scaled features (first column = kWh) and
    return (anomaly_indexes, y_unscaled, residuals, scaler).

    features_scaled shape: (N, num_features)
    """
    scaler = StandardScaler()
    # Fit only on the first column (kWh) for inverse transform later
    kwh_scaled = features_scaled[:, 0].reshape(-1, 1)
    scaler.fit(kwh_scaled)

    X, y = [], []
    for i in range(timesteps, len(features_scaled)):
        X.append(features_scaled[i - timesteps:i])
        y.append(features_scaled[i][0])  # kWh feature
    X, y = np.array(X), np.array(y)

    if len(X) <= 200:
        return np.array([], dtype=int), np.array([]), np.array([]), scaler

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(timesteps, X.shape[2])),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(X, y, epochs=10, batch_size=64, validation_split=0.1, verbose=0)

    preds = model.predict(X, verbose=0).reshape(-1, 1)
    preds_unscaled = scaler.inverse_transform(preds)[:, 0]
    y_unscaled = scaler.inverse_transform(y.reshape(-1, 1))[:, 0]
    residuals = y_unscaled - preds_unscaled
    threshold = np.std(residuals) * 2
    anomaly_indexes = np.where(np.abs(residuals) > threshold)[0]

    return anomaly_indexes, y_unscaled, residuals, scaler


def run_analysis(df_raw: pd.DataFrame, config: AnalysisConfig | None = None):
    if config is None:
        config = AnalysisConfig()

    holidays = config.holidays or HOLIDAYS_DEFAULT
    base_temp = config.base_temp
    recent_days = config.recent_window_days

    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # ---- AUTO-DETECT METER COLUMNS ----
    kwh_col, kw_col = _detect_meter_columns(df)
    # -----------------------------------

    # Time handling
    df['Interval Start'] = pd.to_datetime(
        df['Interval Start Date'] + ' ' + df['Interval Start Time']
    )
    if config.timezone:
        df['Interval Start'] = df['Interval Start'].dt.tz_localize(
            config.timezone,
            nonexistent='shift_forward',
            ambiguous='NaT'
        )

    df['Consumption - kWh'] = pd.to_numeric(df[kwh_col], errors='coerce')
    df['Metered kW'] = pd.to_numeric(df[kw_col], errors='coerce')
    df['Temp_F'] = pd.to_numeric(df['Temp (F) Mean'], errors='coerce')

    # Data quality checks
    df = df.dropna(subset=['Consumption - kWh', 'Metered kW', 'Temp_F'])
    df = df.sort_values('Interval Start')
    interval_seconds_series = df['Interval Start'].diff().dt.total_seconds().dropna()
    interval_seconds_mode = interval_seconds_series.mode()[0] if not interval_seconds_series.empty else np.nan

    dq_flags = {
        "expected_interval_seconds": interval_seconds_mode,
        "missing_intervals_count": int(
            (df['Interval Start'].max() - df['Interval Start'].min()).total_seconds()
            / interval_seconds_mode + 1 - len(df)
        ) if interval_seconds_mode and not np.isnan(interval_seconds_mode) else np.nan,
        "duplicate_timestamps": int(df['Interval Start'].duplicated().sum()),
    }

    df['Day'] = df['Interval Start'].dt.date
    df['Hour'] = df['Interval Start'].dt.hour
    df['DayOfWeek'] = df['Interval Start'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'] >= 5
    holiday_days = [datetime.strptime(d, '%Y-%m-%d').date() for d in holidays]
    df['Is_Holiday'] = df['Day'].isin(holiday_days)

    df['CDD'] = (df['Temp_F'] - base_temp).clip(lower=0)
    df['HDD'] = (base_temp - df['Temp_F']).clip(lower=0)

    group_cols = ['Day', 'Is_Weekend', 'Is_Holiday']
    daily = df.groupby(group_cols)[['Consumption - kWh', 'CDD', 'HDD']].sum().reset_index()
    daily['Rolling7_avg_kWh'] = daily['Consumption - kWh'].rolling(7).mean()
    daily['Rolling30_avg_kWh'] = daily['Consumption - kWh'].rolling(30).mean()

    last_days = daily.tail(recent_days).copy()
    if len(last_days) >= 5:
        fit = np.polyfit(last_days['CDD'], last_days['Consumption - kWh'], 1)
        daily['CDD_fit'] = fit[0] * daily['CDD'] + fit[1]
        fit2 = np.polyfit(last_days['HDD'], last_days['Consumption - kWh'], 1)
        daily['HDD_fit'] = fit2[0] * daily['HDD'] + fit2[1]
        cdd_corr = last_days['Consumption - kWh'].corr(last_days['CDD'])
        hdd_corr = last_days['Consumption - kWh'].corr(last_days['HDD'])
    else:
        fit = fit2 = [np.nan, np.nan]
        daily['CDD_fit'] = np.nan
        daily['HDD_fit'] = np.nan
        cdd_corr = hdd_corr = np.nan

    resid = daily['Consumption - kWh'] - daily['CDD_fit']
    bad_days = daily[
        resid.notna() &
        (daily['CDD'] > 0) &
        (resid > resid.mean() + 2 * resid.std())
    ][['Day', 'Consumption - kWh', 'CDD', 'CDD_fit']]

    mask_last_recent = daily['Day'] >= (daily['Day'].max() - pd.Timedelta(days=recent_days))
    seg_weekday = daily.loc[
        mask_last_recent & ~daily['Is_Weekend'] & ~daily['Is_Holiday'],
        'Consumption - kWh'
    ].mean()
    seg_weekend = daily.loc[
        mask_last_recent & daily['Is_Weekend'],
        'Consumption - kWh'
    ].mean()
    seg_holiday = daily.loc[
        mask_last_recent & daily['Is_Holiday'],
        'Consumption - kWh'
    ].mean()

    top5_peaks = df.nlargest(5, 'Consumption - kWh')[['Interval Start', 'Consumption - kWh']]
    lowest_base = df.nsmallest(15, 'Consumption - kWh')[['Interval Start', 'Consumption - kWh']].iloc[5:15]

    interval_hours = interval_seconds_mode / 3600 if interval_seconds_mode and not np.isnan(interval_seconds_mode) else np.nan
    peak_kw = df['Metered kW'].max()
    total_hrs = len(df) * interval_hours if interval_hours and not np.isnan(interval_hours) else np.nan
    avg_kw = df['Consumption - kWh'].sum() / total_hrs if total_hrs and not np.isnan(total_hrs) else np.nan
    load_factor = avg_kw / peak_kw if peak_kw and peak_kw > 0 else np.nan

    overnight_base = df[df['Hour'].isin(range(0, 7))]
    overnight_avg = overnight_base.groupby('Day')['Consumption - kWh'].mean().tail(5)

    # ML anomalies (optional)
    anomaly_df = pd.DataFrame()
    if config.run_ml_anomalies:
        features = df[['Consumption - kWh', 'Temp_F', 'Hour', 'DayOfWeek']]
        features_scaled = StandardScaler().fit_transform(features)

        timesteps = 24
        anomaly_indexes, y_unscaled, residuals, _ = _build_anomaly_model(
            features_scaled, timesteps=timesteps
        )

        if anomaly_indexes.size > 0:
            anomaly_times = df['Interval Start'].iloc[anomaly_indexes + timesteps]
            anomaly_vals = y_unscaled[anomaly_indexes]
            anomaly_df = pd.DataFrame({
                'Time': anomaly_times,
                'Actual_kWh': anomaly_vals,
                'Deviation': np.abs(residuals[anomaly_indexes]),
            })
            if not anomaly_df.empty:
                recent_cutoff = anomaly_df['Time'].max() - pd.Timedelta(days=recent_days)
                anomaly_df = anomaly_df[anomaly_df['Time'] > recent_cutoff].nlargest(10, 'Deviation')

    last_recent = df[df['Interval Start'] >= (df['Interval Start'].max() - pd.Timedelta(days=recent_days))].copy()
    day_hours = range(6, 18)
    night_hours = [h for h in range(24) if h not in day_hours]
    day_avg = last_recent[last_recent['Hour'].isin(day_hours)]['Consumption - kWh'].mean()
    night_avg = last_recent[last_recent['Hour'].isin(night_hours)]['Consumption - kWh'].mean()
    day_night_ratio = day_avg / night_avg if night_avg and night_avg > 0 else np.nan

    baseload = last_recent['Consumption - kWh'].min()
    baseload_kWh = baseload * len(last_recent)
    baseload_pct = baseload_kWh / last_recent['Consumption - kWh'].sum() if last_recent['Consumption - kWh'].sum() > 0 else np.nan
    peak_cons = last_recent['Consumption - kWh'].max()
    baseload_gap = (peak_cons - baseload) / peak_cons if peak_cons and peak_cons > 0 else np.nan
    after_hours = last_recent[last_recent['Hour'].isin(night_hours)]['Consumption - kWh'].sum()
    after_hours_pct = after_hours / last_recent['Consumption - kWh'].sum() if last_recent['Consumption - kWh'].sum() > 0 else np.nan
    coefficient_var = last_recent['Consumption - kWh'].std() / last_recent['Consumption - kWh'].mean() if last_recent['Consumption - kWh'].mean() > 0 else np.nan

    holiday_days = [datetime.strptime(h, '%Y-%m-%d').date() for h in holidays]
    holiday_data = df[df['Day'].isin(holiday_days)]
    if not holiday_data.empty:
        holiday_base = holiday_data.groupby('Day')['Consumption - kWh'].mean()
    else:
        holiday_base = pd.Series(dtype=float)

    kpis = {
        'facility_name': config.facility_name,
        'load_factor': load_factor,
        'day_night_ratio': day_night_ratio,
        'baseload_pct': baseload_pct,
        'baseload_gap': baseload_gap,
        'after_hours_pct': after_hours_pct,
        'coeff_var': coefficient_var,
        'weekday_kWh_last30': seg_weekday,
        'weekend_kWh_last30': seg_weekend,
        'holiday_kWh_last30': seg_holiday,
        'cdd_corr_last30': cdd_corr,
        'hdd_corr_last30': hdd_corr,
        'cdd_slope_last30': fit[0],
        'hdd_slope_last30': fit2[0],
        'recent_window_days': recent_days,
    }

    heatmap_data = last_recent.pivot_table(
        index=last_recent['Interval Start'].dt.date,
        columns='Hour',
        values='Consumption - kWh',
        aggfunc='mean'
    )
    fig_heatmap, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=.1, ax=ax)
    ax.set_title(f'Electricity Consumption Heatmap — Last {recent_days} Days')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Date')
    plt.tight_layout()

    return {
        'kpis': kpis,
        'daily': daily,
        'heatmap_fig': heatmap_data,
        'top5_peaks': top5_peaks,
        'lowest_base': lowest_base,
        'overnight_avg': overnight_avg,
        'holiday_base': holiday_base,
        'anomalies': anomaly_df,
        'bad_days': bad_days,
        'data_quality': dq_flags,
    }
