import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import run_analysis

st.set_page_config(
    page_title="GreenedFM Interval Dashboard",
    layout="wide"
)

st.title("GreenedFM – Electricity Interval Heatmaps & Insights")

uploaded = st.file_uploader("Upload interval CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV exported from the meter to see the dashboard.")
else:
    with st.spinner("Running analysis..."):
        df_raw = pd.read_csv(uploaded, header=1)
        results = run_analysis(df_raw)

    kpis = results["kpis"]
    daily = results["daily"]
    heatmap_df = results["heatmap_fig"]  # pivot table
    anomalies = results["anomalies"]
    bad_days = results["bad_days"]

    # ------------------------------------------------------------------
    # KPI row
    # ------------------------------------------------------------------
    st.subheader("Key metrics (last 30 days)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Load factor",
        f"{kpis['load_factor']:.2f}" if pd.notna(kpis['load_factor']) else "n/a"
    )
    c2.metric(
        "Day/Night ratio",
        f"{kpis['day_night_ratio']:.2f}" if pd.notna(kpis['day_night_ratio']) else "n/a"
    )
    c3.metric("Baseload % of total", f"{kpis['baseload_pct']*100:.1f}%")
    c4.metric("After-hours %", f"{kpis['after_hours_pct']*100:.1f}%")
    c5.metric("Coeff. of variation", f"{kpis['coeff_var']*100:.1f}%")

    st.caption(
        f"Weekday mean kWh (30d): {kpis['weekday_kWh_last30']:.0f} | "
        f"Weekend: {kpis['weekend_kWh_last30']:.0f} | "
        f"Holidays: {kpis['holiday_kWh_last30'] if pd.notna(kpis['holiday_kWh_last30']) else 'n/a'}"
    )

    # ------------------------------------------------------------------
    # Tabs for different views (Operations first = default)
    # ------------------------------------------------------------------
    tab_ops, tab_overview, tab_anoms = st.tabs(
        ["Operations", "Overview", "Anomalies & Events"]
    )

    # ------------------------------------------------------------------
    # Operations tab: improved heatmap + baseload/peaks/overnight
    # ------------------------------------------------------------------
    with tab_ops:
        st.subheader("Interval heatmap (last 30 days)")

        # heatmap_df index = dates, columns = hours
        hm = heatmap_df.copy()
        hm.index = pd.to_datetime(hm.index)
        y_labels = hm.index.strftime("%b %d")

        fig_heatmap = px.imshow(
            hm.values,
            x=hm.columns,              # hours
            y=y_labels,                # formatted dates
            labels=dict(x="Hour of Day", y="Date", color="kWh"),
            aspect="auto",
            color_continuous_scale="YlOrRd",
            origin="lower",
        )
        fig_heatmap.update_xaxes(
            type="category",
            tickmode="linear",
            tick0=0,
            dtick=1,
            title_text="Hour of Day",
        )
        fig_heatmap.update_yaxes(
            title_text="Date",
            tickmode="linear",
        )
        fig_heatmap.update_layout(
            margin=dict(l=80, r=20, t=10, b=40),
            yaxis_autorange="reversed",  # most recent at bottom
            hoverlabel=dict(bgcolor="white"),
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Peaks vs baseload tables
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Top 5 peak intervals")
            st.dataframe(results["top5_peaks"])
        with cc2:
            st.subheader("Lowest baseload intervals (6–15)")
            st.dataframe(results["lowest_base"])

        # Overnight + holidays
        cc3, cc4 = st.columns(2)
        with cc3:
            st.subheader("Overnight baseload (last 5 days)")
            st.dataframe(results["overnight_avg"])
        with cc4:
            st.subheader("Holiday baseload")
            st.dataframe(results["holiday_base"])

    # ------------------------------------------------------------------
    # Overview tab: daily kWh + CDD/HDD, KPI context
    # ------------------------------------------------------------------
    with tab_overview:
        st.subheader("Daily kWh and rolling averages")

        fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
        # Daily kWh
        fig_daily.add_trace(
            go.Scatter(
                x=daily["Day"],
                y=daily["Consumption - kWh"],
                mode="lines",
                name="Daily kWh",
            ),
            secondary_y=False,
        )
        # 7‑day rolling
        fig_daily.add_trace(
            go.Scatter(
                x=daily["Day"],
                y=daily["Rolling7_avg_kWh"],
                mode="lines",
                name="7‑day avg",
            ),
            secondary_y=False,
        )
        # 30‑day rolling
        fig_daily.add_trace(
            go.Scatter(
                x=daily["Day"],
                y=daily["Rolling30_avg_kWh"],
                mode="lines",
                name="30‑day avg",
            ),
            secondary_y=False,
        )
        # CDD on secondary axis
        fig_daily.add_trace(
            go.Bar(
                x=daily["Day"],
                y=daily["CDD"],
                name="CDD",
                opacity=0.3,
            ),
            secondary_y=True,
        )

        # Bad‑day markers (CDD outliers)
        if not bad_days.empty:
            daily_idx = daily.set_index("Day")
            y_bad = [
                daily_idx.loc[d, "Consumption - kWh"]
                for d in bad_days["Day"]
                if d in daily_idx.index
            ]
            fig_daily.add_trace(
                go.Scatter(
                    x=bad_days["Day"],
                    y=y_bad,
                    mode="markers",
                    marker=dict(color="red", size=9),
                    name="Bad days",
                ),
                secondary_y=False,
            )

        fig_daily.update_xaxes(title_text="Day")
        fig_daily.update_yaxes(title_text="kWh", secondary_y=False)
        fig_daily.update_yaxes(title_text="CDD", secondary_y=True)

        st.plotly_chart(fig_daily, use_container_width=True)

    # ------------------------------------------------------------------
    # Anomalies & Events tab: anomaly scatter + tables
    # ------------------------------------------------------------------
    with tab_anoms:
        st.subheader("Interval kWh with anomaly overlay")

        if "last_recent" in results:
            last_recent = results["last_recent"]
            fig_interval = go.Figure()
            fig_interval.add_trace(
                go.Scatter(
                    x=last_recent["Interval Start"],
                    y=last_recent["Consumption - kWh"],
                    mode="lines",
                    name="kWh",
                )
            )

            if anomalies is not None and not anomalies.empty:
                fig_interval.add_trace(
                    go.Scatter(
                        x=anomalies["Time"],
                        y=anomalies["Actual_kWh"],
                        mode="markers",
                        marker=dict(color="red", size=9),
                        name="Anomalies",
                    )
                )

            fig_interval.update_xaxes(title_text="Interval start")
            fig_interval.update_yaxes(title_text="kWh")
            st.plotly_chart(fig_interval, use_container_width=True)

        st.subheader("Top 10 recent ML anomalies")
        if anomalies is not None and not anomalies.empty:
            st.dataframe(anomalies)
        else:
            st.write("No significant anomalies detected.")

        st.subheader("Weather-driven bad days (CDD outliers)")
        if not bad_days.empty:
            st.dataframe(bad_days)
        else:
            st.write("No significant weather-driven outliers.")

        dq = results.get("data_quality", {})
        if dq:
            st.subheader("Data quality checks")
            st.json(dq)
