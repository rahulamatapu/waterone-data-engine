"""WaterOne Water Quality Health Monitor & Telemetry Lakehouse Dashboard."""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="WaterOne Health Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path("database/water_lakehouse.duckdb")


@st.cache_data
def get_anomalies_for_contamination(contamination: float) -> pd.DataFrame:
    from src.models.anomaly_detector import WaterQualityAnomalyDetector
    detector = WaterQualityAnomalyDetector()
    return detector.train_and_score(custom_contamination=contamination, persist_to_duckdb=False)


def load_data():
    data = {}
    con = duckdb.connect(":memory:")

    daily_pq = Path("data/gold/mart_daily_water_quality.parquet")
    if daily_pq.exists():
        data["daily"] = con.execute(f"SELECT * FROM read_parquet('{daily_pq}') ORDER BY date ASC;").df()

    samples_pq = Path("data/silver/fact_water_quality_samples.parquet")
    contam_pq = Path("data/silver/dim_contaminant.parquet")
    if samples_pq.exists() and contam_pq.exists():
        data["compliance"] = con.execute(f"""
            WITH primary_samples AS (
                SELECT 
                    c.contaminant_id,
                    c.display_name,
                    c.chemical_symbol,
                    c.category,
                    s.result_value,
                    s.unit,
                    c.mcl,
                    c.action_level,
                    c.warning_threshold,
                    c.critical_threshold,
                    s.range_min,
                    s.range_max,
                    s.met_standard,
                    s.category as sample_category,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.contaminant_id 
                        ORDER BY CASE WHEN s.category ILIKE 'Regulated -%' THEN 1 ELSE 2 END, s.sample_id
                    ) as rn
                FROM read_parquet('{samples_pq}') s
                JOIN read_parquet('{contam_pq}') c ON s.contaminant_id = c.contaminant_id
                WHERE s.result_value IS NOT NULL
            )
            SELECT * FROM primary_samples WHERE rn = 1 ORDER BY category, display_name;
        """).df()

        data["all_samples"] = con.execute(f"""
            SELECT 
                parameter_clean AS "Parameter",
                category AS "Category",
                CASE 
                    WHEN result_value IS NOT NULL THEN CAST(ROUND(result_value, 2) AS VARCHAR) || ' ' || COALESCE(unit, '')
                    ELSE 'ND'
                END AS "Tested Level",
                COALESCE(CAST(mcl_value AS VARCHAR) || ' ' || COALESCE(unit, ''), '—') AS "MCL / Standard",
                COALESCE(CAST(mclg_value AS VARCHAR) || ' ' || COALESCE(unit, ''), '—') AS "MCLG (Goal)",
                CASE 
                    WHEN range_min IS NOT NULL AND range_max IS NOT NULL THEN CAST(ROUND(range_min, 2) AS VARCHAR) || ' - ' || CAST(ROUND(range_max, 2) AS VARCHAR) || ' ' || COALESCE(unit, '')
                    ELSE '—'
                END AS "Range",
                COALESCE(sample_frequency, 'Annual') AS "Frequency",
                CASE WHEN met_standard THEN '✅ Compliant' ELSE '❌ Exceeded' END AS "Status"
            FROM read_parquet('{samples_pq}')
            ORDER BY category, parameter_clean;
        """).df()

    fac_pq = Path("data/silver/dim_facility.parquet")
    if fac_pq.exists():
        data["facilities"] = con.execute(f"SELECT * FROM read_parquet('{fac_pq}') ORDER BY facility_type, facility_name;").df()

    vio_pq = Path("data/silver/fact_violations.parquet")
    if vio_pq.exists():
        data["violations"] = con.execute(f"SELECT * FROM read_parquet('{vio_pq}') ORDER BY compliance_begin_date DESC;").df()

    visits_pq = Path("data/silver/fact_site_visits.parquet")
    if visits_pq.exists():
        data["site_visits"] = con.execute(f"SELECT * FROM read_parquet('{visits_pq}') ORDER BY visit_date DESC;").df()

    con.close()
    return data


# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/water.png", width=64)
st.sidebar.title("WaterOne Monitor")
st.sidebar.caption("PWS ID: KS2009110 | Johnson County, KS")
st.sidebar.markdown("---")

st.sidebar.subheader("🎛️ Anomaly Sensitivity")
contamination_slider = st.sidebar.slider(
    "Isolation Forest Contamination",
    min_value=0.01,
    max_value=0.15,
    value=0.05,
    step=0.01,
    help="Higher values increase detection sensitivity for upstream river stress."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Lakehouse Architecture:**
    - 🥉 **Bronze**: Raw PDF, CSV & USGS Telemetry
    - 🥈 **Silver**: Pandera-Validated Star Schema
    - 🥇 **Gold**: Feature Store & Anomaly Marts
    - 🗄️ **Engine**: Local Embedded DuckDB
    """
)

# --- MAIN DASHBOARD ---
st.title("💧 Water District No. 1 of Johnson County")
st.subheader("Real-Time Water Quality Health Monitor & Upstream Stress Platform")

data = load_data()

if not data or "daily" not in data:
    st.warning("⚠️ Lakehouse database is not initialized yet. Run `poetry run waterpipe run` from your terminal to populate the data.")
    st.stop()

df_daily = data.get("daily", pd.DataFrame())
df_comp = data.get("compliance", pd.DataFrame())
df_fac = data.get("facilities", pd.DataFrame())

# Dynamically score anomalies based on the user-selected contamination sensitivity slider
df_anom = get_anomalies_for_contamination(contamination_slider)

# 1. EXECUTIVE KPI SUMMARY
st.markdown("### 1. Executive Health Scorecard & Pre-Breach Warnings")

latest_day = df_daily.iloc[-1] if not df_daily.empty else None
latest_safety_index = latest_day["water_safety_index"] if latest_day is not None else 98.5
latest_blended_turb = latest_day["blended_turbidity_ntu"] if latest_day is not None else 28.5
latest_flow = latest_day["total_upstream_flow_cfs"] if latest_day is not None else 49500.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Overall Water Safety Score",
        value=f"{latest_safety_index:.1f} / 100",
        delta="Optimal (100% EPA Compliance)" if latest_safety_index >= 85 else "- Elevated River Turbidity",
    )
with col2:
    st.metric(
        label="Blended Raw Intake Turbidity",
        value=f"{latest_blended_turb:.2f} NTU",
        delta=f"Baseline finished: 0.23 NTU",
        delta_color="inverse",
    )
with col3:
    st.metric(
        label="Total Upstream River Flow",
        value=f"{latest_flow:,.0f} cfs",
        delta="Missouri + Kansas Rivers",
    )
with col4:
    active_facs = len(df_fac[df_fac["facility_activity"] == "Active"]) if not df_fac.empty else 0
    st.metric(
        label="Operational Facilities",
        value=f"{active_facs} Active",
        delta=f"of {len(df_fac)} registered facilities",
    )

# Pre-Breach Alert Banner
active_warnings = df_anom[df_anom["alert_level"] != "NORMAL"] if not df_anom.empty else pd.DataFrame()
crit_count = int((df_anom["alert_level"] == "CRITICAL_ALERT").sum()) if not df_anom.empty else 0
warn_count = int((df_anom["alert_level"] == "PRE_BREACH_WARNING").sum()) if not df_anom.empty else 0

st.caption(f"🎯 **Model Sensitivity**: Contamination = `{contamination_slider * 100:.0f}%` | Flagged Events: `{warn_count} Warnings`, `{crit_count} Critical`")

if not active_warnings.empty:
    top_anom = active_warnings.iloc[0]
    alert_type = top_anom["alert_level"]
    alert_box = st.error if alert_type == "CRITICAL_ALERT" else st.warning
    alert_box(
        f"🚨 **{alert_type} Detected on {str(top_anom['date'])[:10]}** | "
        f"**Severity Index:** {top_anom['severity_index']:.1f}/100 | "
        f"**Driver:** `{top_anom['primary_contributing_factor']}`\n\n"
        f"👉 **Operator Guidance:** {top_anom['recommended_action']}"
    )

    with st.expander(f"🔍 Inspect All {len(active_warnings)} Flagged Outlier Dates (Sensitivity = {contamination_slider * 100:.0f}%)"):
        st.dataframe(
            active_warnings[[
                "date", "severity_index", "alert_level",
                "primary_contributing_factor", "blended_turbidity_ntu", "river_discharge_cfs", "recommended_action"
            ]],
            use_container_width=True,
        )
else:
    st.success(f"✅ **All River Intakes Nominal**: No pre-breach alerts triggered at `{contamination_slider * 100:.0f}%` sensitivity threshold.")

st.markdown("---")

# 2. KEY REGULATED CONTAMINANTS BENCHMARK
st.markdown("### 2. Regulatory Compliance & Contaminant Benchmarks")
if not df_comp.empty:
    c_cols = st.columns(4)
    key_params = ["Lead", "Copper", "Fluoride", "Turbidity"]
    for i, p_name in enumerate(key_params):
        match = df_comp[df_comp["display_name"].str.contains(p_name, case=False, na=False)]
        with c_cols[i % 4]:
            if not match.empty:
                r = match.iloc[0]
                val = r["result_value"]
                unit = r["unit"]

                # Determine regulatory limit type (EPA MCL vs EPA Action Level)
                if pd.notna(r.get("mcl")):
                    limit_label = f"EPA MCL: {r['mcl']} {unit}"
                    is_safe = val <= r["mcl"] if val is not None else True
                elif pd.notna(r.get("action_level")):
                    limit_label = f"Action Level: {r['action_level']} {unit}"
                    is_safe = val <= r["action_level"] if val is not None else True
                elif pd.notna(r.get("critical_threshold")):
                    limit_label = f"Limit: {r['critical_threshold']} {unit}"
                    is_safe = val <= r["critical_threshold"] if val is not None else True
                else:
                    limit_label = "Regulated Safe"
                    is_safe = True

                status_delta = f"{limit_label} (Compliant)" if is_safe else f"{limit_label} (Exceeded)"
                st.metric(
                    label=f"{r['display_name']} ({unit})",
                    value=f"{val:.2f} {unit}" if val is not None else "ND",
                    delta=status_delta,
                    delta_color="normal" if is_safe else "inverse",
                )

    df_all_samples = data.get("all_samples", pd.DataFrame())

    with st.expander("📋 View Complete Water Quality Parameters & Standards"):
        tab_primary, tab_all = st.tabs([
            "🏆 Core Regulated Benchmarks (10 Parameters)",
            f"🔬 All Tested Parameters ({len(df_all_samples)} Parameters & Secondary Standards)"
        ])

        with tab_primary:
            def format_primary_row(row):
                v = row["result_value"]
                u = row["unit"]
                if pd.notna(row.get("mcl")):
                    std = f"{row['mcl']} {u}"
                    stype = "EPA MRDL" if "Chlorine" in row["display_name"] else "EPA Primary MCL"
                elif pd.notna(row.get("action_level")):
                    std = f"{row['action_level']} {u}"
                    stype = "EPA Action Level (AL)"
                elif pd.notna(row.get("critical_threshold")):
                    std = f"{row['critical_threshold']} {u}"
                    stype = "Critical Standard"
                else:
                    std = "Regulated Safe"
                    stype = "Standard"

                warn = f"{row['warning_threshold']} {u}" if pd.notna(row.get("warning_threshold")) else "—"
                sym = f" ({row['chemical_symbol']})" if pd.notna(row.get("chemical_symbol")) and row["chemical_symbol"] else ""

                return {
                    "Contaminant": f"{row['display_name']}{sym}",
                    "Category": row["category"],
                    "WaterOne Level": f"{v:.2f} {u}" if v is not None else "ND",
                    "Regulatory Standard": std,
                    "Standard Type": stype,
                    "Warning Threshold": warn,
                    "Compliance Status": "✅ Met Standard" if row.get("met_standard", True) else "❌ Exceeded"
                }

            formatted_primary = pd.DataFrame([format_primary_row(r) for _, r in df_comp.iterrows()])
            st.dataframe(formatted_primary, width="stretch", hide_index=True)

        with tab_all:
            if not df_all_samples.empty:
                st.dataframe(df_all_samples, width="stretch", hide_index=True)
            else:
                st.info("No detailed sample records loaded.")

st.markdown("---")

# 3. UPSTREAM RIVER STRESS & SOURCE BLENDING MONITOR
st.markdown("### 3. Upstream River Stress & WaterOne Intake Blending Monitor")
st.caption(
    "Blending Weights: Missouri River (51.4%) + Wolcott Collector (27.5%) + Kansas River (19.6%) + Deep Wells (1.6%)"
)

if not df_daily.empty:
    tab1, tab2, tab3 = st.tabs(["Turbidity & Clarity Dynamics", "Streamflow Discharges (cfs)", "Treatment Stress Ratio"])

    with tab1:
        st.line_chart(
            df_daily.set_index("date")[["blended_turbidity_ntu", "turbidity_7d_ma", "kansas_river_turbidity_fnu", "missouri_river_turbidity_fnu"]],
            use_container_width=True,
        )
    with tab2:
        st.line_chart(
            df_daily.set_index("date")[["kansas_river_flow_cfs", "missouri_river_flow_cfs", "total_upstream_flow_cfs"]],
            use_container_width=True,
        )
    with tab3:
        st.area_chart(
            df_daily.set_index("date")[["treatment_stress_ratio"]],
            use_container_width=True,
        )

st.markdown("---")

# 4. DIMENSIONAL INFRASTRUCTURE INSPECTOR
st.markdown("### 4. Dimensional Infrastructure & Sanitary Survey Inspector")

if not df_fac.empty:
    fac_tab1, fac_tab2 = st.tabs(["Facilities Master Inventory", "KDHE Site Visits & Sanitary Surveys"])

    with fac_tab1:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            selected_type = st.selectbox(
                "Filter Facility Type",
                options=["All"] + sorted(list(df_fac["facility_type"].unique())),
            )
        filtered_fac = df_fac if selected_type == "All" else df_fac[df_fac["facility_type"] == selected_type]
        st.dataframe(
            filtered_fac[[
                "facility_id", "facility_name", "facility_type",
                "facility_activity", "water_type", "filtration_status"
            ]],
            use_container_width=True,
        )

    with fac_tab2:
        df_visits = data.get("site_visits", pd.DataFrame())
        if not df_visits.empty:
            st.dataframe(
                df_visits[[
                    "visit_id", "visit_date", "visit_reason",
                    "has_findings", "treatment_evaluation", "visit_comments"
                ]],
                use_container_width=True,
            )

st.markdown("---")
st.caption("WaterOne Data Engine • Local Columnar Lakehouse • DuckDB + Apache Parquet + Pandera + Scikit-Learn")

