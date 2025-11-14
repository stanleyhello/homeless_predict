from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
import streamlit as st

# ---------- LOAD ARTIFACTS ----------
APP_DIR = Path(__file__).resolve().parent
model = load(APP_DIR / "model.joblib")

try:
    pipeline = load(APP_DIR / "pipeline.joblib")
except FileNotFoundError:
    pipeline = None

FEATURE_COLUMNS = list(getattr(model, "feature_names_in_", []))
if pipeline and isinstance(pipeline, dict):
    FEATURE_COLUMNS = pipeline.get("feature_columns", FEATURE_COLUMNS)
if not FEATURE_COLUMNS:
    raise RuntimeError("Unable to determine model feature columns.")
LAG_COLUMNS = [col for col in FEATURE_COLUMNS if col.startswith("homeless_count_lag")]

DATA_PATH = APP_DIR / "with_covid_flag.csv"
data = pd.read_csv(DATA_PATH)
data["year_month"] = pd.to_datetime(data["year_month"])
data = data.sort_values("year_month").reset_index(drop=True)

if "precipitation" not in data.columns:
    data["precipitation"] = 0.0

data["year_month_ordinal"] = data["year_month"].map(lambda dt: dt.toordinal())
for lag in range(1, 7):
    data[f"homeless_count_lag{lag}"] = data["homeless_count"].shift(lag)

lag_drop_cols = [col for col in LAG_COLUMNS if col in data.columns]
data = data.dropna(subset=lag_drop_cols).reset_index(drop=True)
base = data.iloc[[-1]].copy()

# ---------- BUILD JAN 2024 BASE ROW ----------
jan = base.copy()
jan.loc[:, "year_month"] = "2024-01-01"

# If you have explicit month/year/time_index features, set them here as well

st.set_page_config(page_title="SD Homelessness Forecast", layout="centered")
st.title("Downtown San Diego Homelessness Forecast – What-if Simulator")

st.write("This tool forecasts the number of unhoused people for **Jan 2024** and lets you explore how changing key inputs affects the prediction.")

# ---------- READ BASELINE VALUES ----------
def get_val(col, default=0.0):
    return float(jan[col].iloc[0]) if col in jan.columns else default

temp0   = get_val("average_temp")
zori0   = get_val("ZORI")
unemp0  = get_val("unemployment_rate")
evict0  = get_val("evictions")
lag1_0  = get_val("homeless_count_lag1")
lag2_0  = get_val("homeless_count_lag2")
covid0  = int(get_val("covid", 0))
precip0 = get_val("precipitation")

# ---------- SLIDERS ----------
st.subheader("Adjust Jan 2024 inputs")

col1, col2 = st.columns(2)

with col1:
    avg_temp = st.slider(
        "Average temperature (°F)",
        min_value=40,
        max_value=95,
        value=int(round(temp0))
    )
    zori = st.slider(
        "Rent index (ZORI)",
        min_value=int(zori0 * 0.9),
        max_value=int(zori0 * 1.1),
        value=int(zori0)
    )
    covid = st.checkbox("COVID-related impact flag", value=bool(covid0))
    precip = st.slider(
        "Precipitation (inches)",
        min_value=0.0,
        max_value=5.0,
        value=float(np.round(precip0, 2)),
        step=0.1,
    )

with col2:
    unemp = st.slider(
        "Unemployment rate (%)",
        min_value=2.0,
        max_value=10.0,
        value=float(np.round(unemp0, 1)),
        step=0.1
    )
    evict = st.slider(
        "Evictions (monthly count)",
        min_value=0,
        max_value=int(max(evict0 * 1.5, evict0 + 50)),
        value=int(evict0)
    )

    # Optionally expose lag sliders, or keep the historical values fixed
    lag1 = st.number_input(
        "Homeless count last month (lag1)",
        value=int(lag1_0)
    )
    lag2 = st.number_input(
        "Homeless count two months ago (lag2)",
        value=int(lag2_0)
    )

# ---------- BUILD FEATURE ROW FOR PREDICTION ----------
def build_row(temp, z, u, e, c, p, l1, l2):
    row = jan.copy()
    row["average_temp"] = temp
    row["ZORI"] = z
    row["unemployment_rate"] = u
    row["evictions"] = e
    row["covid"] = int(c)
    row["precipitation"] = p
    row["homeless_count_lag1"] = l1
    row["homeless_count_lag2"] = l2
    row["year_month"] = pd.to_datetime(row["year_month"])
    row["year_month_ordinal"] = row["year_month"].map(lambda dt: dt.toordinal())
    return row

def predict(row_df):
    if pipeline is not None and not isinstance(pipeline, dict):
        X = pipeline.transform(row_df)
    else:
        X = row_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    y = model.predict(X)
    return float(y[0])

row_current = build_row(avg_temp, zori, unemp, evict, covid, precip, lag1, lag2)
pred_current = predict(row_current)

st.markdown("### Predicted homelessness for Jan 2024")
st.metric("Predicted count", f"{int(round(pred_current))}")

# ---------- WHAT-IF TABLES (5 ticks per feature) ----------
st.markdown("### What-if analysis by feature")

def scenario_table(feature, values):
    rows = []
    for v in values:
        r = row_current.copy()
        r[feature] = v
        rows.append({
            "feature": feature,
            "value": v,
            "predicted_homeless_count": int(round(predict(r)))
        })
    return pd.DataFrame(rows)

def five_ticks(min_v, max_v):
    return np.linspace(min_v, max_v, 5)

tab1, tab2, tab3, tab4 = st.tabs(["Temperature", "ZORI", "Unemployment", "Evictions"])

with tab1:
    vals = five_ticks(50, 90)
    st.write("Temperature what-if:")
    st.dataframe(scenario_table("average_temp", vals))

with tab2:
    vals = five_ticks(zori0 * 0.95, zori0 * 1.05)
    st.write("Rent index what-if:")
    st.dataframe(scenario_table("ZORI", vals))

with tab3:
    vals = five_ticks(max(2, unemp0 - 2), unemp0 + 2)
    st.write("Unemployment what-if:")
    st.dataframe(scenario_table("unemployment_rate", vals))

with tab4:
    vals = five_ticks(max(0, evict0 * 0.5), max(evict0 * 1.5, evict0 + 50))
    st.write("Evictions what-if:")
    st.dataframe(scenario_table("evictions", vals))
