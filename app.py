from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from joblib import load

# ------------------------------
# LOAD ARTIFACTS
# ------------------------------

APP_DIR = Path(__file__).resolve().parent

model = load_model(APP_DIR / "model_keras.keras", compile=False)
scaler = load(APP_DIR / "scaler.joblib")
feature_cols = load(APP_DIR / "feature_cols.joblib")
# Ensure the loaded object is always a simple 1-D list
if hasattr(feature_cols, "columns"):
    feature_cols = list(feature_cols.columns)
elif not isinstance(feature_cols, list):
    feature_cols = list(feature_cols)

# Load the last known row (2023-08)
base = pd.read_csv(APP_DIR / "last_row_2023.csv")

# Ensure correct types
for col in base.columns:
    base[col] = base[col].infer_objects(copy=False)

# ------------------------------
# PREP TARGET ROW FOR 2023-09
# ------------------------------

TARGET_DATE = "2023-09-01"
TARGET_LABEL = "September 2023"

target = base.copy()
target["year_month"] = TARGET_DATE
target["month"] = 9
target["year"] = 2023

# ----------------------------------------
# Extract baseline values for sliders
# ----------------------------------------

def get_val(col, default=0.0):
    return float(target[col].iloc[0]) if col in target.columns else default

avg_temp0 = get_val("average_temp")
zori0 = get_val("ZORI")
unemp0 = get_val("unemployment_rate")
evict0 = get_val("evictions")
covid0 = int(get_val("covid", 0))
precip0 = get_val("precipitation")

# Lag features (you created lag 3–8, but your UI only references lag1/lag2 earlier)
lag3_0 = get_val("count_lag_3")
lag4_0 = get_val("count_lag_4")
lag5_0 = get_val("count_lag_5")
lag6_0 = get_val("count_lag_6")
lag7_0 = get_val("count_lag_7")
lag8_0 = get_val("count_lag_8")

median_rent0 = get_val("median_rent_city_lag1")
temp_lag1_0 = get_val("avg_temp_lag1")
unemp_lag1_0 = get_val("unemployment_rate_lag1")
evict_lag1_0 = get_val("evictions_lag1")
precip_lag1_0 = get_val("precipitation_lag1")

cpi0 = get_val("cpi")
cpi1_0 = get_val("cpi_lag1")
cpi2_0 = get_val("cpi_lag2")
cpi3_0 = get_val("cpi_lag3")

# ------------------------------
# STREAMLIT PAGE
# ------------------------------

st.title("Housing Insecurity Predictor")
st.write(
    "Using data through August 2023, this tool predicts the homeless population "
    "in downtown San Diego for the following month. Adjust the parameters to see "
    "how changes in conditions may affect next month's projected count."
)
# st.subheader(f"Adjust inputs to forecast {TARGET_LABEL}")

prediction_container = st.container()

# ------------------------------
# SLIDERS — only for features users can modify
# ------------------------------

avg_temp = st.slider(
    "Average temperature (°F)",
    min_value=40, max_value=95,
    value=int(round(avg_temp0)),
)

zori = st.slider(
    "Rent index (ZORI)",
    min_value=int(zori0 * 0.9), max_value=int(zori0 * 1.1),
    value=int(zori0),
)

unemp = st.slider(
    "Unemployment rate (%)",
    min_value=2.0, max_value=12.0,
    value=float(np.round(unemp0, 1)),
    step=0.1,
)

evict = st.slider(
    "Evictions",
    min_value=0,
    max_value=max(2000, int(evict0 * 1.5)),
    value=int(evict0)
)

precip = st.slider(
    "Precipitation (inches)",
    min_value=0.0, max_value=5.0,
    value=float(np.round(precip0, 2)),
    step=0.1
)

covid = st.slider(
    "COVID Period (0/1)",
    min_value=0, max_value=1,
    value=int(covid0),
)

# ------------------------------
# BUILD INPUT ROW
# ------------------------------

row = target.copy()

# Replace editable fields
row["average_temp"] = avg_temp
row["ZORI"] = zori
row["unemployment_rate"] = unemp
row["evictions"] = evict
row["precipitation"] = precip
row["covid"] = covid

# Keep all other columns unchanged

# ------------------------------
# PREP FOR MODEL
# ------------------------------

# Reindex to model feature order
X = row.reindex(columns=feature_cols, fill_value=0)

# Scale
X_scaled = scaler.transform(X)

# Predict
y_pred = float(model.predict(X_scaled)[0][0])

# ------------------------------
# DISPLAY RESULT
# ------------------------------

with prediction_container:
    st.markdown("### Projected Unhoused Population:")
    st.metric(label=TARGET_LABEL, value=int(round(y_pred)))

st.markdown("---")

tab_train, tab_use, tab_limits = st.tabs(
    ["HOW WE TRAINED THE MODEL", "USE CASES", "LIMITATIONS"]
)

with tab_train:
    st.markdown(
        """
#### Choosing Our Scope
We focused specifically on downtown San Diego, because this area has the most consistent and long-running data on unsheltered homelessness. The Downtown San Diego Partnership (DSDP) provides a monthly observational count of individuals experiencing homelessness, giving us the strongest time series foundation.

#### Collecting the Datasets
- **Homeless Count (Target Variable)** — Source: Downtown San Diego Partnership (DSDP), monthly observational count in downtown San Diego.
- **Average Monthly Temperature** — Source: NOAA Weather NOWDATA (San Diego International Airport); chosen because the airport shares the modeled geography.
- **ZORI (Zillow Observed Rent Index)** — Mean rent for listings in the 35th–65th percentile range covering the City of San Diego.
- **Unemployment Rate** — Monthly unemployment rate for San Diego County. Source: Federal Reserve Bank of St. Louis (<https://fred.stlouisfed.org/series/CASAND5URN/>)
- **Evictions** — Source: CalMatters (<https://calmatters.org/housing/homelessness/2023/11/california-evictions-post-pandemic/>); chart by Jeanne Kuang (CalMatters), analysis by Tim Thomas (UC Berkeley Urban Displacement).
- **CPI (Consumer Price Index)** — Source: U.S. Bureau of Labor Statistics (<https://www.bls.gov/charts/consumer-price-index/consumer-price-index-by-category.htm#>); reflects inflationary pressure on households.
- **Precipitation** — Source: RCC ACIS Climate Data (<https://scacis.rcc-acis.org/>); total monthly precipitation for San Diego.
- **COVID Flag** — Binary feature marking March 2020 (emergency declarations) through June 2022 (end of emergency expansions and eviction moratoriums).

#### Initial Modeling Approaches
- **Ensemble Models** — Tried Random Forests and XGBoost; both severely overfit because of the small dataset and temporal structure.
- **Neural Network Approach** — Implemented a lightweight neural network with batch normalization to capture non-linear patterns without needing large data volumes.

#### Feature Engineering and Model Refinement
- **Feature Selection** — Combined exploratory data analysis with SHAP values. Removed shelter beds (annual updates only) and Industrial Production Index (duplicated CPI trends and dampened monthly variation).
- **Adding Lags** — Added lagged versions of key variables to capture delayed effects (e.g., rent shocks influencing homelessness months later). Tuned lag lengths individually to minimize bias and variance.

#### Variance Issues and How We Fixed Them
- **Challenge** — Model initially had too little variance, then too much after adding CPI lags.
- **Solution** — Adopted a randomized 75%/25% train-test split instead of chronological splitting, which improved generalization and stabilized predictions.

#### Final Neural Network Hyperparameters
- Learning rate: 0.05 (exponential decay at 0.96)
- Batch size: 5
- Epochs: 500

#### Final Model Performance
- **R² score:** 0.58 — explains 58% of the month-to-month variation, strong for a small, noisy social dataset.
- **Mean Absolute Error:** 159 people — acceptable for policy analysis and scenario testing (downtown counts typically range from 600–1,400), though not for precise monthly totals.

#### Other Models Tried
- **Prophet (Time-Series Model)** — Performed poorly; failed to capture structure (R² = –1.017).
"""
    )

with tab_use:
    st.markdown(
        """
This model can support San Diego organizations working on homelessness response, planning, and policy decisions. Below are examples of how key agencies could benefit from forecasting monthly homelessness trends.

- **Regional Task Force on Homelessness (RTFHSD)** — Leads research, policy, and funding distribution across the region.
  - Funding justification for emergency dollars or expanded beds when an uptick is predicted.
  - Policy evaluations and “what-if” simulations by adjusting inputs such as rent, unemployment, temperature, and evictions.
  - Communicating expected trends to policymakers and service providers earlier in the month.
- **County of San Diego – Office of Evaluation, Performance, and Analytics (OEPA)** — Provides analytics to county leadership and programs.
  - Early detection of upcoming increases via monthly briefings.
  - Resource allocation recommendations for outreach teams, funding, or housing vouchers.
  - Economic trend analysis by testing how rent, CPI, and unemployment shifts could translate into homelessness spikes.
- **City of San Diego – Homelessness Strategies & Solutions (HSS)** — Operates shelters, outreach, safe parking, and housing navigation.
  - Operational planning for shelter demand and outreach coverage.
  - Staffing and supply preparation (food, hygiene, transportation) when higher counts are expected.
  - Coordinating response with nonprofit partners in anticipation of increased encampment activity.
"""
    )

with tab_limits:
    st.markdown(
        """
#### Short-term prediction window only
- Forecasts just one month ahead and requires the latest real data to predict the next month.
- Best used as a “what-if” scenario explorer rather than a long-term forecasting system.
- Multi-year projections would likely need a simulation or system-dynamics model.

#### Limited historical data
- Only 115 months of observations, which is small for ML models tackling complex social issues.
- Captures broad patterns but may miss rare events or sharp spikes.

#### Missing external factors
- Excludes hard-to-measure drivers such as migration, encampment sweeps, sudden shelter changes, or targeted interventions.
- These factors can shift monthly counts in ways the model cannot capture.

#### Accuracy depends on realistic input ranges
- Sliders let you change next month’s temperature, rent, unemployment, etc., but the model is only reliable within historical ranges.
- Extreme values (e.g., jumping unemployment from 4% to 25% in one month) produce unrealistic outputs.
- Interpret results as most meaningful when exploring small or moderate adjustments.
"""
    )
