# Housing Insecurity Predictor (Hackathon)

Streamlit demo and lightweight neural network that forecasts next-month unsheltered homelessness in downtown San Diego. Built for SDSU Big Data Hackathon; deployed at https://housinginsecuritypredictor.streamlit.app/.

## What the app does
- Lets you adjust next-month drivers (rent index, unemployment, evictions, temperature, precipitation, COVID flag) and instantly see the projected downtown count.
- Uses August 2023 as the most recent baseline and predicts September 2023 in-app (`app.py`).
- Pretrained artifacts are checked in: `model_keras.keras`, `scaler.joblib`, `feature_cols.joblib`, and `last_row_2023.csv`.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Recreate training
- Data: `dataset.csv` (115 monthly observations after cleaning). Target is `homeless_count`.
- Script: `train_nn.py` builds lags, fits the neural net, prints test R², and rewrites the artifacts + `last_row_2023.csv`.
- Models tried: random forest, XGBoost, Prophet (poor fit); the batch-normalized neural net performed best for this dataset.

## Data sources (downtown San Diego focus)
- Homeless count (target): Downtown San Diego Partnership monthly observational count.
- Average monthly temperature: NOAA NOWData (San Diego International Airport).
- Zillow Observed Rent Index (ZORI): City of San Diego.
- Unemployment rate: FRED series CASAND5URN (San Diego County).
- Evictions: CalMatters analysis of Judicial Council data (Jeanne Kuang/Tim Thomas).
- CPI: U.S. Bureau of Labor Statistics.
- Precipitation: RCC ACIS climate data.
- COVID flag: 1 from Mar 2020–Jun 2022 to capture emergency-era disruptions.
- Dropped: shelter beds (annual updates) and Industrial Production Index (overlaps CPI, smoothed variance).

## Feature engineering and model
- Lags for homeless_count (months 3–8) plus 1-month lags for rent, temperature, unemployment, evictions, precipitation, and CPI; CPI also lagged 2–3 months to reflect slower pass-through.
- Added year and month to capture seasonality; removed rows with incomplete lags and interpolated minor gaps.
- Neural net: dense layers with batch normalization; learning rate 0.05 with exponential decay (0.96), batch size 5, 500 epochs; 75/25 randomized train/test split.
- Performance: R² ≈ 0.58, MAE ≈ 159 people (suitable for scenario analysis, not exact counts).
- Observations: ensemble models overfit; heavy feature sets (CPI, IPI, shelter beds, precipitation, beds) could hurt variance; Prophet underperformed (R² ≈ -1.0).

## Hackathon log (high level)
- **Saturday:** scoped to downtown due to data availability; gathered mentor feedback; worried about dataset size.
- **Sunday:** collected weather, rent, evictions, unemployment, CPI, precipitation, and downtown counts; wrangled GIS homeless data into a master CSV.
- **Monday:** linear regression → random forest → XGBoost (poor fits); added lags; tried COVID flag; early neural net experiments.
- **Tuesday:** XGBoost tuning, added trend and seasonal features; overfitting; neural net with lagged features and seasonality improved; CPI lags fixed variance; achieved ~0.5 R² before later regressions.
- **Wednesday:** realized earlier evaluation mistake; retraced to simpler feature set; random forest best at ~-0.8; trimmed abnormal months; delta-prediction idea; renewed focus on UI for interactive scenario testing.
- **Thursday:** built Streamlit app with sliders; documented training and limitations; targeted use cases for San Diego agencies; final neural net artifacts saved for deployment.

## Intended users and use cases
- Regional Task Force on Homelessness (RTFHSD): funding justifications and what-if policy simulations.
- County OEPA: early-warning analytics for outreach resourcing.
- City Homelessness Strategies & Solutions: staffing and shelter capacity planning.

## Limitations and future ideas
- Short horizon (one-month-ahead) and small dataset; best as a scenario explorer.
- Sensitive to unrealistic slider values—keep within historical ranges.
- Missing drivers like migration, encampment sweeps, sudden shelter openings/closures.
- Future: richer policy/seasonal dummies, simulation or system-dynamics model for multi-year forecasts, automated monthly data refresh and retraining.

