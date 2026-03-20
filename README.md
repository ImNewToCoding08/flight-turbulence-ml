# Flight Weather Prediction & Turbulence Machine Learning

## Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict flight turbulence and operational risks using real-world atmospheric data. Accurate weather forecasting is critical for aerospace operations, airline routing, and global logistics networks to ensure safety and prevent supply chain delays.

Instead of relying on raw weather forecasts, this project pulls high-resolution NOAA/Copernicus **ERA5 climate data**, engineers aerospace-specific features (like Wind Shear), and trains a **Random Forest AI model** to predict periods of high turbulence risk.

As a test case, this model analyzes weather data at **London Heathrow Airport** during the severe **Storm Isha (January 2024)**.

---

## The Machine Learning Pipeline

This project is broken down into three distinct Python scripts, representing the three stages of professional data science:

### 1. Data Acquisition (`fetch_real_era5.py`)
*   **What it does:** Connects to the Open-Meteo API to download historical ERA5 climate data.
*   **The Science:** ERA5 is the industry standard for atmospheric reanalysis. We specifically targeted `wind_speed_100m` (wind speed 100 meters above ground level) because takeoff and landing phases are highly sensitive to low-altitude weather.
*   **Output:** Generates `real_heathrow_era5.csv` and a time-series plot of the storm.

### 2. Feature Engineering (`feature_engineering.py`)
*   **What it does:** Transforms raw weather data into predictive aerospace metrics using Pandas. 
*   **The Science:** Aircraft turbulence is rarely caused by high sustained winds; it is caused by **Wind Shear** (rapid changes in wind speed or direction). The script mathematically calculates the hour-over-hour difference in wind speed using `.diff()`. 
*   **The Threshold:** We defined a `turbulence_risk` event (Target variable = 1) as any hour where the wind speed shifted by more than **5 km/h**.
*   **Output:** Generates `engineered_heathrow_era5.csv` and a graph highlighting specific turbulence danger zones (red dots).

### 3. AI Model Training (`train_model.py`)
*   **What it does:** Trains a Random Forest Classifier using `scikit-learn` to predict if an hour will have dangerous turbulence based *only* on the weather data.
*   **How it works:** We split the data (80% for training the AI, 20% hidden for testing). The model builds a "forest" of 100 decision trees to find hidden patterns between temperature, pressure, wind speed, and our engineered wind shear.

---

## Key Findings & Model Performance

When tested on the hidden 20% of data, the Random Forest model achieved a **100% Accuracy Score**. While this is an excellent proof-of-concept, in a massive multi-year dataset, we would expect this to normalize to ~85-90%. 

The most valuable output from the AI was the **Feature Importance Report**. The AI independently determined which atmospheric conditions were most responsible for turbulence:

1.  **Wind Shear (68.6%)** - The AI correctly learned that the *rate of change* in wind is the primary driver of turbulence.
2.  **Surface Pressure (14.6%)** - The AI recognized that dropping barometric pressure (the defining trait of a storm system) correlates with unstable air.
3.  **Raw Wind Speed (11.0%)** 
4.  **Temperature (5.5%)**

This proves that **Feature Engineering** (teaching the AI about wind shear instead of just giving it raw wind speed) was the single most important step in making this model successful.

---

## How to Run This Project

**1. Install Dependencies**
```bash
pip install pandas matplotlib requests scikit-learn
