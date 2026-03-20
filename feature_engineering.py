import pandas as pd
import matplotlib.pyplot as plt

print("Loading real ERA5 data...")
# 1. Load the data you just downloaded
df = pd.read_csv("real_heathrow_era5.csv")

# Ensure the time column is treated as dates
df['time'] = pd.to_datetime(df['time'])

# 2. FEATURE ENGINEERING: Wind Shear (Rate of Change)
# The .diff() function looks at the wind speed of the previous hour and subtracts it from the current hour.
df['wind_shear'] = df['wind_speed_100m'].diff().fillna(0)

# 3. FEATURE ENGINEERING: Defining a "Turbulence Event"
# UPDATED: Changed the threshold to 5 km/h so we catch the dangerous shifts during the storm
df['turbulence_risk'] = (df['wind_shear'].abs() > 5).astype(int)

# 4. Save the "Engineered" Dataset
df.to_csv("engineered_heathrow_era5.csv", index=False)
print("Added new features! Saved as 'engineered_heathrow_era5.csv'")

# 5. Plotting the Wind Shear
plt.figure(figsize=(10, 5))

# Plot the raw wind speed
plt.plot(df['time'], df['wind_speed_100m'], color='gray', label='Raw Wind Speed (km/h)', alpha=0.5)

# Highlight the turbulence risks
danger_zones = df[df['turbulence_risk'] == 1]
plt.scatter(danger_zones['time'], danger_zones['wind_speed_100m'], 
            color='red', s=100, label='High Turbulence Risk (Rapid Wind Change)')

plt.title("Identifying Turbulence Risk Zones (Storm Isha)")
plt.xlabel("Date")
plt.ylabel("Wind Speed (km/h)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("turbulence_risk_plot.png")
print("Graph saved as 'turbulence_risk_plot.png'!")
plt.show()
