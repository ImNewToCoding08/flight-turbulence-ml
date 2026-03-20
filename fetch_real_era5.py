import requests
import pandas as pd
import matplotlib.pyplot as plt

print("Connecting to the ERA5 Climate Database...")

# 1. Setting up the API Request
# An API (Application Programming Interface) is just a URL that returns pure data instead of a webpage.
url = "https://archive-api.open-meteo.com/v1/archive"

# 2. Defining our Flight Parameters
# We are asking for weather data exactly at London Heathrow Airport.
# We are looking at a specific week in January 2024 when Storm Isha hit the UK.
params = {
    "latitude": 51.47,        # Heathrow Latitude
    "longitude": -0.45,       # Heathrow Longitude
    "start_date": "2024-01-20", 
    "end_date": "2024-01-25",
    # We are asking for Temperature, Pressure, and Wind Speed from the ERA5 model
    "hourly": ["temperature_2m", "surface_pressure", "wind_speed_100m"],
    "models": "era5"          
}

# 3. Fetching the Data
# The 'requests' library acts like a browser, going to the URL with our parameters and bringing back the data.
response = requests.get(url, params=params)
data = response.json() # .json() converts the web data into a readable Python dictionary

# 4. Converting to Pandas
# We isolate just the 'hourly' data section and turn it into a Pandas DataFrame (our digital spreadsheet).
df = pd.DataFrame(data['hourly'])

# Convert the text timestamps into proper chronological time objects
df['time'] = pd.to_datetime(df['time'])

# 5. Saving YOUR Real Data
# This saves the real ERA5 data directly into your VS Code folder!
df.to_csv("real_heathrow_era5.csv", index=False)
print("Success! Saved as 'real_heathrow_era5.csv'")

# 6. Plotting the Storm
plt.figure(figsize=(10, 5))
plt.plot(df['time'], df['wind_speed_100m'], color='red', linewidth=2)
plt.title("Real ERA5 Wind Speeds at Heathrow (Storm Isha, Jan 2024)")
plt.xlabel("Date")
plt.ylabel("Wind Speed (km/h)")
plt.grid(True)
plt.tight_layout()

plt.savefig("real_storm_plot.png")
print("Graph saved as 'real_storm_plot.png'!")
plt.show()
