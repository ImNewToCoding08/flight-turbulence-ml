import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading engineered data...")
# 1. Load the data we prepared in the last step
df = pd.read_csv("engineered_heathrow_era5.csv")

# 2. Define our "Features" (X) and our "Target" (y)
# X represents the data the AI is allowed to look at.
X = df[['temperature_2m', 'surface_pressure', 'wind_speed_100m', 'wind_shear']]
# y is the answer key we are trying to predict (Turbulence Risk).
y = df['turbulence_risk']

# 3. Splitting the Data
# We hide 20% of the data from the AI to use as a "test" later, ensuring it doesn't just memorize the answers.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Building the Model
print("\nTraining the Random Forest AI...")
# We build a "forest" of 100 decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# The .fit() command is where the actual "Machine Learning" happens. 
# The model looks at the training features (X_train) and learns how they correlate to turbulence (y_train).
model.fit(X_train, y_train)

# 5. Testing the Model
print("Testing the AI on data it has never seen...")
# We ask the model to predict turbulence on the hidden 20% of data (X_test)
predictions = model.predict(X_test)

# 6. Grading the Results
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Performance Report:")
print(classification_report(y_test, predictions))

# 7. Finding out what the AI thinks is most important
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance (%)': model.feature_importances_ * 100
}).sort_values(by='Importance (%)', ascending=False)

print("\nWhat data matters most to the AI?")
print(feature_importance)
