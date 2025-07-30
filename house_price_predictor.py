import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("housing.csv")

# Basic info
print("Dataset preview:")
print(df.head())

# Select input and output
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict new value
new_house = [[2500, 3, 10]]  # Area=2500 sqft, 3 bedrooms, 10 years old
predicted_price = model.predict(new_house)
print(f"Predicted Price for new house: ₹{predicted_price[0]:.2f}")
