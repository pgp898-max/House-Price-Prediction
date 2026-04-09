import pandas as pd

df = pd.read_csv(r"C:\Users\drgha\OneDrive\Desktop\60009240274\house price prediction\data\Housing.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()
df.dropna(inplace=True)
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()
import pickle

with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)