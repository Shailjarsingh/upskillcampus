import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection and Preparation
train_data = pd.read_csv('/content/train_aWnotuB.csv')
normal_data = pd.read_csv('/content/datasets_8494_11879_test_BdBKkAj.csv')
np.random.seed(42)
days = pd.date_range(start='2023-01-01', periods=30, freq='D')
junctions = ['Junction_1', 'Junction_2', 'Junction_3', 'Junction_4']
peak_hours = list(range(7, 10)) + list(range(16, 19))
data = []

for day in days:
    for junction in junctions:
        for hour in range(24):
            traffic_volume = np.random.poisson(lam=100 if hour in peak_hours else 30)
            data.append([day, junction, hour, traffic_volume])

traffic_data = pd.DataFrame(data, columns=['Date', 'Junction', 'Hour', 'Traffic_Volume'])

# Step 2: Data Preprocessing
traffic_data['Day_of_Week'] = traffic_data['Date'].dt.dayofweek
traffic_data['Is_Holiday'] = traffic_data['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)

# Step 3: Data Analysis
X = traffic_data[['Hour', 'Day_of_Week', 'Is_Holiday']]
y = traffic_data['Traffic_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

# Step 4: Visualization
plt.figure(figsize=(14, 7))
sns.lineplot(data=traffic_data, x='Date', y='Traffic_Volume', hue='Junction')
plt.title('Traffic Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Traffic Volume')
plt.legend(title='Junction')
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(data=traffic_data.groupby('Hour')['Traffic_Volume'].mean().reset_index(), x='Hour', y='Traffic_Volume')
plt.title('Average Traffic Volume by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Traffic Volume')
plt.show()

# Step 5: Reporting
print("Key Findings:")
print(f"1. The mean squared error of the traffic volume prediction model is: {mse:.2f}")
print("2. Coefficients of the linear regression model indicate the impact of each feature on traffic volume:")
print(coefficients)
print("3. Visualizations show the variation of traffic volume over time and by hour, highlighting peak traffic periods.")
