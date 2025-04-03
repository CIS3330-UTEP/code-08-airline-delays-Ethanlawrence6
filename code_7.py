import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)
#ARR_DELAY is the column name that should be used as dependent variable (Y).
print(df.info())
print(df.describe())
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
plt.hist(df['ARR_DELAY'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

features = ['ARR_DELAY', 'DEP_DELAY', 'DISTANCE', 'CANCELLED', 'CRS_ELAPSED_TIME']
df_model = df[features].dropna()

X = df_model[['DEP_DELAY', 'DISTANCE', 'CRS_ELAPSED_TIME']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
Y = df_model['ARR_DELAY']

model = sm.OLS(Y, X).fit()

print(model.summary())

fig, ax = plt.subplots()
fig = sm.graphics.influence_plot(model, 'DEP_DELAY', ax=ax)
plt.title('Fit Plot: ARR_DELAY vs DEP_DELAY')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Arrival Delay (minutes)')
plt.grid(True)
plt.show()