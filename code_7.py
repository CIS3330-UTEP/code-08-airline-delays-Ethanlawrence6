import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("Flight_Delays_2018.csv")

df = df[['ARR_DELAY', 'DEP_DELAY', 'DISTANCE', 'ORIGIN', 'DEST', 'AIR_TIME']]
df = df.dropna()

print("Summary Statistics:")
print(df.descriibe())

plt.hist(df['ARR_DELAY'], bins=50)
plt.title("Arrival Delay Distribution")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Frequency")
plt.show()

correlation = df.corr(numeric_only=True)['ARR_DELAY'].sort_values(ascending=False)
print("\nCorrelation with ARR_DELAY:\n", correlation)

top_airports = df['ORIGIN'].value_counts().head(5).index.tolist()
df_filtered = df[df['ORIGIN'].isin(top_airports)]

df_filtered.boxplot(column='ARR_DELAY', by='ORIGIN')
plt.title("Arrival Delay by Origin Airport")
plt.suptitle("")
plt.xlabel("Origin Airport")
plt.ylabel("Arrival Delay (min)")
plt.show()

X = df_filtered[['DEP_DELAY', 'DISTANCE', 'AIR_TIME']]
Y = df_filtered['ARR_DELAY']

X = sm.add_constant(X)  

model = sm.OLS(Y, X).fit()

print(model.summary())
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, "DEP_DELAY", ax=ax)
plt.show()