from matplotlib import pyplot as plt
import seaborn as sns, pandas as pd, numpy as np

df=pd.read_csv("manhattan.csv")
df = df.select_dtypes(include=['number'])
df

df.mean()

df.mode()

df.median()

print(f"Max Rent is: {df.max()['rent']}")
print(f"Min Rent is: {df.max()['rent']}")
print(f"Difference in Rent is: {df.max()['rent']-df.min()['rent']}")

df.var()

df.std()

df.skew()

df.kurt()

sns.boxplot(x="building_age_yrs", data=df)
plt.title("Boxplot")
plt.show()

sns.barplot(x=df["min_to_subway"],y=df["building_age_yrs"],palette="viridis")
plt.xlabel('Mins to Subway')
plt.ylabel('Building Age')
plt.title('Bar Graph')
plt.show()

sns.histplot(df['rent'], bins=10, kde=False, color='blue')
plt.xlabel('Rent')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

sns.scatterplot(x="rent",y="floor",data=df)
plt.title('Scatterplot')
plt.show()

sns.lineplot(x="rent",y="bedrooms",data=df)
plt.title("Lineplot")
plt.show()
