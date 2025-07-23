import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load and inspect data
df = pd.read_csv("Mall_Customers.csv")
df_display = df.head().copy()
df_display.index = range(1, len(df_display) + 1)
print("First few rows:\n", df_display)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nCustomerID unique:", df['CustomerID'].is_unique)

# 2. Descriptive statistics
desc_stats = df.describe()
desc_stats.index = range(1, len(desc_stats) + 1)
print("\nDescriptive statistics:\n", desc_stats)
gender_counts = df['Gender'].value_counts()
gender_counts.index = range(1, len(gender_counts) + 1)
print("\nGender counts:\n", gender_counts)

# 3. Feature engineering for groups
df['Age_Group'] = pd.cut(df['Age'], bins=[17,25,35,50,70], labels=['18-25','26-35','36-50','51+'])
df['Income_Bracket'] = pd.cut(df['Annual Income (k$)'], bins=[0,40,70,100,150], labels=['Low','Medium','High','Very High'])

# 4. Visualizations (optional, show if running interactively)
plt.figure(figsize=(4,3))
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(6,3))
sns.histplot(df['Age'], bins=15, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,3))
sns.histplot(df['Annual Income (k$)'], bins=15, kde=True)
plt.title("Annual Income Distribution")
plt.show()

plt.figure(figsize=(6,3))
sns.histplot(df['Spending Score (1-100)'], bins=15, kde=True)
plt.title("Spending Score Distribution")
plt.show()

# 5. Correlation checks
num_cols = ['Age','Annual Income (k$)','Spending Score (1-100)']
cor = df[num_cols].corr()
print("\nCorrelation matrix:\n", cor.round(2))
plt.figure(figsize=(4,3))
sns.heatmap(cor, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Pearson r/p-values
print("\nPearson r and p-values:")
for i, col1 in enumerate(num_cols):
    for j in range(i+1, len(num_cols)):
        col2 = num_cols[j]
        r, p = pearsonr(df[col1], df[col2])
        print(f"{col1} & {col2}: r={r:.2f}, p={p:.3f}")

# Normality check
print("\nShapiro-Wilk Test (Normality):")
for col in num_cols:
    stat, p = shapiro(df[col])
    print(f"{col}: stat={stat:.3f}, p={p:.3g} - {'Non-normal' if p<0.05 else 'Normal'}")

# 6. Grouped statistics
print("\nStats by Gender:")
grouped_stats = df.groupby('Gender')[num_cols].agg(['mean','std','median']).round(2)
grouped_stats.index = range(1, len(grouped_stats) + 1)
print(grouped_stats)

# 7. Pivot for spending by group/income (handle nulls with .fillna('-'))
pivot = df.pivot_table(index='Age_Group', columns='Income_Bracket',
                       values='Spending Score (1-100)', aggfunc='mean', observed=False)
pivot_display = pivot.round(1).fillna('-')
pivot_display.index = range(1, len(pivot_display) + 1)
print("\nMean Spending Score by Age Group & Income:\n", pivot_display)
plt.figure(figsize=(6,4))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap='YlGnBu')
plt.title('Mean Spending Score: Age Group vs Income')
plt.show()

# 8. Top 10 by spending score (unique IDs)
top10_spending = df.nlargest(10, 'Spending Score (1-100)').drop_duplicates('CustomerID')
top10_spending_display = top10_spending[['CustomerID','Gender','Age','Annual Income (k$)','Spending Score (1-100)']].copy()
top10_spending_display.index = range(1, len(top10_spending_display) + 1)
print("\nTop 10 by Spending Score:\n", top10_spending_display)

# 9. Top 10 by annual income (unique IDs)
top10_income = df.nlargest(10, 'Annual Income (k$)').drop_duplicates('CustomerID')
top10_income_display = top10_income[['CustomerID','Gender','Age','Annual Income (k$)','Spending Score (1-100)']].copy()
top10_income_display.index = range(1, len(top10_income_display) + 1)
print("\nTop 10 by Annual Income:\n", top10_income_display)

# 10. KMeans clustering on scaled features
features = ['Age','Annual Income (k$)','Spending Score (1-100)']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method for k
inertia = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.plot(range(2,7), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

# Use k=4 for clustering (from visual elbow)
kmeans = KMeans(n_clusters=4, random_state=1)
df['Cluster'] = kmeans.fit_predict(X_scaled)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centers (Age, Income, Spending):\n", np.round(centers, 1))
cluster_summary = df.groupby('Cluster')[features].mean().round(2)
cluster_summary.index = range(1, len(cluster_summary) + 1)
print("\nCluster Summary:\n", cluster_summary)

plt.figure(figsize=(7,5))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='Set2', data=df, s=60)
plt.scatter(centers[:,1], centers[:,2], c='black', marker='X', s=200, label='Centers')
plt.title('K-Means Customer Segments')
plt.legend()
plt.show()

# 11. Percent of high/low spenders in each cluster
median_spend = df['Spending Score (1-100)'].median()
df['Spending_Category'] = np.where(df['Spending Score (1-100)'] >= median_spend, 'High', 'Low')
cluster_spend = df.groupby(['Cluster', 'Spending_Category']).size().unstack(fill_value=0)
cluster_spend_pct = cluster_spend.div(cluster_spend.sum(axis=1), axis=0) * 100
cluster_spend_pct_display = cluster_spend_pct.round(1)
cluster_spend_pct_display.index = range(1, len(cluster_spend_pct_display) + 1)
print("\nPercentage of High and Low Spenders by Cluster:\n", cluster_spend_pct_display)

# 12. Save final result
df.to_csv("mall_customers_with_clusters.csv", index=False)
print("\nResults saved to mall_customers_with_clusters.csv")
