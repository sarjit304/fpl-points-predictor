# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (.venv)
#     language: python
#     name: venv
# ---

# %% [markdown]
# # FPL Points Predictor - EDA

# %% [markdown]
# # Install necessary packages

# %%
# !pip install pandas matplotlib

# %% [markdown]
# # Import necessary packages

# %%
import math
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

# %% [markdown]
# # Read the CSV file

# %%
df = pd.read_csv('data/processed/cleaned_all_seasons.csv')

# %% [markdown]
# # Display first 5 rows

# %%
df.head()

# %% [markdown]
# # Dataset shape

# %%
df.shape

# %% [markdown]
# # Dataset Info

# %%
df.info()

# %% [markdown]
# # Dataset missing values

# %%
df.isnull().sum()

# %%



# %%



# %%

# %% [markdown]
# # Numeric summary

# %%
df.describe()

# %% [markdown]
# # Categorical summary

# %%
df.describe(include='object')

# %% [markdown]
# # Correlation Analysis

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# %% [markdown]
# # Performance by Position/Team

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.boxplot(data=df, x='position', y='ict_index', ax=axes[0,0])
axes[0,0].set_title('ICT Index by Position')

sns.boxplot(data=df, x='team', y='influence', ax=axes[0,1])
axes[0,1].set_title('Influence by Team')
axes[0,1].tick_params(axis='x', rotation=90)


sns.boxplot(data=df, x='was_home', y='threat', ax=axes[1,0])
axes[1,0].set_title('Threat: Home vs Away')

df.groupby('round')['ict_index'].mean().plot(ax=axes[1,1], marker='o')
axes[1,1].set_title('Avg ICT Index by Gameweek')
plt.tight_layout()
plt.show()


# %% [markdown]
# # Outliers and Summary Statistics

# %%
def find_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5*IQR) | (df[column] > Q3 + 1.5*IQR)]
    return len(outliers)

print("Outliers count:")
for var in ['ict_index', 'influence', 'threat']:
    print(f"{var}: {find_outliers(df, var)}")

summary = pd.DataFrame({
    'n_rows': len(df),
    'n_players': df['name'].nunique(),
    'mean_ict': df['ict_index'].mean(),
    'ict_std': df['ict_index'].std(),
    'missing_pct': df.isnull().sum().mean() / len(df) * 100
}, index=[0])

print("\nEDA Summary:\n", summary)

# %% [markdown]
# # Most Goals by Team

# %%
team_goals = df.groupby('team')['goals_scored'].sum().sort_values(ascending=False).head(10)
print("Top 10 Teams by Goals Scored:\n", team_goals)

plt.figure(figsize=(10, 6))
team_goals.plot(kind='bar')
plt.title('Top 10 Teams: Most Goals Scored')
plt.xlabel('Team')
plt.ylabel('Total Goals')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Most Goals by Player

# %%
player_goals = df.groupby('name')['goals_scored'].sum().sort_values(ascending=False).head(10)
print("Top 10 Players by Goals:\n", player_goals)

plt.figure(figsize=(10, 6))
player_goals.plot(kind='bar')
plt.title('Top 10 Players: Most Goals Scored')
plt.xlabel('Player')
plt.ylabel('Total Goals')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Most Assists by Player

# %%
player_assists = df.groupby('name')['assists'].sum().sort_values(ascending=False).head(10)
print("Top 10 Players by Assists:\n", player_assists)

plt.figure(figsize=(10, 6))
player_assists.plot(kind='bar')
plt.title('Top 10 Players: Most Assists')
plt.xlabel('Player')
plt.ylabel('Total Assists')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Most FPL Points by Player

# %%
player_points = df.groupby('name')['total_points'].sum().sort_values(ascending=False).head(10)
print("Top 10 Players by Total FPL Points:\n", player_points)

plt.figure(figsize=(10, 6))
player_points.plot(kind='bar', color='green')
plt.title('Top 10 Players: Most FPL Points')
plt.xlabel('Player')
plt.ylabel('Total Points')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Goals vs FPL Points Scatter

# %%
top_players = df.nlargest(50, 'goals_scored')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=top_players, x='goals_scored', y='total_points', hue='position', s=100)
plt.title('Goals vs Total FPL Points (Top 50 Goalscorers)')
plt.xlabel('Goals Scored')
plt.ylabel('Total FPL Points')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# # Home vs Away Performance

# %%
home_away = df.groupby('was_home')[['ict_index', 'influence', 'threat']].mean()
print("Avg Performance Home vs Away:\n", home_away)

plt.figure(figsize=(8, 5))
home_away['ict_index'].plot(kind='bar')
plt.title('Avg ICT Index: Home vs Away')
plt.xlabel('Home Game')
plt.ylabel('Avg ICT Index')
plt.xticks([0,1], ['Away', 'Home'], rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Summary Statistics Table

# %%
summary_stats = df[['influence', 'threat', 'ict_index', 'minutes']].agg(['mean', 'std', 'min', 'max'])
print("Performance Summary Statistics:\n")
print(summary_stats.round(2))

top_metrics = pd.DataFrame({
    'Best Gameweek': [df.loc[df['ict_index'].idxmax(), 'round']],
    'Max ICT': [df['ict_index'].max()],
    'Max Influence': [df['influence'].max()],
    'Max Threat': [df['threat'].max()],
    'Avg Minutes': [df['minutes'].mean()]
})

print("\nKey Achievements:\n", top_metrics.round(2))

comparison_data = pd.DataFrame({
    'Metric': ['Influence', 'Threat', 'ICT Index', 'Minutes'],
    'Mean': summary_stats.loc['mean', :],
    'Max': summary_stats.loc['max', :]
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

comparison_data.set_index('Metric')[['Mean', 'Max']].plot(kind='bar', ax=ax1, width=0.6)
ax1.set_title('Mean vs Maximum Performance')
ax1.set_ylabel('Value')
ax1.legend(title='Statistic')
ax1.tick_params(axis='x', rotation=0)

categories = list(comparison_data['Metric'])
N = len(categories)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

ax2 = plt.subplot(122, projection='polar')
mean_values = list(comparison_data['Mean']) + list(comparison_data['Mean'])[:1]
max_values = list(comparison_data['Max']) + list(comparison_data['Max'])[:1]

ax2.plot(angles, mean_values, 'o-', linewidth=2, label='Mean')
ax2.fill(angles, mean_values, alpha=0.25)
ax2.plot(angles, max_values, 'o-', linewidth=2, label='Max')
ax2.fill(angles, max_values, alpha=0.25)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories)
ax2.set_ylim(0)
ax2.set_title('Performance Comparison (Radar)')
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.show()

# %%
