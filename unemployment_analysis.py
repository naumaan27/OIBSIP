# -----------------------
# Unemploymnet Analysis Script
# -----------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from datetime import datetime

file2 = pd.read_csv(r"C:\Users\nauma\Downloads\dataset\Unemployment_Rate_upto_11_2020.csv") 
file1 = pd.read_csv(r"C:\Users\nauma\Downloads\dataset\Unemployment in India.csv")  
print(file1.head())
print(file2.head())
print(file1.info())
print(file2.info())

merged_data = pd.merge(file1, file2[['Region', 'Date', 'longitude', 'latitude']],
                       on=['Region', 'Date'], how='inner')

merged_data['Date'] = pd.to_datetime(merged_data['Date'], errors='coerce', dayfirst=True)

merged_data.dropna(inplace=True)

merged_data.reset_index(drop=True, inplace=True)

plt.figure(figsize=(15, 8))
for region in merged_data['Region'].unique():
    region_data = merged_data[merged_data['Region'] == region]
    plt.plot(region_data['Date'], region_data['Estimated Unemployment Rate (%)'], label=region)

plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.title('Unemployment Rate Trends by Region')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(data=merged_data, x='Area', y='Estimated Unemployment Rate (%)', errorbar=None)
plt.title('Average Unemployment Rate by Area')
plt.ylabel('Unemployment Rate (%)')
plt.xlabel('Area')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(merged_data[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
m = folium.Map(location=[merged_data['latitude'].mean(), merged_data['longitude'].mean()], zoom_start=5)

marker_cluster = MarkerCluster().add_to(m)

for idx, row in merged_data.iterrows():
    popup_text = f"Region: {row['Region']}<br>Date: {row['Date'].date()}<br>Unemployment Rate: {row['Estimated Unemployment Rate (%)']}%"
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup_text
    ).add_to(marker_cluster)

m.save('unemployment_map.html')

print("Interactive unemployment map saved as 'unemployment_map.html'")

overall_trend = merged_data.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()

plt.figure(figsize=(15, 6))
plt.plot(overall_trend['Date'], overall_trend['Estimated Unemployment Rate (%)'], color='darkblue')
plt.title('Overall Unemployment Rate Trend')
plt.xlabel('Date')
plt.ylabel('Average Unemployment Rate (%)')
plt.grid()
plt.show()
