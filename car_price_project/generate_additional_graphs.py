# -*- coding: utf-8 -*-
"""
generate_additional_graphs.py
==============================
Generates 5 additional visualization graphs for the car price prediction project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "used_cars.csv"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("GENERATING 5 ADDITIONAL GRAPHS")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from used_cars.csv")

# Clean price column
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).replace("$", "").replace(",", "")
    try:
        return float(price_str)
    except:
        return np.nan

df['price_usd'] = df['price'].apply(clean_price)

# Clean mileage column
def clean_mileage(mile_str):
    if pd.isna(mile_str):
        return np.nan
    mile_str = str(mile_str).replace(",", "").replace(" mi.", "").replace("mi", "").strip()
    try:
        return float(mile_str)
    except:
        return np.nan

df['mileage'] = df['milage'].apply(clean_mileage)

# Remove rows with missing critical data
df = df.dropna(subset=['price_usd', 'model_year'])
df = df[df['price_usd'] > 0]

print(f"After cleaning: {len(df)} rows")

# Set modern style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ================================================================
# GRAPH 1: Top 15 Brands by Count - Enhanced Bar Chart
# ================================================================
print("\n1. Generating top_brands_count.png...")
top_brands = df['brand'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(top_brands)))
bars = ax.barh(range(len(top_brands)), top_brands.values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, top_brands.values)):
    ax.text(value + 5, bar.get_y() + bar.get_height()/2, f'{int(value)}', 
            va='center', fontsize=11, fontweight='bold')

ax.set_yticks(range(len(top_brands)))
ax.set_yticklabels(top_brands.index, fontsize=11)
ax.set_xlabel('Number of Cars', fontsize=13, fontweight='bold')
ax.set_ylabel('Brand', fontsize=13, fontweight='bold')
ax.set_title('Top 15 Car Brands in Dataset', fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "top_brands_count.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved top_brands_count.png")

# ================================================================
# GRAPH 2: Price Distribution by Transmission Type - Box Plot
# ================================================================
print("2. Generating price_by_transmission.png...")
df_trans = df[df['transmission'].notna() & (df['price_usd'] < 100000)].copy()
# Simplify transmission names
df_trans['trans_clean'] = df_trans['transmission'].str.replace('A/T', 'Automatic').str.replace('M/T', 'Manual')
trans_order = df_trans.groupby('trans_clean')['price_usd'].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
box_plot = sns.boxplot(data=df_trans, x='trans_clean', y='price_usd', 
                       order=trans_order, palette='Set2', ax=ax, linewidth=2)

# Customize box plot
for patch in box_plot.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.7))
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

ax.set_xlabel('Transmission Type', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax.set_title('Price Distribution by Transmission Type', fontsize=16, fontweight='bold', pad=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x/1000)}K'))
ax.tick_params(axis='both', labelsize=11)
plt.xticks(rotation=30, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(ASSETS_DIR / "price_by_transmission.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved price_by_transmission.png")

# ================================================================
# GRAPH 3: Average Price by Year - Enhanced Line Chart
# ================================================================
print("3. Generating avg_price_by_year.png...")
year_price = df.groupby('model_year')['price_usd'].mean().sort_index()
year_price = year_price[year_price.index >= 2000]

fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
ax.plot(year_price.index, year_price.values, marker='o', linewidth=3, 
        markersize=8, color='#E63946', markeredgecolor='white', 
        markeredgewidth=2, label='Average Price', zorder=3)
ax.fill_between(year_price.index, year_price.values, alpha=0.2, color='#E63946')

# Add trend line
z = np.polyfit(year_price.index, year_price.values, 2)
p = np.poly1d(z)
ax.plot(year_price.index, p(year_price.index), "--", alpha=0.8, 
        linewidth=2, color='#457B9D', label='Trend')

ax.set_xlabel('Model Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Price (USD)', fontsize=13, fontweight='bold')
ax.set_title('Average Car Price Trend Over Years (2000-2024)', fontsize=16, fontweight='bold', pad=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x/1000)}K'))
ax.tick_params(axis='both', labelsize=11)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(ASSETS_DIR / "avg_price_by_year.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved avg_price_by_year.png")

# ================================================================
# GRAPH 4: Mileage vs Price - Enhanced Scatter with Density
# ================================================================
print("4. Generating mileage_vs_price_scatter.png...")
df_scatter = df[(df['mileage'].notna()) & (df['price_usd'] < 80000) & (df['mileage'] < 250000)].copy()

fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')

# Create scatter plot with gradient colors
scatter = ax.scatter(df_scatter['mileage'], df_scatter['price_usd'], 
                    alpha=0.6, s=50, c=df_scatter['model_year'], 
                    cmap='viridis', edgecolors='black', linewidth=0.5)

# Add trendline
z = np.polyfit(df_scatter['mileage'], df_scatter['price_usd'], 1)
p = np.poly1d(z)
trend_x = np.linspace(df_scatter['mileage'].min(), df_scatter['mileage'].max(), 100)
ax.plot(trend_x, p(trend_x), 
        "r-", alpha=0.8, linewidth=3, label=f'Trend Line', zorder=2)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Model Year', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

ax.set_xlabel('Mileage (miles)', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax.set_title('Car Price vs Mileage (Colored by Model Year)', fontsize=16, fontweight='bold', pad=20)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x/1000)}K'))
ax.tick_params(axis='both', labelsize=11)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(ASSETS_DIR / "mileage_vs_price_scatter.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved mileage_vs_price_scatter.png")

# ================================================================
# GRAPH 5: Fuel Type Distribution - Modern Pie Chart
# ================================================================
print("5. Generating fuel_type_distribution.png...")
fuel_counts = df['fuel_type'].fillna('Unknown').value_counts().head(6)

fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
colors_fuel = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
explode = [0.05 if i == 0 else 0 for i in range(len(fuel_counts))]

wedges, texts, autotexts = ax.pie(fuel_counts, 
                                    labels=fuel_counts.index, 
                                    autopct='%1.1f%%',
                                    startangle=90, 
                                    colors=colors_fuel, 
                                    explode=explode,
                                    shadow=True,
                                    textprops={'fontsize': 12, 'weight': 'bold'},
                                    wedgeprops={'edgecolor': 'white', 'linewidth': 2})

# Enhance percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

# Add count labels
for i, (label, count) in enumerate(zip(fuel_counts.index, fuel_counts.values)):
    angle = (wedges[i].theta2 - wedges[i].theta1) / 2. + wedges[i].theta1
    x = 1.25 * np.cos(np.radians(angle))
    y = 1.25 * np.sin(np.radians(angle))
    ax.text(x, y, f'{int(count)} cars', ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
            edgecolor='gray', alpha=0.8))

ax.set_title('Fuel Type Distribution in Dataset', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "fuel_type_distribution.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved fuel_type_distribution.png")

print("\n" + "=" * 60)
print("SUCCESSFULLY GENERATED 5 NEW GRAPHS")
print("=" * 60)
print(f"\nAll graphs saved to: {ASSETS_DIR}")
print("\nGenerated files:")
print("  1. top_brands_count.png")
print("  2. price_by_transmission.png")
print("  3. avg_price_by_year.png")
print("  4. mileage_vs_price_scatter.png")
print("  5. fuel_type_distribution.png")
