"""
================================================================================
CODTECH INTERNSHIP - TASK 1
API Integration and Data Visualization
--------------------------------------------------------------------------------
Description : Fetches real-time weather data from Open-Meteo API (FREE, no key
              needed) for 5 major Indian cities and creates a multi-panel
              visualization dashboard saved as a PNG file.
Libraries   : requests, matplotlib, seaborn, pandas
================================================================================
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

# ── 1. CITY CONFIGURATION ─────────────────────────────────────────────────────
# Latitude and longitude of 5 Indian cities
CITIES = {
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Pune":      {"lat": 18.5204, "lon": 73.8567},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
}

# ── 2. DATA FETCHING ──────────────────────────────────────────────────────────
def fetch_weather(city_name, lat, lon):
    """
    Fetch 7-day hourly forecast from Open-Meteo API.
    Returns a DataFrame with temperature, humidity, wind speed, precipitation.
    Falls back to realistic simulated data if the API is unreachable.
    """
    import numpy as np

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,precipitation",
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()["hourly"]
        df = pd.DataFrame({
            "time":          pd.to_datetime(data["time"]),
            "temperature":   data["temperature_2m"],
            "humidity":      data["relativehumidity_2m"],
            "wind_speed":    data["windspeed_10m"],
            "precipitation": data["precipitation"],
        })
    except Exception:
        # ── Realistic fallback mock data (city-specific ranges) ──────────────
        print(f"  (API unavailable – using simulated data for {city_name})")
        base_temp = {"Mumbai": 33, "Delhi": 40, "Pune": 32, "Bangalore": 28, "Kolkata": 36}
        base_hum  = {"Mumbai": 82, "Delhi": 45, "Pune": 62, "Bangalore": 70, "Kolkata": 78}
        np.random.seed(abs(hash(city_name)) % 2**31)
        n = 7 * 24  # hourly points
        times = pd.date_range("2025-05-01", periods=n, freq="h")
        df = pd.DataFrame({
            "time":          times,
            "temperature":   np.clip(base_temp[city_name] + np.random.randn(n) * 3 +
                                     2 * np.sin(np.linspace(0, 4 * np.pi, n)), 18, 48),
            "humidity":      np.clip(base_hum[city_name]  + np.random.randn(n) * 8, 20, 100),
            "wind_speed":    np.clip(10 + np.random.randn(n) * 5, 0, 60),
            "precipitation": np.clip(np.random.exponential(0.3, n), 0, 20),
        })

    df["city"] = city_name
    return df


def fetch_all_cities():
    """Loop through all cities and combine into one DataFrame."""
    frames = []
    for city, coords in CITIES.items():
        print(f"  Fetching data for {city}...")
        df = fetch_weather(city, coords["lat"], coords["lon"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── 3. SUMMARY STATISTICS ────────────────────────────────────────────────────
def build_summary(df):
    """Compute daily averages per city."""
    df["date"] = df["time"].dt.date
    summary = (
        df.groupby(["city", "date"])
        .agg(
            avg_temp  = ("temperature",   "mean"),
            avg_hum   = ("humidity",      "mean"),
            avg_wind  = ("wind_speed",    "mean"),
            total_pcp = ("precipitation", "sum"),
        )
        .reset_index()
    )
    return summary


# ── 4. VISUALIZATION DASHBOARD ───────────────────────────────────────────────
def plot_dashboard(df, summary):
    """Create a 2×2 dashboard with 4 charts."""
    # Style
    sns.set_theme(style="darkgrid", palette="tab10")
    fig = plt.figure(figsize=(18, 12), facecolor="#f5f5f5")
    fig.suptitle(
        "🌤  7-Day Weather Dashboard — 5 Indian Cities\n"
        f"Data source: Open-Meteo API  |  Generated: {datetime.now().strftime('%d %b %Y %H:%M')}",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Chart 1: Temperature trend (line) ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for city in CITIES:
        city_df = summary[summary["city"] == city]
        ax1.plot(range(len(city_df)), city_df["avg_temp"], marker="o", label=city, linewidth=2)
    ax1.set_title("📈 Average Daily Temperature (°C)", fontweight="bold")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(fontsize=8)
    ax1.set_xticks(range(7))
    ax1.set_xticklabels([f"D{i+1}" for i in range(7)])

    # ── Chart 2: Average humidity bar chart ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    city_avg = summary.groupby("city")["avg_hum"].mean().sort_values(ascending=False)
    bars = ax2.bar(city_avg.index, city_avg.values, color=sns.color_palette("tab10", len(city_avg)))
    ax2.set_title("💧 Average Humidity (%) per City", fontweight="bold")
    ax2.set_ylabel("Humidity (%)")
    ax2.set_ylim(0, 100)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    # ── Chart 3: Wind speed heatmap ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    pivot = summary.pivot_table(values="avg_wind", index="city", columns="date", aggfunc="mean")
    pivot.columns = [f"D{i+1}" for i in range(len(pivot.columns))]
    sns.heatmap(pivot, ax=ax3, cmap="YlOrRd", annot=True, fmt=".1f",
                linewidths=0.5, cbar_kws={"label": "km/h"})
    ax3.set_title("🌬  Wind Speed Heatmap (km/h)", fontweight="bold")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("")

    # ── Chart 4: Total precipitation (stacked area) ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    pivot_pcp = summary.pivot_table(values="total_pcp", index="date", columns="city", aggfunc="sum").fillna(0)
    pivot_pcp.index = [f"D{i+1}" for i in range(len(pivot_pcp))]
    pivot_pcp.plot(kind="bar", ax=ax4, colormap="tab10", width=0.7)
    ax4.set_title("🌧  Daily Total Precipitation (mm)", fontweight="bold")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Precipitation (mm)")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.tick_params(axis="x", rotation=0)

    plt.savefig("task1_weather_dashboard.png", dpi=150, bbox_inches="tight")
    print("  ✅ Dashboard saved → task1_weather_dashboard.png")
    plt.close()


# ── 5. MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TASK 1 — API Integration & Data Visualization")
    print("=" * 60)

    print("\n[1/3] Fetching weather data from Open-Meteo API...")
    all_data = fetch_all_cities()
    print(f"  Total records fetched: {len(all_data)}")

    print("\n[2/3] Computing summary statistics...")
    summary = build_summary(all_data)
    print(summary.groupby("city")[["avg_temp", "avg_hum", "avg_wind"]].mean().round(2))

    print("\n[3/3] Generating visualization dashboard...")
    plot_dashboard(all_data, summary)

    print("\n✅ Task 1 Complete!")
