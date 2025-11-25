import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
ASSETS = {
    "AAPL": "AAPL.csv",
    "MSFT": "MSFT.csv",
    "SPY": "SPY.csv",
    "QQQ": "QQQ.csv",
    "TLT": "TLT.csv",
    "GLD": "GLD.csv"
}

os.makedirs("plots", exist_ok=True)


# ------------------------------------------------------
# FUNCTION: Clean Investing.com CSV
# ------------------------------------------------------
def load_and_clean(path):
    df = pd.read_csv(path)

    # --- Date ---
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

    # --- Prices ---
    price_cols = ["Price", "Open", "High", "Low"]
    for col in price_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Volume ---
    vol = df["Vol."].astype(str)
    unit = vol.str.extract(r"([MB])")[0]  # extract M or B
    number = vol.str.replace(r"[MB]", "", regex=True)

    number = (
        number.str.replace(".", "", regex=False)
              .str.replace(",", ".", regex=False)
    )
    number = pd.to_numeric(number, errors="coerce")

    factor = unit.map({"M": 1e6, "B": 1e9})
    df["Volume"] = number * factor

    # --- Daily Change % ---
    df["Return"] = (
        df["Change %"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Return"] = pd.to_numeric(df["Return"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ------------------------------------------------------
# FUNCTION: Plot for a single asset
# ------------------------------------------------------
def generate_plots(name, df):
    # Closing price
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Price"])
    plt.title(f"{name} Closing Price (2015–2024)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_closing_price.png")
    plt.close()

    # Volume
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Volume"])
    plt.title(f"{name} Trading Volume (2015–2024)")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_volume.png")
    plt.close()

    # Daily return histogram
    plt.figure(figsize=(10, 5))
    plt.hist(df["Return"].dropna(), bins=50)
    plt.title(f"{name} Daily Return Distribution")
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_return_hist.png")
    plt.close()

    # 30-day Moving Average
    df["MA30"] = df["Price"].rolling(window=30).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Price"], label="Price")
    plt.plot(df["Date"], df["MA30"], label="MA30")
    plt.title(f"{name} Price + 30-Day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{name}_ma30.png")
    plt.close()


# ------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------
all_data = {}

for asset, file in ASSETS.items():
    df = load_and_clean(file)
    all_data[asset] = df
    generate_plots(asset, df)

print("Individual plots saved to /plots folder.")


# ------------------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------------------

# Create a unified price DataFrame indexed by Date
price_df = pd.DataFrame()

for asset, df in all_data.items():
    price_df[asset] = df.set_index("Date")["Price"]

# Keep only dates where all assets have price
price_df = price_df.dropna()

# Compute correlation
corr = price_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Asset Price Correlation Heatmap (2015–2024)")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

print("Correlation heatmap saved!")
