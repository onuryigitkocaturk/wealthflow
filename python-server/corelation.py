import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ASSETS = {
    "AAPL": "AAPL.csv",
    "MSFT": "MSFT.csv",
    "SPY":  "SPY.csv",
    "QQQ":  "QQQ.csv",
    "TLT":  "TLT.csv",
    "GLD":  "GLD.csv"
}

def load_returns(path):
    df = pd.read_csv(path)

    # Date
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

    # Change %
    df["Return"] = (
        df["Change %"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(".", "", regex=False)   # remove thousands separators
        .str.replace(",", ".", regex=False)  # convert comma decimal
    )

    df["Return"] = pd.to_numeric(df["Return"], errors="coerce") / 100.0

    return df[["Date", "Return"]].set_index("Date")

# --- LOAD ALL RETURNS ---
returns = pd.DataFrame()

for name, file in ASSETS.items():
    r = load_returns(file)
    returns[name] = r["Return"]

# keep only rows where all exist
returns = returns.dropna()

# --- CORRELATION ---
corr = returns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Return Correlation Heatmap (2015–2024)")
plt.tight_layout()
plt.show()
