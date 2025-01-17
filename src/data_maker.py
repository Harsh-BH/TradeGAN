import pandas as pd
import os
import yfinance as yf

def fetch_data_for_both(input_csv, output_folder="data",
                        start_date="2010-01-01", end_date="2021-12-31",
                        exchange_suffix=".NS"):
    """
    Fetch data from yfinance for both 'ticker_x' and 'ticker_y' on each row.
    The output CSV filenames will exclude the '.NS' suffix.
    """

    # 1. Read the master CSV
    try:
        master_df = pd.read_csv(input_csv)
        print(f"Successfully read {input_csv}")
    except FileNotFoundError:
        print(f"Input CSV file '{input_csv}' not found.")
        return
    except Exception as e:
        print(f"Error reading {input_csv}: {e}")
        return

    # 2. Create the output folder if not present
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder '{output_folder}' is ready.")

    # 3. Iterate over each row in the CSV
    for idx, row in master_df.iterrows():
        # Extract the tickers from the row
        ticker_x = row['ticker_x']  # e.g., "RELIANCE"
        ticker_y = row['ticker_y']  # e.g., "NIFTY_EN"

        # Optionally append a suffix for NSE if needed
        yf_ticker_x = ticker_x + exchange_suffix

        print(f"\nFetching data for X={yf_ticker_x} and Y={ticker_y}...")

        for current_ticker in [yf_ticker_x, ticker_y]:
            try:
                # Fetch data from yfinance
                df_yf = yf.download(current_ticker, start=start_date, end=end_date, progress=False)

                # Check if data is empty
                if df_yf.empty:
                    print(f"No data for {current_ticker}. Skipping...")
                    continue

                # Keep only required columns and rename them
                df_yf = df_yf[['Open', 'Close']].copy()
                df_yf.rename(columns={'Open': 'AdjOpen', 'Close': 'AdjClose'}, inplace=True)

                # Reset index and rename the date column
                df_yf.reset_index(inplace=True)
                if 'Date' in df_yf.columns:
                    df_yf.rename(columns={'Date': 'date'}, inplace=True)

                # Remove the exchange suffix if present
                if current_ticker.endswith(exchange_suffix):
                    base_ticker = current_ticker[:-len(exchange_suffix)]
                else:
                    base_ticker = current_ticker
                # Build file name
                out_path = os.path.join(output_folder, f"{base_ticker}.csv")

                # Save to CSV
                df_yf.to_csv(out_path, index=False)
                print(f"Saved {out_path}")

            except Exception as e:
                print(f"Error fetching data for {current_ticker}: {e}")

    print("\nAll possible tickers processed.")

# Example usage
if __name__ == "__main__":
    fetch_data_for_both(
        input_csv="stocks-etfs-list.csv",
        output_folder="data",
        start_date="2010-01-01",
        end_date="2021-12-31",
        exchange_suffix=".NS"
    )
