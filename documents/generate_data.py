from sec_edgar_downloader import Downloader
import pandas as pd
from pandas import DataFrame

dl = Downloader("HaojunMah", "mahhaojun03@gmail.com")
COMPANY_TICKER = "MSFT"

# Get 10-K filings
dl.get("10-K", COMPANY_TICKER, limit=1)

# Get 10-Q filings
dl.get("10-Q", COMPANY_TICKER, limit=4)

# Get 8-K filings
dl.get("8-K", COMPANY_TICKER, limit=1)

# Get Proxy Statements
dl.get("DEF 14A", COMPANY_TICKER, limit=1)

revenue_data = {
    'year': [2023, 2023, 2023, 2023, 2022, 2022, 2022, 2022],
    'quarter': ['Q4', 'Q3', 'Q2', 'Q1', 'Q4', 'Q3', 'Q2', 'Q1'],
    'revenue_usd_billions': [61.9, 56.5, 52.9, 52.7, 51.9, 50.1, 49.4, 51.7],
    'net_income_usd_billions': [21.9, 22.3, 17.4, 16.4, 17.6, 16.7, 16.7, 18.8]
}

# Create DataFrame from dictionary
df = pd.DataFrame(revenue_data)
# Save DataFrame to CSV file
CSV_PATH = "sec-edgar-filings/revenue_summary.csv"
df.to_csv(CSV_PATH, index=False)
