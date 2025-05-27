import pandas as pd
import numpy as np
import os

# Change working directory to the script's parent directory (project root)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import datasets
print("Importing datasets...")
crsp_q_ccm = pd.read_csv('./data/crsp_q_ccm_1.csv')
wrdsapps_finratio = pd.read_csv('./data/wrdsapps_finratio.csv')

# Convert date columns to datetime
crsp_q_ccm['datadate'] = pd.to_datetime(crsp_q_ccm['datadate'], errors='coerce')
wrdsapps_finratio['public_date'] = pd.to_datetime(wrdsapps_finratio['public_date'], errors='coerce')

# Convert to year-month format
crsp_q_ccm['year_month'] = crsp_q_ccm['datadate'].dt.to_period('M')
wrdsapps_finratio['year_month'] = wrdsapps_finratio['public_date'].dt.to_period('M')

# Perform the inner join
print("Merging datasets...")
full_data = crsp_q_ccm.merge(
    wrdsapps_finratio,
    left_on=['GVKEY', 'year_month'],
    right_on=['gvkey', 'year_month'],
    how='inner'
)

# Drop redundant columns
full_data.drop(columns=['gvkey', 'datadate', 'public_date'], inplace=True, errors='ignore')

# Reorder columns
cols = ['GVKEY', 'year_month'] + [col for col in full_data.columns if col not in ['GVKEY', 'year_month']]
full_data = full_data[cols]

# Sort by GVKEY and year_month
full_data.sort_values(by=['GVKEY', 'year_month'], inplace=True)

# Find GVKEYs with the most entries
print("Finding GVKEYs with most entries...")
gvkey_counts = full_data.groupby("GVKEY")["year_month"].nunique()
max_entries = gvkey_counts.max()
gvkeys_with_max_entries = gvkey_counts[gvkey_counts == max_entries].index.tolist()
print(f"GVKEYs with most entries ({max_entries}): {gvkeys_with_max_entries}")

# Filter data to keep only GVKEYs with max entries
data_all_dates_trimmed = full_data[full_data["GVKEY"].isin(gvkeys_with_max_entries)]
data_all_dates_trimmed.reset_index(drop=True, inplace=True)

# Define the features we want to keep
features = [
    'trt1m',  # Target variable

    # 1. Market-Related Factors (Macroeconomic & Market-wide)
    'divyield',  # Dividend Yield
    'bm',  # Book-to-Market Ratio
    'pe_exi', 'pe_inc',  # Price-to-Earnings Ratios
    'evm',  # Enterprise Value Multiple
    'de_ratio', 'debt_capital',  # Debt/Market Cap Ratios
    'ps',  # Price-to-Sales
    'ptb',  # Price-to-Book

    # 2. Profitability & Growth Factors
    'roe', 'roa', 'roce',  # Return on Equity, Assets, Capital Employed
    'gpm', 'npm', 'opmad', 'opmbd',  # Profit Margins (Gross, Net, Operating)
    'rd_sale',  # R&D to Sales
    'adv_sale',  # Advertising Expense to Sales
    'staff_sale',  # Labour Expense to Sales

    # 3. Risk & Leverage Factors
    'dltt_be',  # Long-term Debt/Book Equity 
    'debt_assets',  # Total Debt/Total Assets
    'debt_ebitda',  # Debt/EBITDA
    'intcov', 'intcov_ratio',  # Interest Coverage Ratios
    'ocf_lct',  # Operating CF/Current Liabilities
    'cash_debt',  # Cash Flow/Total Debt

    # 4. Liquidity & Efficiency Factors
    'at_turn',  # Asset Turnover
    'inv_turn',  # Inventory Turnover
    'rect_turn',  # Receivables Turnover
    'pay_turn',  # Payables Turnover
    'curr_ratio', 'quick_ratio', 'cash_ratio',  # Liquidity Ratios

    # 5. Size & Trading Activity
    'cshoq', 'cshom',  # Common Shares Outstanding
    'prccm',  # Market Price per Share (used for Market Cap calculation)
    'cshtrm',  # Trading Volume
    
    # 6. Sector Info
    'gsector' # GICS Sector code
]

# Keep only the desired columns
data_all_dates_trimmed = data_all_dates_trimmed[['GVKEY', 'year_month'] + features]

# Identify companies with complete data (excluding cshoq)
print("Finding companies with complete data...")
cols_to_check = [col for col in data_all_dates_trimmed.columns if col not in ['GVKEY', 'year_month', 'cshoq']]
complete_gvkeys = data_all_dates_trimmed.groupby('GVKEY', group_keys=False).filter(
    lambda group: (group[cols_to_check].isna().mean() <= 0).all()
)['GVKEY'].unique()

print(f"Number of companies with complete data: {len(complete_gvkeys)}")

# Create final_data with only companies that have complete data
final_data = data_all_dates_trimmed[data_all_dates_trimmed['GVKEY'].isin(complete_gvkeys)].copy()
final_data.reset_index(drop=True, inplace=True)

# Ensure data is sorted by date within each company
final_data.sort_values(['GVKEY', 'year_month'], inplace=True)

# Interpolate missing cshoq values within each company group
final_data['cshoq'] = final_data.groupby('GVKEY')['cshoq'].transform(lambda x: x.interpolate(method='linear'))

# Fill remaining NaNs using backward and forward fill
final_data['cshoq'] = final_data.groupby('GVKEY')['cshoq'].transform(lambda x: x.bfill().ffill())

# Convert year_month from Period to string for CSV output
final_data['year_month'] = final_data['year_month'].astype(str)

# Save to CSV
output_file = './data/final_data.csv'
print(f"Saving final data to {output_file}...")
final_data.to_csv(output_file, index=False)

print(f"Final data saved successfully with {final_data.shape[0]} rows and {final_data.shape[1]} columns.")
print(f"Sample of final data:\n{final_data.head()}")
