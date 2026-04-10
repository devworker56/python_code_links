"""
Multi-Symbol True Confluence Portfolio Simulator
Tracks a combined P&L across multiple tickers using Statistical Z-Scores.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================================
# 1. MASTER CONFIGURATION
# ============================================================================
SYMBOLS = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'ORCL']
START_DATE = '2014-01-01'
END_DATE = '2018-01-01'

# PORTFOLIO SETTINGS
STARTING_CASH = 100000 
SHARES_PER_TRADE = 250   # Number of shares to trade per signal per stock
Z_THRESHOLD = 3      # The "Confluence" trigger
LOOKBACK = 20           # Window for statistical normalization

# ============================================================================
# 2. THE CONFLUENCE ENGINE
# ============================================================================
def get_confluence_indicators(symbol):
    print(f"Processing {symbol}...")
    df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 4-Factor Model
    df['M'] = df['Close'].pct_change().rolling(3).sum()
    sigma = (df['High'] - df['Low']) / df['Close']
    df['C'] = sigma.shift(1) / sigma
    df['S'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
    df['V'] = np.exp(-abs((df['Volume'] / df['Volume'].shift(1)) - 1))
    
    # PCI & Z-Score
    df['PCI'] = df['M'] * df['C'] * df['S'] * df['V']
    df['Z_Score'] = (df['PCI'] - df['PCI'].rolling(LOOKBACK).mean()) / df['PCI'].rolling(LOOKBACK).std()
    
    return df[['Close', 'PCI', 'Z_Score']].dropna()

# Pre-calculate data for all symbols
portfolio_data = {s: get_confluence_indicators(s) for s in SYMBOLS}

# Align all data to a common date index
all_dates = sorted(set().union(*(df.index for df in portfolio_data.values())))

# ============================================================================
# 3. MULTI-SYMBOL SIMULATION LOOP
# ============================================================================
print("\nStarting Multi-Symbol Simulation...")

cash = STARTING_CASH
positions = {s: 0 for s in SYMBOLS} # Current shares held for each ticker
portfolio_history = []
trade_count = 0

for date in all_dates:
    current_portfolio_value = cash
    
    for s in SYMBOLS:
        if date not in portfolio_data[s].index:
            continue
            
        # Get current state for this ticker
        price = portfolio_data[s].loc[date, 'Close']
        z = portfolio_data[s].loc[date, 'Z_Score']
        pci = portfolio_data[s].loc[date, 'PCI']
        current_shares = positions[s]
        
        # LOGIC: SELL/SHORT (Peak Confluence)
        if z > Z_THRESHOLD and current_shares >= 0:
            order = SHARES_PER_TRADE if current_shares == 0 else SHARES_PER_TRADE * 2
            cash += (order * price)
            positions[s] -= order
            trade_count += 1
            
        # LOGIC: BUY/COVER (Trough Confluence)
        elif z < -Z_THRESHOLD and current_shares <= 0:
            order = SHARES_PER_TRADE if current_shares == 0 else SHARES_PER_TRADE * 2
            cash -= (order * price)
            positions[s] += order
            trade_count += 1
            
        # LOGIC: NEUTRAL EXIT (Confluence Dissipated)
        elif (current_shares < 0 and pci <= 0) or (current_shares > 0 and pci >= 0):
            if current_shares != 0:
                cash += (current_shares * price)
                positions[s] = 0
                trade_count += 1

        # Add value of current stock holding to portfolio value
        current_portfolio_value += (positions[s] * price)
    
    portfolio_history.append({'Date': date, 'Total_Value': current_portfolio_value})

# ============================================================================
# 4. FINAL RESULTS & VISUALIZATION
# ============================================================================
results_df = pd.DataFrame(portfolio_history).set_index('Date')
final_pnl = results_df['Total_Value'].iloc[-1] - STARTING_CASH

print(f"\n{'='*40}")
print(f"MASTER PORTFOLIO SUMMARY")
print(f"{'='*40}")
print(f"Tickers:      {', '.join(SYMBOLS)}")
print(f"Total Trades: {trade_count}")
print(f"Final P&L:    ${final_pnl:,.2f}")
print(f"Return:       {(final_pnl/STARTING_CASH)*100:.2f}%")
print(f"{'='*40}")

# Plotting the Master Growth Curve
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(14, 7))
plt.plot(results_df.index, results_df['Total_Value'], color='blue', linewidth=2)
plt.fill_between(results_df.index, STARTING_CASH, results_df['Total_Value'], 
                 where=(results_df['Total_Value'] >= STARTING_CASH), color='green', alpha=0.2)
plt.fill_between(results_df.index, STARTING_CASH, results_df['Total_Value'], 
                 where=(results_df['Total_Value'] < STARTING_CASH), color='red', alpha=0.2)
plt.axhline(STARTING_CASH, color='black', linestyle='--', alpha=0.5)
plt.title(f"Master Portfolio Growth (True Confluence Model)")
plt.ylabel("Portfolio Value ($)")
plt.show()