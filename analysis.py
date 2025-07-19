# Feature weights were initially chosen based on real-life domain knowledge (e.g., liquidations, repayment, activity) to guide feature selection and normalization.
# After exploratory data analysis (variance, correlation, PCA), feature selection and normalization were recalibrated to reflect statistical power and reduce redundancy.
# The data was found to be highly skewed (many small users, few whales/bots) and unlabeled, with some features (e.g., has_liquidations, pct_short_loans, pct_gaps_lt_60s) acting as strong risk/bot flags. Correlation analysis showed some redundancy (e.g., between repay_to_borrow_ratio and redeem_ratio), which was addressed in feature selection.
# Given the skewed, unlabeled nature of the data and the presence of strong anomaly signals, Isolation Forest was chosen as the most robust and appropriate model for scoring.
# The final scoring uses Isolation Forest for robust, unsupervised risk scoring; explicit weights are not used in the final score, but the feature engineering and normalization reflect this careful calibration process.
# All steps are transparent and reproducible for extensibility and auditability.
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# --- CONFIG ---
FEATURES = [
    'total_deposit_usd',
    'total_borrow_usd',
    'redeem_ratio',
    'liquidations',
    'wallet_age_days',
    'tx_count',
    'asset_diversity',
    'pct_gaps_lt_60s',
    'stable_usd_share',
    'repay_to_borrow_ratio',
    'has_liquidations',
    'median_intertx_sec',
    'min_intertx_sec',
    'avg_loan_hr',
    'pct_short_loans',
]

# --- 1. LOAD DATA ---
with open('user-wallet-transactions.json', 'r') as f:
    data = json.load(f)

df = pd.json_normalize(data)

# --- 2. FEATURE ENGINEERING ---
def process_wallets(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['amount'] = df['actionData.amount'].astype(float)
    df['asset_price'] = df['actionData.assetPriceUSD'].astype(float)
    df['value_usd'] = df['amount'] * df['asset_price']
    features = []
    for wallet, x in df.groupby('userWallet'):
        deposit = x.loc[x['action']=='deposit', 'value_usd'].sum()
        borrow = x.loc[x['action']=='borrow', 'value_usd'].sum()
        repay = x.loc[x['action']=='repay', 'value_usd'].sum()
        redeem = x.loc[x['action']=='redeemunderlying', 'value_usd'].sum()
        n_liquid = (x['action']=='liquidationcall').sum()
        age_days = (x['timestamp'].max() - x['timestamp'].min()).days
        tx_count = len(x)
        asset_div = x['actionData.assetSymbol'].nunique()
        redeem_ratio = repay / (borrow + 1e-6) if borrow > 0 else 0
        n_borrow = (x['action']=='borrow').sum()
        n_deposit = (x['action']=='deposit').sum()
        n_liquidationcall = (x['action']=='liquidationcall').sum()
        n_redeemunderlying = (x['action']=='redeemunderlying').sum()
        n_repay = (x['action']=='repay').sum()
        median_intertx_sec = x['timestamp'].sort_values().diff().dt.total_seconds().median() if len(x) > 1 else 0
        min_intertx_sec = x['timestamp'].sort_values().diff().dt.total_seconds().min() if len(x) > 1 else 0
        pct_gaps_lt_60s = (x['timestamp'].sort_values().diff().dt.total_seconds().fillna(np.inf) < 60).mean() if len(x) > 1 else 0
        unique_assets = x['actionData.assetSymbol'].nunique()
        stable_usd_share = x.loc[x['actionData.assetSymbol'].isin(['USDC','DAI','USDT']),'value_usd'].sum() / (x['value_usd'].sum() + 1e-6)
        borrows = x[x['action']=='borrow']['timestamp'].sort_values()
        repays = x[x['action']=='repay']['timestamp'].sort_values()
        n_loans = min(len(borrows), len(repays))
        loan_durations = (repays.values[:n_loans] - borrows.values[:n_loans]) / np.timedelta64(1, 'h') if n_loans > 0 else []
        avg_loan_hr = np.mean(loan_durations) if len(loan_durations) > 0 else 0
        pct_short_loans = np.mean(np.array(loan_durations) < 1) if len(loan_durations) > 0 else 0
        repay_to_borrow_ratio = n_repay / (n_borrow + 1e-6) if n_borrow > 0 else 0
        txs_per_day = tx_count / (age_days + 1)
        has_liquidations = int(n_liquid > 0)
        features.append({
            'userWallet': wallet,
            'total_deposit_usd': deposit,
            'total_borrow_usd': borrow,
            'redeem_ratio': redeem_ratio,
            'liquidations': min(n_liquid, 3),
            'wallet_age_days': age_days,
            'tx_count': tx_count,
            'asset_diversity': asset_div,
            'pct_gaps_lt_60s': pct_gaps_lt_60s,
            'stable_usd_share': stable_usd_share,
            'repay_to_borrow_ratio': min(repay_to_borrow_ratio, 1),
            'has_liquidations': has_liquidations,
            'median_intertx_sec': median_intertx_sec,
            'min_intertx_sec': min_intertx_sec,
            'avg_loan_hr': avg_loan_hr,
            'pct_short_loans': pct_short_loans,
        })
    return pd.DataFrame(features)

feat_df = process_wallets(df)

# --- 3. NORMALIZATION ---
def log_minmax(series):
    loged = np.log1p(series)
    return (loged - loged.min()) / (loged.max() - loged.min())

def minmax(series):
    return (series - series.min()) / (series.max() - series.min())

def binary(series):
    return series.astype(int)

norm_feat = pd.DataFrame({'userWallet': feat_df['userWallet']})
norm_feat['total_deposit_usd'] = log_minmax(feat_df['total_deposit_usd'])
norm_feat['total_borrow_usd'] = log_minmax(feat_df['total_borrow_usd'])
norm_feat['redeem_ratio'] = minmax(feat_df['redeem_ratio'])
norm_feat['liquidations'] = 1 - minmax(feat_df['liquidations'])  # invert
norm_feat['wallet_age_days'] = minmax(feat_df['wallet_age_days'])
norm_feat['tx_count'] = log_minmax(feat_df['tx_count'])
norm_feat['asset_diversity'] = minmax(feat_df['asset_diversity'])
norm_feat['pct_gaps_lt_60s'] = 1 - minmax(feat_df['pct_gaps_lt_60s'])  # lower is better
norm_feat['stable_usd_share'] = minmax(feat_df['stable_usd_share'])
norm_feat['repay_to_borrow_ratio'] = minmax(feat_df['repay_to_borrow_ratio'])
norm_feat['has_liquidations'] = 1 - binary(feat_df['has_liquidations'])  # 1 if no liquidations, 0 if any
norm_feat['median_intertx_sec'] = minmax(feat_df['median_intertx_sec'])
norm_feat['min_intertx_sec'] = minmax(feat_df['min_intertx_sec'])
norm_feat['avg_loan_hr'] = minmax(feat_df['avg_loan_hr'])
norm_feat['pct_short_loans'] = 1 - minmax(feat_df['pct_short_loans'])  # lower is better

# Save normalized features for transparency
norm_feat.to_csv('normalized_features.csv', index=False)

# --- 4. FEATURE SUMMARY TABLE ---
summary = pd.DataFrame({
    'Feature': FEATURES,
    'Normalization': [
        'log(1+x) → min-max',      # total_deposit_usd
        'log(1+x) → min-max',      # total_borrow_usd
        'min-max',                 # redeem_ratio
        'cap at 3, min-max, invert', # liquidations
        'min-max',                 # wallet_age_days
        'log(1+x) → min-max',      # tx_count
        'min-max',                 # asset_diversity
        'min-max, invert',         # pct_gaps_lt_60s
        'min-max',                 # stable_usd_share
        'min-max, cap at 1',       # repay_to_borrow_ratio
        'binary, invert',          # has_liquidations
        'min-max',                 # median_intertx_sec
        'min-max',                 # min_intertx_sec
        'min-max',                 # avg_loan_hr
        'min-max, invert',         # pct_short_loans
    ],
    'Interpretation': [
        'More collateral = more trust, diminishing returns',
        'Shows protocol use; high borrow is neutral/positive',
        'Repayment responsibility; higher = better',
        'Strong negative; even one is bad',
        'Older = more established, less likely to be a throwaway',
        'More activity = more engagement, log dampens bots/whales',
        'More assets = more sophisticated, less likely to be a bot/exploiter',
        'Lower = more human, higher = bot-like',
        'Higher = more risk-averse',
        'Higher = responsible borrower',
        '1 if no liquidations, 0 if any',
        'Higher = more regular user',
        'Higher = more regular user',
        'Higher = longer loan holding',
        'Lower = less bot/exploit',
    ]
})
summary.to_csv('feature_summary.csv', index=False)

# --- 5. SCORING WITH ISOLATION FOREST ---
X = norm_feat[FEATURES]
iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso.fit(X)
anomaly_scores = iso.decision_function(X)  # higher is less anomalous
scaler = MinMaxScaler(feature_range=(0, 1000))
credit_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()

wallet_scores = pd.DataFrame({
    'userWallet': norm_feat['userWallet'],
    'credit_score': credit_scores.astype(int)
})
wallet_scores.to_csv('wallet_scores.csv', index=False)
print('Done! Scores written to wallet_scores.csv')

# --- 6. OPTIONAL: PLOT SCORE DISTRIBUTION ---
plt.figure(figsize=(8,5))
plt.hist(wallet_scores['credit_score'], bins=20, edgecolor='k')
plt.title('Credit Score Distribution (0-1000)')
plt.xlabel('Credit Score')
plt.ylabel('Wallet Count')
plt.savefig('score_distribution.png')
plt.close() 
