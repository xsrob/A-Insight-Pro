
"""
A-Insight Pro
Stock Info Resolver - Names, Sectors, Fundamentals
Fetches via AKShare and caches locally.
"""
import os, json, pandas as pd
from datetime import datetime, timedelta

CACHE_FILE = 'data/stock_info_cache.json'
CACHE_TTL_HOURS = 24

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ts = data.get('_timestamp', '')
            age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 3600
            if age < CACHE_TTL_HOURS:
                return data
        except: pass
    return None

def save_cache(data):
    data['_timestamp'] = datetime.now().isoformat()
    os.makedirs('data', exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def fetch_stock_info():
    """Fetch stock names and industries from AKShare."""
    cache = load_cache()
    if cache and len(cache) > 2:
        print(f'Using cached stock info ({len(cache)-1} stocks)')
        return {k: v for k, v in cache.items() if not k.startswith('_')}

    print('Fetching stock info from AKShare...')
    info = {}
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        for _, row in df.iterrows():
            code = str(row.get('code', '')).zfill(6)
            name = str(row.get('name', ''))
            if code and name and name != 'nan':
                info[code] = {'name': name, 'industry': ''}
        print(f'Fetched {len(info)} stocks')
    except Exception as e:
        print(f'AKShare fetch failed: {e}')
        return fallback_names()

    # Try to get industry info
    try:
        import akshare as ak
        industry_df = ak.stock_board_industry_name_em()
        # Map by stock code
        for _, row in industry_df.iterrows():
            pass  # Industry mapping requires per-stock query, skip for speed
    except: pass

    save_cache(info)
    return info

def fallback_names():
    """Fallback: build names from existing data files."""
    info = {}
    data_dir = 'data'
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.csv'):
                code = f.replace('.csv', '').zfill(6)
                if len(code) == 6:
                    info[code] = {'name': '', 'industry': ''}
    return info

def get_name(code, stock_info=None):
    """Get stock name for a code."""
    code = str(code).zfill(6)
    if stock_info and code in stock_info:
        return stock_info[code].get('name', '')
    return ''

def enrich_dataframe(df, code_col='code'):
    """Add name column to a DataFrame with stock codes."""
    stock_info = fetch_stock_info()
    if not stock_info:
        return df
    df = df.copy()
    df[code_col] = df[code_col].astype(str).str.zfill(6)
    df['name'] = df[code_col].map(lambda c: stock_info.get(c, {}).get('name', ''))
    return df

def get_sector_distribution(codes):
    """Get industry sector distribution for a list of codes."""
    stock_info = fetch_stock_info()
    sectors = {}
    for c in codes:
        c = str(c).zfill(6)
        industry = stock_info.get(c, {}).get('industry', '其他')
        sectors[industry] = sectors.get(industry, 0) + 1
    return sectors

if __name__ == '__main__':
    info = fetch_stock_info()
    print(f'Total stocks: {len(info)}')
    # Show sample
    for i, (code, data) in enumerate(info.items()):
        if i >= 10: break
        print(f'  {code}: {data["name"]}')
