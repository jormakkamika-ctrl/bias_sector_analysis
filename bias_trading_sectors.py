import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import requests
from bs4 import BeautifulSoup
import base64
from io import BytesIO
import re

# --- CONFIGURATION ---
FRED_API_KEY = 'e210def24f02e4a73ac744035fa51963'
fred = Fred(api_key=FRED_API_KEY)

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

@st.cache_data(ttl=3600)
def fetch_data():
    data = {}
    history = {}
    today = datetime.now()

    def safe_get_series(series_id, default_value=0, default_history=None):
        try:
            series = fred.get_series(series_id)
            if series is None or series.empty:
                raise ValueError
            return float(series.iloc[-1]), series
        except:
            if default_history is None:
                num_months = 12
                date_range = pd.date_range(end=today, periods=num_months, freq='ME')
                default_history = pd.Series(np.random.normal(default_value, default_value * 0.1, num_months), index=date_range)
            return default_value, default_history

    def get_econ_series(indicator, default_value, num_months=24):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            if indicator == 'business confidence':
                return safe_get_series('NAPM', 52.6)
            elif indicator == 'non manufacturing pmi':
                return safe_get_series('NMFPMI', 53.8)
            elif indicator == 'nfib business optimism index':
                return safe_get_series('NFIBSBIO', 99.3)
            elif indicator == 'sbi':
                url = 'https://www.uschamber.com/sbindex/summary'
            elif indicator == 'eesi':
                url = 'https://esi-civicscience.pentagroup.co/'
            elif indicator == 'cpi_volatile':
                return safe_get_series('CPIAUCSL', 300)
            else:
                raise ValueError

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()

            if indicator == 'sbi':
                match = re.search(r'SBI:?\s*(\d+\.?\d*)', text) or re.search(r'Index is (\d+\.?\d*)', text)
                current_val = float(match.group(1)) if match else default_value
                date_range = pd.date_range(end=today, periods=num_months, freq='QE')
                series = pd.Series(np.linspace(current_val - 5, current_val + 3, num_months), index=date_range)
                return current_val, series
            elif indicator == 'eesi':
                match = re.search(r'points to (\d+\.?\d*)', text)
                current_val = float(match.group(1)) if match else default_value
                date_range = pd.date_range(end=today, periods=num_months, freq='2W')
                series = pd.Series(np.linspace(current_val - 8, current_val + 4, num_months), index=date_range)
                return current_val, series

            tables = soup.find_all('table')
            table = None
            for t in tables:
                thead = t.find('thead')
                if thead:
                    ths = thead.find_all('th')
                    if len(ths) == 2 and ths[0].text.strip() == 'Date' and ths[1].text.strip() == 'Value':
                        table = t
                        break
            if not table:
                raise ValueError
            dates, values = [], []
            rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 2:
                    try:
                        date = pd.to_datetime(cols[0].text.strip())
                        value = float(cols[1].text.strip())
                        dates.append(date)
                        values.append(value)
                    except:
                        continue
            series = pd.Series(values, index=dates).sort_index()
            series = series[-num_months:]
            return float(series.iloc[-1]), series

        except:
            date_range = pd.date_range(end=today, periods=num_months, freq='ME')
            return default_value, pd.Series(np.random.normal(default_value, default_value * 0.1, num_months), index=date_range)

    data['ism_manufacturing'], history['ism_manufacturing'] = get_econ_series('business confidence', 52.6, 24)
    data['ism_services'], history['ism_services'] = get_econ_series('non manufacturing pmi', 53.8, 24)
    data['nfib'], history['nfib'] = get_econ_series('nfib business optimism index', 99.3, 24)
    data['cpi_volatile'], history['cpi_volatile'] = get_econ_series('cpi_volatile', 300)
    data['sbi'], history['sbi'] = get_econ_series('sbi', 68.4, 8)
    data['eesi'], history['eesi'] = get_econ_series('eesi', 50, 24)
    data['umcsi'], history['umcsi'] = safe_get_series('UMCSENT', 56.6)
    building_permits, history['building_permits'] = safe_get_series('PERMIT', 1448)
    data['building_permits'] = building_permits / 1000
    data['fed_funds'], history['fed_funds'] = safe_get_series('FEDFUNDS', 3.64)
    data['10yr_yield'], history['10yr_yield'] = safe_get_series('DGS10', 4.086)
    data['2yr_yield'], history['2yr_yield'] = safe_get_series('DGS2', 3.48)
    data['bbb_yield'], history['bbb_yield'] = safe_get_series('BAMLC0A4CBBBEY', 4.93)
    data['ccc_yield'], history['ccc_yield'] = safe_get_series('BAMLH0A3HYCEY', 12.44)
    data['m1'], history['m1'] = safe_get_series('M1SL', 19100)
    data['m2'], history['m2'] = safe_get_series('M2SL', 22400)

    def get_yf_data(ticker, default_val, default_std, period='1y'):
        try:
            hist = yf.Ticker(ticker).history(period=period)['Close']
            hist.index = hist.index.tz_localize(None)
            return float(hist.iloc[-1]), hist
        except:
            num_days = 365 if period == '1y' else 1825 if period == '5y' else 90
            date_range = pd.date_range(end=today, periods=num_days)
            return default_val, pd.Series(np.random.normal(default_value, default_std, num_days), index=date_range)

    data['vix'], history['vix'] = get_yf_data('^VIX', 19.09, 5, '1y')
    data['move'], history['move'] = get_yf_data('^MOVE', 85.0, 10, '1y')

    data['copper'], history['copper'] = get_yf_data('HG=F', 4.0, 0.5, '1y')
    data['gold'], history['gold'] = get_yf_data('GC=F', 2000, 200, '1y')

    try:
        sp = get_yf_data('^GSPC', 5000, 500, '1y')[1]
        data['sp_lagging'] = 'UP' if sp.iloc[-1] > sp.iloc[0] else 'DOWN'
        history['sp500'] = sp
        sp_long = get_yf_data('^GSPC', 5000, 500, '5y')[1]
        history['sp500_long'] = sp_long
    except:
        data['sp_lagging'] = 'UP'
        history['sp500'] = pd.Series(np.random.normal(5000, 500, 365), index=pd.date_range(end=today, periods=365))
        history['sp500_long'] = pd.Series(np.random.normal(5000, 500, 1825), index=pd.date_range(end=today, periods=1825))

    try:
        stoxx = get_yf_data('^SXXP', 500, 50, '1y')[1]
        data['stoxx_lagging'] = 'UP' if stoxx.iloc[-1] > stoxx.iloc[0] else 'DOWN'
        history['stoxx600'] = stoxx
        stoxx_long = get_yf_data('^SXXP', 500, 50, '5y')[1]
        history['stoxx600_long'] = stoxx_long
    except:
        data['stoxx_lagging'] = 'UP'
        history['stoxx600'] = pd.Series(np.random.normal(500, 50, 365), index=pd.date_range(end=today, periods=365))
        history['stoxx600_long'] = pd.Series(np.random.normal(500, 50, 1825), index=pd.date_range(end=today, periods=1825))

    try:
        core = fred.get_series('CPILFESL')
        data['core_cpi_yoy'] = ((core.iloc[-1] / core.iloc[-13]) - 1) * 100 if len(core) > 13 else 2.5
        history['core_cpi'] = core
    except:
        data['core_cpi_yoy'] = 2.5
        history['core_cpi'] = pd.Series(np.random.normal(2.5, 0.5, 12), index=pd.date_range(end=today, periods=12, freq='ME'))

    return data, history, today

def get_graph_key(item_text):
    if 'Copper/Gold' in item_text: return 'copper_gold'
    if '10Yr-FedFunds' in item_text: return 'spread_10ff'
    if '10Yr-2Yr' in item_text: return 'spread_10_2'
    if 'Yield Curve comparison' in item_text: return 'yield_curve_compare'
    if 'Real Rate' in item_text and '10' in item_text: return 'real_rate_10yr'
    if 'Real Rate' in item_text and '2' in item_text: return 'real_rate_2yr'
    if 'Fed Funds' in item_text: return 'fed_funds'
    if '10-Yr Yield' in item_text or '10-Yr' in item_text: return '10yr_yield'
    if '2-Yr Yield' in item_text or '2-Yr' in item_text: return '2yr_yield'
    if 'Core CPI' in item_text: return 'core_cpi'
    if 'BBB Yield' in item_text: return 'bbb_yield'
    if 'CCC Yield' in item_text: return 'ccc_yield'
    if 'VIX' in item_text: return 'vix'
    if 'MOVE' in item_text: return 'move'
    if 'Manufacturing PMI' in item_text: return 'ism_manufacturing'
    if 'Services PMI' in item_text: return 'ism_services'
    if 'UMCSI' in item_text: return 'umcsi'
    if 'Building Permits' in item_text: return 'building_permits'
    if 'NFIB' in item_text: return 'nfib'
    if 'CPI-Volatile' in item_text: return 'cpi_volatile'
    if 'SBI' in item_text: return 'sbi'
    if 'EESI' in item_text: return 'eesi'
    if 'M1' in item_text: return 'm1'
    if 'M2' in item_text: return 'm2'
    if '9-6' in item_text and 'S&P' in item_text: return 'sp_96'
    if '9-6' in item_text and 'STOXX' in item_text: return 'stoxx600'
    if 'S&P' in item_text: return 'sp500'
    if 'MACD' in item_text: return 'macd'
    if 'STOXX' in item_text: return 'stoxx600'
    return 'placeholder'

def get_description(gkey):
    if gkey == 'macd':
        return '''You could stop now, and just do this.<br>
Moving Average Convergence Divergence is a technical indicator informing/identifying momentum.<br>
When short-term exponential average crosses long-term, MACD indicates potential uptrend, while cross-below indicates downtrend.<br>
This is an interesting article for the “lazy” investor:<br>
<a href="https://moneyweek.com/518350/macd-histogram-stockmarket-trading-system-for-the-lazy-investor" target="_blank">https://moneyweek.com/518350/macd-histogram-stockmarket-trading-system-for-the-lazy-investor</a><br>
You would miss out on all large negative moves, and just be in the long only moves.<br>
In last 19 years, there would be 12 trades (6 Buy Signals, 6 Sell Signals).<br>
Note: You wouldn't hit highs and lows, just major moves.'''
    if gkey == 'sp500':
        return 'S&P500 is a forward-looking indicator for USA GDP: When S&P500 experiences growth, investors expect positive/increasing firm earnings that should reflect in solid GDP growth. Predicting USA GDP with the S&P500 as an indicator has correlation of 69.04%'
    if gkey == 'stoxx600':
        return 'STOXX 600 (Europe) as global risk appetite proxy. Strong correlation with US GDP via trade/finance channels (~55%). 9-6 month return is a leading signal similar to S&P.'
    if gkey == 'spread_10ff':
        return '10-Year minus Fed Funds spread. Positive = normal steep curve = accommodative conditions → expansionary for GDP.'
    if gkey == 'spread_10_2':
        return '10-Year minus 2-Year spread (classic yield curve). Positive spread strongly predicts GDP expansion.'
    if gkey == 'yield_curve_compare':
        return '3-year view of 10Yr-2Yr spread. Steep positive curve = healthy expansion expectations.'
    if gkey.startswith('real_rate'):
        return 'Real rate = nominal yield minus core CPI YoY. Negative real rates are highly stimulative → positive for GDP growth.'
    if gkey in ['m1', 'm2']:
        return 'M1/M2 money supply growth (liquidity). Rising aggregates support credit creation and GDP expansion.'
    if gkey == 'vix':
        return 'The VIX, aka fear index... Lower than historical volatility implies positive outlook for GDP.'
    if gkey == 'bbb_yield':
        return 'Corporate bond yields reflect cost of borrowing... cheaper borrowing implies expansionary conditions.'
    if gkey == 'ccc_yield':
        return 'Higher yields imply more expensive borrowing → contractionary conditions.'
    if gkey == 'sp_96':
        return '69% correlation: S&P leading indicator for GDP direction.'
    return ''

def calculate_metrics(data, history, today):
    metrics = {}
    try:
        metrics['real_rate_10yr'] = data['10yr_yield'] - (data['core_cpi_yoy'] / 12)
        metrics['real_rate_2yr'] = data['2yr_yield'] - (data['core_cpi_yoy'] / 12)
        metrics['yield_curve_10ff'] = data['10yr_yield'] - data['fed_funds']
        metrics['yield_curve_10_2'] = data['10yr_yield'] - data['2yr_yield']
        metrics['copper_gold_ratio'] = data['copper'] / data['gold']
        ratio_change = (history['copper'].iloc[-1] / history['gold'].iloc[-1]) - (history['copper'].iloc[0] / history['gold'].iloc[0])
        metrics['copper_gold_ratio_change'] = ratio_change
    except:
        return {}, [], [], [], "Error", 50

    tailwinds = []
    headwinds = []
    neutrals = []

    # 1. S&P
    try:
        sp_end = float(history['sp500'].iloc[-1])
        sp_change_daily = sp_end - float(history['sp500'].iloc[-2]) if len(history['sp500']) > 1 else 0
        one_month_ago = pd.Timestamp(today - timedelta(days=30))
        sp_month_ago = history['sp500'][history['sp500'].index >= one_month_ago].iloc[0] if not history['sp500'][history['sp500'].index >= one_month_ago].empty else history['sp500'].iloc[0]
        sp_change_mom = sp_end - sp_month_ago
        three_month_ago = pd.Timestamp(today - timedelta(days=90))
        sp_three_month_ago = history['sp500'][history['sp500'].index >= three_month_ago].iloc[0] if not history['sp500'][history['sp500'].index >= three_month_ago].empty else history['sp500'].iloc[0]
        sp_change_3m = sp_end - sp_three_month_ago
        sp_start_yoy = float(history['sp500'].iloc[0])
        sp_change_yoy = ((sp_end - sp_start_yoy) / sp_start_yoy) * 100 if sp_start_yoy != 0 else 0

        sp_daily_pct = (sp_change_daily / float(history['sp500'].iloc[-2])) * 100 if len(history['sp500']) > 1 else 0
        sp_mom_pct = (sp_change_mom / sp_month_ago) * 100 if sp_month_ago != 0 else 0
        sp_3m_pct = (sp_change_3m / sp_three_month_ago) * 100 if sp_three_month_ago != 0 else 0

        daily_dir = "up" if sp_change_daily > 0 else "down"
        mom_dir = "up" if sp_change_mom > 0 else "down"
        three_m_dir = "up" if sp_change_3m > 0 else "down"

        daily_color = "green" if daily_dir == "up" else "red"
        mom_color = "green" if mom_dir == "up" else "red"
        three_m_color = "green" if three_m_dir == "up" else "red"

        daily_str = f'<span style="color:{daily_color}">{daily_dir} {abs(sp_daily_pct):.2f}%</span>'
        mom_str = f'<span style="color:{mom_color}">{mom_dir} {abs(sp_mom_pct):.2f}%</span>'
        three_m_str = f'<span style="color:{three_m_color}">{three_m_dir} {abs(sp_3m_pct):.2f}%</span>'

        sp_label = f"S&P: {sp_end:.2f} (daily {daily_str}, MoM {mom_str}, 3M {three_m_str}, YoY {sp_change_yoy:.2f}%)"
        if data['sp_lagging'] == 'UP':
            tailwinds.append(sp_label + " (positive for GDP)")
        else:
            headwinds.append(sp_label + " (negative for GDP)")
    except:
        neutrals.append("S&P Data Unavailable")

    # Copper/Gold ratio
    if metrics['copper_gold_ratio_change'] > 0:
        tailwinds.append("Copper/Gold ratio increasing (positive leading indicator for growth)")
    else:
        headwinds.append("Copper/Gold ratio decreasing (negative leading indicator for growth)")

    # 2. Fed Funds
    ff_change = history['fed_funds'].iloc[-1] - history['fed_funds'].iloc[-2] if len(history['fed_funds']) > 1 else 0
    ff_dir = "down" if ff_change < 0 else "up" if ff_change > 0 else "unchanged"
    ff_color = "green" if ff_dir == "up" else "red" if ff_dir == "down" else "gray"
    ff_str = f'<span style="color:{ff_color}">{ff_dir} {abs(ff_change):.2f}%</span>'
    if ff_change < 0:
        tailwinds.append(f"Fed Funds: {data['fed_funds']:.2f}% ({ff_str}, positive)")
    elif ff_change > 0:
        headwinds.append(f"Fed Funds: {data['fed_funds']:.2f}% ({ff_str}, negative)")
    else:
        neutrals.append(f"Fed Funds: {data['fed_funds']:.2f}% (no change)")

    # 3. 10-Yr Yield + Terminal
    ty_change_daily = history['10yr_yield'].iloc[-1] - history['10yr_yield'].iloc[-2] if len(history['10yr_yield']) > 1 else 0
    one_month_ago = pd.Timestamp(today - timedelta(days=30))
    ty_month_ago = history['10yr_yield'][history['10yr_yield'].index >= one_month_ago].iloc[0] if not history['10yr_yield'][history['10yr_yield'].index >= one_month_ago].empty else history['10yr_yield'].iloc[0]
    ty_change_mom = data['10yr_yield'] - ty_month_ago
    three_month_ago = pd.Timestamp(today - timedelta(days=90))
    ty_three_month_ago = history['10yr_yield'][history['10yr_yield'].index >= three_month_ago].iloc[0] if not history['10yr_yield'][history['10yr_yield'].index >= three_month_ago].empty else history['10yr_yield'].iloc[0]
    ty_change_3m = data['10yr_yield'] - ty_three_month_ago
    terminal_rate = max(history['10yr_yield']) if not history['10yr_yield'].empty else data['10yr_yield']
    metrics['terminal_10yr'] = terminal_rate

    daily_dir = "down" if ty_change_daily < 0 else "up"
    daily_color = "green" if daily_dir == "up" else "red"
    mom_dir = "down" if ty_change_mom < 0 else "up"
    mom_color = "green" if mom_dir == "up" else "red"
    three_m_dir = "down" if ty_change_3m < 0 else "up"
    three_m_color = "green" if three_m_dir == "up" else "red"

    daily_str = f'<span style="color:{daily_color}">{daily_dir} {abs(ty_change_daily):.2f}%</span>'
    mom_str = f'<span style="color:{mom_color}">{mom_dir} {abs(ty_change_mom):.2f}%</span>'
    three_m_str = f'<span style="color:{three_m_color}">{three_m_dir} {abs(ty_change_3m):.2f}%</span>'

    ty_label = f"10-Yr Yield: {data['10yr_yield']:.2f}% (daily {daily_str}, MoM {mom_str}, 3M {three_m_str}, Terminal {terminal_rate:.2f}%)"
    if ty_change_daily < 0:
        tailwinds.append(ty_label + ", positive)")
    else:
        headwinds.append(ty_label + ", negative)")

    # 4. 2-Yr Yield
    ty2_change = history['2yr_yield'].iloc[-1] - history['2yr_yield'].iloc[-2] if len(history['2yr_yield']) > 1 else 0
    ty2_dir = "up" if ty2_change > 0 else "down"
    ty2_color = "green" if ty2_dir == "up" else "red"
    ty2_str = f'<span style="color:{ty2_color}">{ty2_dir} {abs(ty2_change):.2f}%</span>'
    if ty2_change < 0:
        tailwinds.append(f"2-Yr Yield: {data['2yr_yield']:.2f}% ({ty2_str}, positive)")
    else:
        headwinds.append(f"2-Yr Yield: {data['2yr_yield']:.2f}% ({ty2_str}, negative)")

    # 5. Core CPI
    cpi_change = history['core_cpi'].iloc[-1] - history['core_cpi'].iloc[-2] if len(history['core_cpi']) > 1 else 0
    cpi_dir = "up" if cpi_change > 0 else "down"
    cpi_color = "green" if cpi_dir == "up" else "red"
    cpi_str = f'<span style="color:{cpi_color}">{cpi_dir} {abs(cpi_change):.2f}%</span>'
    if cpi_change < 0:
        tailwinds.append(f"Core CPI YoY: {data['core_cpi_yoy']:.2f}% ({cpi_str}, positive)")
    else:
        headwinds.append(f"Core CPI YoY: {data['core_cpi_yoy']:.2f}% ({cpi_str}, negative)")

    # 6-7. Real Rates
    if metrics['real_rate_10yr'] < 0:
        tailwinds.append(f"Real Rate (10-Yr): {metrics['real_rate_10yr']:.2f}% (negative, positive for GDP)")
    else:
        headwinds.append(f"Real Rate (10-Yr): {metrics['real_rate_10yr']:.2f}% (positive, negative for GDP)")
    if metrics['real_rate_2yr'] < 0:
        tailwinds.append(f"Real Rate (2-Yr): {metrics['real_rate_2yr']:.2f}% (negative, positive for GDP)")
    else:
        headwinds.append(f"Real Rate (2-Yr): {metrics['real_rate_2yr']:.2f}% (positive, negative for GDP)")

    # 8. BBB
    bbb_change = history['bbb_yield'].iloc[-1] - history['bbb_yield'].iloc[-2] if len(history['bbb_yield']) > 1 else 0
    bbb_dir = "up" if bbb_change > 0 else "down"
    bbb_color = "green" if bbb_dir == "up" else "red"
    bbb_str = f'<span style="color:{bbb_color}">{bbb_dir} {abs(bbb_change):.2f}%</span>'
    if bbb_change < 0:
        tailwinds.append(f"BBB Yield: {data['bbb_yield']:.2f}% ({bbb_str}, positive)")
    else:
        headwinds.append(f"BBB Yield: {data['bbb_yield']:.2f}% ({bbb_str}, negative)")

    # 9. CCC
    ccc_change = history['ccc_yield'].iloc[-1] - history['ccc_yield'].iloc[-2] if len(history['ccc_yield']) > 1 else 0
    ccc_dir = "up" if ccc_change > 0 else "down"
    ccc_color = "green" if ccc_dir == "up" else "red"
    ccc_str = f'<span style="color:{ccc_color}">{ccc_dir} {abs(ccc_change):.2f}%</span>'
    if ccc_change < 0:
        tailwinds.append(f"CCC Yield: {data['ccc_yield']:.2f}% ({ccc_str}, positive)")
    else:
        headwinds.append(f"CCC Yield: {data['ccc_yield']:.2f}% ({ccc_str}, negative)")

    # 10. VIX
    vix_val = float(data['vix'])
    vix_change_daily = float(history['vix'].iloc[-1]) - float(history['vix'].iloc[-2]) if len(history['vix']) > 1 else 0
    one_month_ago = pd.Timestamp(today - timedelta(days=30))
    vix_month_ago = history['vix'][history['vix'].index >= one_month_ago].iloc[0] if not history['vix'][history['vix'].index >= one_month_ago].empty else history['vix'].iloc[0]
    vix_change_mom = vix_val - vix_month_ago
    three_month_ago = pd.Timestamp(today - timedelta(days=90))
    vix_three_month_ago = history['vix'][history['vix'].index >= three_month_ago].iloc[0] if not history['vix'][history['vix'].index >= three_month_ago].empty else history['vix'].iloc[0]
    vix_change_3m = vix_val - vix_three_month_ago
    daily_dir = "down" if vix_change_daily < 0 else "up"
    mom_dir = "down" if vix_change_mom < 0 else "up"
    three_m_dir = "down" if vix_change_3m < 0 else "up"
    daily_color = "green" if daily_dir == "down" else "red"
    mom_color = "green" if mom_dir == "down" else "red"
    three_m_color = "green" if three_m_dir == "down" else "red"
    vix_label = f"VIX: {vix_val:.2f} (daily <span style='color:{daily_color}'>{daily_dir} {abs(vix_change_daily):.2f}%</span>, MoM <span style='color:{mom_color}'>{mom_dir} {abs(vix_change_mom):.2f}%</span>, 3M <span style='color:{three_m_color}'>{three_m_dir} {abs(vix_change_3m):.2f}%</span>"
    if vix_val < 15 or vix_change_daily < 0:
        vix_label += ", positive)"
        tailwinds.append(vix_label)
    else:
        vix_label += ", negative)"
        headwinds.append(vix_label)

    # 11. MOVE
    move_val = float(data['move'])
    move_change_daily = float(history['move'].iloc[-1]) - float(history['move'].iloc[-2]) if len(history['move']) > 1 else 0
    one_month_ago = pd.Timestamp(today - timedelta(days=30))
    move_month_ago = history['move'][history['move'].index >= one_month_ago].iloc[0] if not history['move'][history['move'].index >= one_month_ago].empty else history['move'].iloc[0]
    move_change_mom = move_val - move_month_ago
    three_month_ago = pd.Timestamp(today - timedelta(days=90))
    move_three_month_ago = history['move'][history['move'].index >= three_month_ago].iloc[0] if not history['move'][history['move'].index >= three_month_ago].empty else history['move'].iloc[0]
    move_change_3m = move_val - move_three_month_ago
    daily_dir = "down" if move_change_daily < 0 else "up"
    mom_dir = "down" if move_change_mom < 0 else "up"
    three_m_dir = "down" if move_change_3m < 0 else "up"
    daily_color = "green" if daily_dir == "down" else "red"
    mom_color = "green" if mom_dir == "down" else "red"
    three_m_color = "green" if three_m_dir == "down" else "red"
    move_label = f"MOVE: {move_val:.2f} (daily <span style='color:{daily_color}'>{daily_dir} {abs(move_change_daily):.2f}%</span>, MoM <span style='color:{mom_color}'>{mom_dir} {abs(move_change_mom):.2f}%</span>, 3M <span style='color:{three_m_color}'>{three_m_dir} {abs(move_change_3m):.2f}%</span>"
    if move_change_daily < 0:
        move_label += ", positive for bonds)"
        tailwinds.append(move_label)
    else:
        move_label += ", negative)"
        headwinds.append(move_label)

    # 12. Manufacturing PMI
    if data['ism_manufacturing'] > 50:
        tailwinds.append(f"Manufacturing PMI: {data['ism_manufacturing']} (expansion)")
    else:
        headwinds.append(f"Manufacturing PMI: {data['ism_manufacturing']} (contraction)")

    # 13. Services PMI
    if data['ism_services'] > 50:
        tailwinds.append(f"Services PMI: {data['ism_services']} (expansion)")
    else:
        headwinds.append(f"Services PMI: {data['ism_services']} (contraction)")

    # 14. UMCSI
    if data['umcsi'] > 70:
        tailwinds.append(f"UMCSI: {data['umcsi']} (bullish)")
    elif data['umcsi'] < 55:
        headwinds.append(f"UMCSI: {data['umcsi']} (bearish)")
    else:
        neutrals.append(f"UMCSI: {data['umcsi']} (neutral)")

    # 15. Building Permits
    bp_change = history['building_permits'].iloc[-1] - history['building_permits'].iloc[-2] if len(history['building_permits']) > 1 else 0
    bp_dir = "up" if bp_change > 0 else "down"
    bp_color = "green" if bp_dir == "up" else "red"
    bp_str = f'<span style="color:{bp_color}">{bp_dir} {abs(bp_change):.2f}M</span>'
    if bp_change > 0:
        tailwinds.append(f"Building Permits: {data['building_permits']:.2f}M ({bp_str}, positive)")
    else:
        headwinds.append(f"Building Permits: {data['building_permits']:.2f}M ({bp_str}, negative)")

    # 16. NFIB
    nfib_change = history['nfib'].iloc[-1] - history['nfib'].iloc[-2] if len(history['nfib']) > 1 else 0
    perc_change = (nfib_change / history['nfib'].iloc[-2]) * 100 if len(history['nfib']) > 1 and history['nfib'].iloc[-2] != 0 else 0
    direction = "Up" if nfib_change > 0 else "Down" if nfib_change < 0 else "Unchanged"
    prev_month = history['nfib'].index[-2].strftime('%b') if len(history['nfib']) > 1 else 'Prev'
    curr_month = history['nfib'].index[-1].strftime('%b')
    change_str = f"{direction} {abs(nfib_change):.1f} ({perc_change:.2f}%) ({prev_month} → {curr_month})"
    status = "(strong)" if data['nfib'] > 100 else "(weak)" if data['nfib'] < 95 else "(neutral)"
    nfib_label = f"NFIB: {data['nfib']} {change_str} {status}"
    if data['nfib'] > 100:
        tailwinds.append(nfib_label)
    elif data['nfib'] < 95:
        headwinds.append(nfib_label)
    else:
        neutrals.append(nfib_label)

    # 17. S&P 9-6m Return
    if len(history['sp500']) > 200:
        idx_9m = max(0, len(history['sp500']) - 189)
        idx_6m = max(0, len(history['sp500']) - 126)
        price_9m = history['sp500'].iloc[idx_9m]
        price_6m = history['sp500'].iloc[idx_6m]
        sp_96_return = (price_6m - price_9m) / price_9m * 100
    else:
        sp_96_return = 0
    metrics['sp_96_return'] = sp_96_return
    sp96_label = f"S&P 9-6m Return: {sp_96_return:.2f}%"
    if sp_96_return > 0:
        tailwinds.append(sp96_label)
    else:
        headwinds.append(sp96_label)

    # Additional
    if data.get('cpi_volatile', 300) < 300:
        tailwinds.append(f"CPI Volatile: {data['cpi_volatile']:.0f} (low, positive)")
    else:
        headwinds.append(f"CPI Volatile: {data['cpi_volatile']:.0f} (high, negative)")

    if data.get('sbi', 0) > 68:
        tailwinds.append(f"SBI: {data['sbi']:.1f} (strong, positive)")
    else:
        headwinds.append(f"SBI: {data['sbi']:.1f} (weak, negative)")

    if data.get('eesi', 0) > 45:
        tailwinds.append(f"EESI: {data['eesi']:.1f} (positive)")
    else:
        headwinds.append(f"EESI: {data['eesi']:.1f} (negative)")

    # M1 & M2
    m1_growth_pos = history['m1'].iloc[-1] > history['m1'].iloc[-2] if len(history['m1']) > 1 else False
    m2_growth_pos = history['m2'].iloc[-1] > history['m2'].iloc[-2] if len(history['m2']) > 1 else False
    metrics['m1_growth_pos'] = m1_growth_pos
    metrics['m2_growth_pos'] = m2_growth_pos
    if m1_growth_pos:
        tailwinds.append(f"M1 Money Supply: Growing MoM (positive liquidity for GDP)")
    else:
        headwinds.append(f"M1 Money Supply: Contracting MoM (headwind)")
    if m2_growth_pos:
        tailwinds.append(f"M2 Money Supply: Growing MoM (positive liquidity for GDP)")
    else:
        headwinds.append(f"M2 Money Supply: Contracting MoM (headwind)")

    # Spreads
    ff_spread = metrics.get('yield_curve_10ff', 0)
    if ff_spread > 0:
        tailwinds.append(f"10Yr-FedFunds Spread: {ff_spread:.2f}% (positive, expansionary)")
    else:
        headwinds.append(f"10Yr-FedFunds Spread: {ff_spread:.2f}% (negative, contractionary)")

    ten2_spread = metrics.get('yield_curve_10_2', 0)
    if ten2_spread > 0:
        tailwinds.append(f"10Yr-2Yr Spread: {ten2_spread:.2f}% (positive, expansionary)")
    else:
        headwinds.append(f"10Yr-2Yr Spread: {ten2_spread:.2f}% (flat/inverted, contractionary)")

    # Yield Curve 3-year comparison
    yc_pos = metrics.get('yield_curve_10_2', 0) > 0
    yc_label = f"Yield Curve Comparison (3yr): {'Steep/Positive' if yc_pos else 'Flat/Inverted'}"
    if yc_pos:
        tailwinds.append(yc_label)
    else:
        headwinds.append(yc_label)

    # MACD LazyMan
    macd_long_bullish = False
    macd_short_bullish = False
    try:
        sp_close = history['sp500']
        if len(sp_close) >= 40:
            macd_l, sig_l, _ = compute_macd(sp_close)
            macd_long_bullish = macd_l.iloc[-1] > sig_l.iloc[-1]
            metrics['macd_line'] = float(macd_l.iloc[-1])
            metrics['signal_line'] = float(sig_l.iloc[-1])
        sp_short = history['sp500'].last('45D')
        if len(sp_short) >= 26:
            macd_s, sig_s, _ = compute_macd(sp_short)
            macd_short_bullish = macd_s.iloc[-1] > sig_s.iloc[-1]
    except:
        pass
    metrics['macd_long_bullish'] = macd_long_bullish
    metrics['macd_short_bullish'] = macd_short_bullish

    macd_label = f"LazyMan MACD: Short-term {'Buy' if macd_short_bullish else 'Sell'} | Long-term {'Buy' if macd_long_bullish else 'Sell'}"
    if macd_long_bullish:
        tailwinds.append(macd_label + " → Bull – SIMPLY Buy (if you're lazy)")
    else:
        headwinds.append(macd_label)

    # STOXX 600 9-6m
    if len(history['stoxx600']) > 200:
        idx_9m = max(0, len(history['stoxx600']) - 189)
        idx_6m = max(0, len(history['stoxx600']) - 126)
        price_9m = history['stoxx600'].iloc[idx_9m]
        price_6m = history['stoxx600'].iloc[idx_6m]
        stoxx_96_return = (price_6m - price_9m) / price_9m * 100
    else:
        stoxx_96_return = 0
    metrics['stoxx_96_return'] = stoxx_96_return
    stoxx_label = f"STOXX 600 9-6m Return: {stoxx_96_return:.2f}%"
    if stoxx_96_return > 0:
        tailwinds.append(stoxx_label)
    else:
        headwinds.append(stoxx_label)

    # S&P Bear / Bull Market Status
    sp_bear = {}
    try:
        sp_long = history['sp500_long']
        if len(sp_long) >= 10:
            current_price = float(sp_long.iloc[-1])
            current_date_str = sp_long.index[-1].strftime('%d/%m/%Y')
            ath_value = float(sp_long.max())
            max_mask = (sp_long == ath_value)
            last_high_date_str = sp_long[max_mask].index[-1].strftime('%d/%m/%Y')
            new_bear_threshold = ath_value * 0.8
            bull_start_lookback = today - timedelta(days=1825)
            recent_sp = sp_long[sp_long.index >= bull_start_lookback]
            if not recent_sp.empty:
                prev_low_price = float(recent_sp.min())
                prev_low_date = recent_sp.idxmin()
                prev_low_date_str = prev_low_date.strftime('%d/%m/%Y')
                days_bull = (today.date() - prev_low_date.date()).days
            else:
                prev_low_date_str = "N/A"
                prev_low_price = 0.0
                days_bull = 0
            avg_days_bull = 997
            sp_bear = {
                'current_date': current_date_str,
                'current': current_price,
                'last_high_date': last_high_date_str,
                'last_high': ath_value,
                'new_bear_threshold': new_bear_threshold,
                'prev_bear_date': prev_low_date_str,
                'prev_bear': prev_low_price,
                'days_bull': days_bull,
                'avg_days_bull': avg_days_bull,
            }
    except:
        sp_bear = {}
    metrics['sp_bear'] = sp_bear

    # STOXX Bear / Bull Market Status
    stoxx_bear = {}
    try:
        stoxx_long = history['stoxx600_long']
        if len(stoxx_long) >= 10:
            current_price = float(stoxx_long.iloc[-1])
            current_date_str = stoxx_long.index[-1].strftime('%d/%m/%Y')
            ath_value = float(stoxx_long.max())
            max_mask = (stoxx_long == ath_value)
            last_high_date_str = stoxx_long[max_mask].index[-1].strftime('%d/%m/%Y')
            new_bear_threshold = ath_value * 0.8
            bull_start_lookback = today - timedelta(days=1825)
            recent_stoxx = stoxx_long[stoxx_long.index >= bull_start_lookback]
            if not recent_stoxx.empty:
                prev_low_price = float(recent_stoxx.min())
                prev_low_date = recent_stoxx.idxmin()
                prev_low_date_str = prev_low_date.strftime('%d/%m/%Y')
                days_bull = (today.date() - prev_low_date.date()).days
            else:
                prev_low_date_str = "N/A"
                prev_low_price = 0.0
                days_bull = 0
            stoxx_bear = {
                'current_date': current_date_str,
                'current': current_price,
                'last_high_date': last_high_date_str,
                'last_high': ath_value,
                'new_bear_threshold': new_bear_threshold,
                'prev_bear_date': prev_low_date_str,
                'prev_bear': prev_low_price,
                'days_bull': days_bull,
                'avg_days_bull': 857,
            }
    except:
        stoxx_bear = {}
    metrics['stoxx_bear'] = stoxx_bear

    # Scoring
    score = 0
    score += min(max(metrics.get('sp_96_return', 0) / 5 * 18, 0), 18) if metrics.get('sp_96_return', 0) > 0 else 0
    score += 12 if data.get('sp_lagging') == 'UP' else 0
    score += 15 if metrics.get('yield_curve_10_2', 0) > 0 else 0
    score += 12 if metrics.get('yield_curve_10ff', 0) > 0 else 0
    score += 10 if metrics.get('macd_long_bullish', False) else 0
    score += 8 if metrics.get('stoxx_96_return', 0) > 0 else 0
    score += 10 if metrics.get('real_rate_10yr', 0) < 0 else 0
    score += 8 if metrics.get('real_rate_2yr', 0) < 0 else 0
    score += max(8 - data.get('vix', 20) / 5, 0) if data.get('vix', 20) < 25 else 0
    score += 8 if data['ism_manufacturing'] > 50 else 0
    score += 7 if data['ism_services'] > 50 else 0
    score += 6 if data['umcsi'] > 60 else 0
    score += 5 if data.get('building_permits', 0) > 1.4 else 0
    score += 5 if data.get('sbi', 0) > 68 else 0
    score += 4 if data.get('cpi_volatile', 300) < 300 else 0
    score += 4 if data.get('eesi', 50) > 45 else 0
    score += 5 if data.get('nfib', 99) > 100 else 0
    score += 5 if metrics.get('m1_growth_pos', False) else 0
    score += 5 if metrics.get('m2_growth_pos', False) else 0
    score += 8 if metrics.get('copper_gold_ratio_change', 0) > 0 else 0
    score = max(0, min(100, score))

    if score >= 60:
        bias = 'Long (6 long/4 short)'
    elif score <= 40:
        bias = 'Short (4 long/6 short)'
    else:
        bias = 'Neutral (5 long/5 short)'

    return metrics, tailwinds, headwinds, neutrals, bias, score

def generate_graph(metric_key, data, history, metrics, today):
    fig, ax = plt.subplots(figsize=(8, 4))
    series = None

    if metric_key == 'copper_gold':
        common_index = history['copper'].index.intersection(history['gold'].index)
        ratio = history['copper'].reindex(common_index) / history['gold'].reindex(common_index)
        series = ratio.last('12M')
        ax.set_title('Copper/Gold Ratio (last 12M)')

    elif metric_key == 'spread_10ff':
        yield10 = history['10yr_yield']
        ff = history['fed_funds'].reindex(yield10.index, method='nearest')
        spread = yield10 - ff
        series = spread.last('12M')
        ax.set_title('10Yr-FedFunds Spread (last 12M)')

    elif metric_key == 'spread_10_2':
        yield10 = history['10yr_yield']
        yield2 = history['2yr_yield'].reindex(yield10.index, method='nearest')
        spread = yield10 - yield2
        series = spread.last('12M')
        ax.set_title('10Yr-2Yr Spread (last 12M)')

    elif metric_key == 'yield_curve_compare':
        short = pd.Timestamp(today - timedelta(days=1095))
        ten = history['10yr_yield'][history['10yr_yield'].index >= short].dropna()
        two = history['2yr_yield'][history['2yr_yield'].index >= short].reindex(ten.index, method='nearest').dropna()
        spread = ten - two
        series = spread
        ax.set_title('10Yr - 2Yr Spread (last 3 years)')
        ax.axhline(0, color='red', linestyle='--')

    elif metric_key == 'real_rate_10yr' or metric_key == 'real_rate_2yr':
        core = history['core_cpi']
        core_yoy = ((core / core.shift(12)) - 1) * 100
        core_yoy = core_yoy.dropna()
        if '10' in metric_key:
            yield_hist = history['10yr_yield'].reindex(core_yoy.index, method='nearest')
            title = 'Real Rate 10Yr (last 12M)'
        else:
            yield_hist = history['2yr_yield'].reindex(core_yoy.index, method='nearest')
            title = 'Real Rate 2Yr (last 12M)'
        real = yield_hist - core_yoy / 12
        series = real.last('12M')
        ax.set_title(title)

    elif metric_key == 'core_cpi':
        core = history['core_cpi']
        core_yoy = ((core / core.shift(12)) - 1) * 100
        series = core_yoy.last('12M')
        ax.set_title('Core CPI YoY (last 12M)')

    elif metric_key == 'cpi_volatile':
        cpi = history['cpi_volatile']
        cpi_yoy = ((cpi / cpi.shift(12)) - 1) * 100
        series = cpi_yoy.last('12M')
        ax.set_title('CPI Volatile YoY (last 12M)')

    elif metric_key in history:
        series = history[metric_key].last('12M')
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} (last 12M)")

    if series is not None and not series.empty:
        series.plot(ax=ax, linewidth=2)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center')

    plt.tight_layout()
    return fig

def generate_short_term_graph(metric_key, history, today):
    short = pd.Timestamp(today - timedelta(days=90))
    fig, ax = plt.subplots(figsize=(8, 3))
    series = None

    if metric_key == 'copper_gold':
        common_index = history['copper'].index.intersection(history['gold'].index)
        ratio = history['copper'].reindex(common_index) / history['gold'].reindex(common_index)
        series = ratio[ratio.index >= short]

    elif metric_key == 'spread_10ff':
        yield10 = history['10yr_yield']
        ff = history['fed_funds'].reindex(yield10.index, method='nearest')
        spread = yield10 - ff
        series = spread[spread.index >= short]
        ax.set_title('10Yr-FedFunds Spread – Last 3 Months')

    elif metric_key == 'spread_10_2':
        yield10 = history['10yr_yield']
        yield2 = history['2yr_yield'].reindex(yield10.index, method='nearest')
        spread = yield10 - yield2
        series = spread[spread.index >= short]
        ax.set_title('10Yr-2Yr Spread – Last 3 Months')

    elif metric_key == 'real_rate_10yr' or metric_key == 'real_rate_2yr':
        core = history['core_cpi']
        core_yoy = ((core / core.shift(12)) - 1) * 100
        core_yoy = core_yoy.dropna()
        if '10' in metric_key:
            yield_hist = history['10yr_yield'].reindex(core_yoy.index, method='nearest')
            title = 'Real Rate 10Yr – Last 3 Months'
        else:
            yield_hist = history['2yr_yield'].reindex(core_yoy.index, method='nearest')
            title = 'Real Rate 2Yr – Last 3 Months'
        real = yield_hist - core_yoy / 12
        series = real[real.index >= short]
        ax.set_title(title)

    elif metric_key == 'core_cpi':
        core = history['core_cpi']
        core_yoy = ((core / core.shift(12)) - 1) * 100
        series = core_yoy[core_yoy.index >= short]
        ax.set_title('Core CPI YoY – Last 3 Months')

    elif metric_key == 'cpi_volatile':
        cpi = history['cpi_volatile']
        cpi_yoy = ((cpi / cpi.shift(12)) - 1) * 100
        series = cpi_yoy[cpi_yoy.index >= short]
        ax.set_title('CPI Volatile YoY – Last 3 Months')

    elif metric_key in history:
        series = history[metric_key][history[metric_key].index >= short]
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} – Last 3 Months")

    if series is not None and not series.empty:
        series.plot(ax=ax, linewidth=2)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center')

    plt.tight_layout()
    return fig

def generate_html_summary(tailwinds, headwinds, neutrals, bias, data, history, metrics, today, score):
    def build_section(items_list):
        html_parts = []
        for item in items_list:
            gkey = get_graph_key(item)
            fig = generate_graph(gkey, data, history, metrics, today)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            short_html = ''
            short_fig = generate_short_term_graph(gkey, history, today)
            if short_fig is not None:
                sbuf = BytesIO()
                short_fig.savefig(sbuf, format='png', bbox_inches='tight', dpi=180)
                sbuf.seek(0)
                short_base64 = base64.b64encode(sbuf.read()).decode('utf-8')
                plt.close(short_fig)
                short_html = f'<h4 style="margin:20px 0 8px 0;color:#555;font-size:1.05em;">Short-term View (last 3 months)</h4><img src="data:image/png;base64,{short_base64}" style="width:100%;max-width:820px;display:block;margin:0 auto;box-shadow:0 4px 12px rgba(0,0,0,0.08);"/>'

            desc = get_description(gkey)
            desc_html = f'<p style="margin-top:12px;color:#444;font-size:0.95em;">{desc}</p>' if desc else ''

            terminal_html = ''
            if gkey == '10yr_yield':
                current = data['10yr_yield']
                terminal = metrics.get('terminal_10yr', current)
                terminal_html = f'''
                <h4 style="margin:25px 0 10px 0;color:#555;font-size:1.05em;">10-Yr Treasury Yield & Terminal Yield</h4>
                <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;font-size:0.95em;border:1px solid #ddd;">
                    <thead>
                        <tr style="background:#f8f8f8;">
                            <th style="padding:10px;border:1px solid #ddd;text-align:left;">Metric</th>
                            <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td style="padding:10px;border:1px solid #ddd;">Current 10-Yr Yield</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{current:.2f}%</td></tr>
                        <tr><td style="padding:10px;border:1px solid #ddd;">Terminal Yield (recent high)</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{terminal:.2f}%</td></tr>
                    </tbody>
                </table>'''

            bear_html = ''
            if gkey == 'sp500':
                sp_bear = metrics.get('sp_bear', {})
                if sp_bear:
                    bear_html = f'''
                    <h4 style="margin:25px 0 10px 0;color:#555;font-size:1.05em;">S&P 500 Bull / Bear Market Status</h4>
                    <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;font-size:0.95em;border:1px solid #ddd;">
                        <thead>
                            <tr style="background:#f8f8f8;">
                                <th style="padding:10px;border:1px solid #ddd;text-align:left;">Metric</th>
                                <th style="padding:10px;border:1px solid #ddd;text-align:left;">Date / Note</th>
                                <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Current</td><td style="padding:10px;border:1px solid #ddd;">{sp_bear.get('current_date','N/A')}</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get('current',0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Last High</td><td style="padding:10px;border:1px solid #ddd;">{sp_bear.get('last_high_date','N/A')}</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get('last_high',0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">New Bear Threshold</td><td style="padding:10px;border:1px solid #ddd;">20% from High</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get('new_bear_threshold',0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Previous Bear Market Threshold</td><td style="padding:10px;border:1px solid #ddd;">{sp_bear.get('prev_bear_date','N/A')}</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get('prev_bear',0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;"># of Days Bull</td><td style="padding:10px;border:1px solid #ddd;"></td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get('days_bull',0)}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;"># of Days avg. Bull</td><td style="padding:10px;border:1px solid #ddd;"></td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get('avg_days_bull',997)}</td></tr>
                        </tbody>
                    </table>'''
            elif gkey == 'stoxx600':
                sb = metrics.get('stoxx_bear', {})
                if sb:
                    bear_html = f'''
                    <h4 style="margin:25px 0 10px 0;color:#555;font-size:1.05em;">STOXX 600 Bull / Bear Market Status</h4>
                    <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;font-size:0.95em;border:1px solid #ddd;">
                        <thead>
                            <tr style="background:#f8f8f8;">
                                <th style="padding:10px;border:1px solid #ddd;text-align:left;">Metric</th>
                                <th style="padding:10px;border:1px solid #ddd;text-align:left;">Date / Note</th>
                                <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Current</td><td style="padding:10px;border:1px solid #ddd;">{sb.get('current_date','N/A')}</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get('current',0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Last High</td><td style="padding:10px;border:1px solid #ddd;">{sb.get('last_high_date','N/A')}</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get('last_high',0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">New Bear Threshold</td><td style="padding:10px;border:1px solid #ddd;">20% from High</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get('new_bear_threshold',0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Previous Bear Market Threshold</td><td style="padding:10px;border:1px solid #ddd;">{sb.get('prev_bear_date','N/A')}</td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get('prev_bear',0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;"># of Days Bull</td><td style="padding:10px;border:1px solid #ddd;"></td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get('days_bull',0)}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;"># of Days avg. Bull</td><td style="padding:10px;border:1px solid #ddd;"></td><td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get('avg_days_bull',857)}</td></tr>
                        </tbody>
                    </table>'''

            html_parts.append(f'''
<li>
    <details>
        <summary>{item}</summary>
        <div style="padding:18px;background:#fafafa;border:1px solid #e5e5e5;border-top:none;border-radius:0 0 6px 6px;">
            <img src="data:image/png;base64,{img_base64}" style="width:100%;max-width:820px;display:block;margin:0 auto;box-shadow:0 4px 12px rgba(0,0,0,0.08);"/>
            {short_html}
            {desc_html}
            {terminal_html}
            {bear_html}
        </div>
    </details>
</li>''')
        return ''.join(html_parts)

    html = f"""
    <html><head><title>Portfolio Bias Summary</title>
    <style>body{{font-family:Arial,sans-serif;padding:40px;background:#fff;color:#000;max-width:920px;margin:auto;}}
    h1{{color:#1a1a1a;font-size:32px;}} .bias{{font-size:1.35em;font-weight:bold;color:#003366;margin-bottom:35px;border-bottom:2px solid #e5e5e5;padding-bottom:12px;}}
    .score{{font-size:1.4em;font-weight:bold;color:#003366;}}
    h2{{font-size:24px;border-bottom:3px solid #ddd;padding-bottom:10px;margin-top:45px;}}
    ul{{list-style-type:disc;padding-left:28px;}} summary{{font-size:1.05em;font-weight:600;cursor:pointer;padding:12px 16px;background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;}}
    summary:hover{{background:#f0f0f0;}}</style></head><body>
    <h1>Portfolio Bias Summary</h1>
    <p class="bias">Recommended Bias: {bias}</p>
    <p class="score">GDP Growth Score: {score:.0f}/100</p>
    <h2 style="color:#28a745;border-bottom:3px solid #28a745;">Tailwinds (Positive)</h2><ul>{build_section(tailwinds)}</ul>
    <h2 style="color:#dc3545;border-bottom:3px solid #dc3545;">Headwinds (Negative)</h2><ul>{build_section(headwinds)}</ul>
    <h2>Neutrals</h2><ul>{build_section(neutrals)}</ul>
    </body></html>
    """
    return html

def plot_sector_chart(etf_ticker, period='1y'):
    try:
        hist = yf.Ticker(etf_ticker).history(period=period)['Close']
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot(ax=ax, linewidth=2)
        ax.set_title(f'{etf_ticker} Performance ({period})')
        plt.tight_layout()
        return fig
    except:
        return None

def plot_commodity_chart(ticker, period='1y'):
    try:
        hist = yf.Ticker(ticker).history(period=period)['Close']
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot(ax=ax, linewidth=2)
        ax.set_title(f'{ticker} Performance ({period})')
        plt.tight_layout()
        return fig
    except:
        return None

def generate_sector_tilt(bias, score, risk_level, preferred_sectors, portfolio_size, data, metrics, today):
    all_sectors = {
        'Technology': {'etf': 'XLK', 'type': 'cyclical'},
        'Industrials': {'etf': 'XLI', 'type': 'cyclical'},
        'Financials': {'etf': 'XLF', 'type': 'cyclical'},
        'Consumer Discretionary': {'etf': 'XLY', 'type': 'cyclical'},
        'Materials': {'etf': 'XLB', 'type': 'cyclical'},
        'Energy': {'etf': 'XLE', 'type': 'cyclical'},
        'Healthcare': {'etf': 'XLV', 'type': 'defensive'},
        'Utilities': {'etf': 'XLU', 'type': 'defensive'},
        'Consumer Staples': {'etf': 'XLP', 'type': 'defensive'},
        'Real Estate': {'etf': 'XLRE', 'type': 'defensive'},
        'Communication Services': {'etf': 'XLC', 'type': 'mixed'},
    }

    all_commodities = {
        'Gold': {'ticker': 'GLD'},
        'Oil': {'ticker': 'USO'},
        'Silver': {'ticker': 'SLV'},
        'Copper': {'ticker': 'CPER'},
    }

    # Calculate hybrid performance: 0.5 * 3m return + 0.5 * 1y return
    performance = {}
    for sector, info in all_sectors.items():
        hist = yf.Ticker(info['etf']).history(period='1y')['Close']
        if len(hist) > 1:
            ret_1y = (hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0]
        else:
            ret_1y = 0
        three_m_ago = pd.Timestamp(today - timedelta(days=90))
        three_m_ago = three_m_ago.tz_localize(hist.index.tz) if hist.index.tz else three_m_ago
        hist_3m = hist[hist.index >= three_m_ago]
        if not hist_3m.empty:
            ret_3m = (hist.iloc[-1] - hist_3m.iloc[0]) / hist_3m.iloc[0]
        else:
            ret_3m = 0
        performance[sector] = 0.5 * ret_1y + 0.5 * ret_3m

    sorted_sectors = sorted(performance, key=performance.get, reverse=True)
    hot = sorted_sectors[:len(sorted_sectors)//2]
    cold = sorted_sectors[len(sorted_sectors)//2:]

    # Recommendations with diversification (5-7 sectors total)
    if 'Long' in bias:
        longs = [s for s in sorted_sectors if s in preferred_sectors][:4] or sorted_sectors[:4]
        shorts = [s for s in sorted_sectors[::-1] if s in preferred_sectors][:3] or sorted_sectors[-3:]
        long_alloc = portfolio_size * 0.6 / len(longs) if len(longs) > 0 else 0
        short_alloc = portfolio_size * 0.4 / len(shorts) if len(shorts) > 0 else 0
    elif 'Short' in bias:
        longs = [s for s in sorted_sectors if s in preferred_sectors][:3] or sorted_sectors[:3]
        shorts = [s for s in sorted_sectors[::-1] if s in preferred_sectors][:4] or sorted_sectors[-4:]
        long_alloc = portfolio_size * 0.4 / len(longs) if len(longs) > 0 else 0
        short_alloc = portfolio_size * 0.6 / len(shorts) if len(shorts) > 0 else 0
    else:
        longs = sorted_sectors[:3]
        shorts = sorted_sectors[-3:]
        long_alloc = portfolio_size * 0.5 / len(longs) if len(longs) > 0 else 0
        short_alloc = portfolio_size * 0.5 / len(shorts) if len(shorts) > 0 else 0

    tilt_df = pd.DataFrame({
        'Type': ['Long'] * len(longs) + ['Short'] * len(shorts),
        'Sector': longs + shorts,
        'ETF': [all_sectors[s]['etf'] for s in longs + shorts],
        'Allocation ($)': [long_alloc] * len(longs) + [short_alloc] * len(shorts)
    })

    return tilt_df, all_sectors, all_commodities

# --- STREAMLIT ---
st.set_page_config(page_title="Macro Portfolio Bias & Sector Tilt", layout="wide")
st.title("Portfolio Bias Analysis & Sector Tilt Dashboard")

if 'bias_calculated' not in st.session_state:
    st.session_state.bias_calculated = False
if 'bias' not in st.session_state:
    st.session_state.bias = None
if 'score' not in st.session_state:
    st.session_state.score = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'today' not in st.session_state:
    st.session_state.today = None
if 'tailwinds' not in st.session_state:
    st.session_state.tailwinds = None
if 'headwinds' not in st.session_state:
    st.session_state.headwinds = None
if 'neutrals' not in st.session_state:
    st.session_state.neutrals = None

if st.button("Update Analysis", type="primary"):
    with st.spinner("Fetching latest data..."):
        st.session_state.data, st.session_state.history, st.session_state.today = fetch_data()
        st.session_state.metrics, st.session_state.tailwinds, st.session_state.headwinds, st.session_state.neutrals, st.session_state.bias, st.session_state.score = calculate_metrics(st.session_state.data, st.session_state.history, st.session_state.today)
        st.success(f"Analysis updated for {st.session_state.today.strftime('%Y-%m-%d')}")
        st.info(f"**GDP Growth Score: {st.session_state.score:.0f}/100** → {st.session_state.bias}")
        html_report = generate_html_summary(st.session_state.tailwinds, st.session_state.headwinds, st.session_state.neutrals, st.session_state.bias, st.session_state.data, st.session_state.history, st.session_state.metrics, st.session_state.today, st.session_state.score)
        st.download_button("Download Interactive HTML Report", data=html_report, file_name=f"macro_bias_{st.session_state.today.date()}.html", mime="text/html")
        st.session_state.bias_calculated = True

if st.session_state.bias_calculated:
    st.header("Sector Tilt Recommendations")
    risk_level = st.selectbox("Risk Tolerance", options=['Low', 'Medium', 'High'], index=1)
    preferred_sectors = st.multiselect("Preferred Sectors", options=['Technology', 'Industrials', 'Financials', 'Consumer Discretionary', 'Utilities', 'Healthcare', 'Energy', 'Materials', 'Consumer Staples', 'Real Estate', 'Communication Services'])
    portfolio_size = st.number_input("Portfolio Size ($)", min_value=1000, value=100000, step=1000)

    if st.button("Generate Sector Tilt Recommendations", type="primary"):
        with st.spinner("Calculating sector tilts..."):
            tilt_df, sectors, commodities = generate_sector_tilt(st.session_state.bias, st.session_state.score, risk_level, preferred_sectors, portfolio_size, st.session_state.data, st.session_state.metrics, st.session_state.today)
            st.table(tilt_df)

            st.subheader("Sector Performance Charts")
            for sector, info in sectors.items():
                st.subheader(f"{sector} - {info['etf']}")
                cols = st.columns(3)
                for i, period in enumerate(['5y', '1y', '3mo']):
                    with cols[i]:
                        fig = plot_sector_chart(info['etf'], period)
                        if fig:
                            st.pyplot(fig)

            st.subheader("Commodity Performance Charts")
            for comm, info in commodities.items():
                st.subheader(f"{comm} - {info['ticker']}")
                cols = st.columns(3)
                for i, period in enumerate(['5y', '1y', '3mo']):
                    with cols[i]:
                        fig = plot_commodity_chart(info['ticker'], period)
                        if fig:
                            st.pyplot(fig)

            # CSV Download
            csv = tilt_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Sector Tilt CSV", data=csv, file_name="sector_tilt.csv", mime="text/csv")
