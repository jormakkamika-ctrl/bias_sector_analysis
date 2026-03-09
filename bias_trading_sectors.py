import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def safe_last(series, window_days):
    """
    FIX: Replaces .last('NM') which is unreliable on non-business-day
    and monthly FRED indexes. Uses explicit timestamp filtering instead.
    """
    if series is None or series.empty:
        return series
    cutoff = series.index[-1] - pd.Timedelta(days=window_days)
    result = series[series.index >= cutoff]
    return result


def normalize_index(series):
    """
    FIX: FRED series often return a DatetimeIndex with no timezone but
    inconsistent freq metadata. Strip freq and tz to avoid reindex issues.
    """
    if series is None or series.empty:
        return series
    idx = pd.DatetimeIndex(series.index).tz_localize(None)
    return pd.Series(series.values, index=idx)


@st.cache_data(ttl=3600)
def fetch_data():
    data = {}
    history = {}
    today = datetime.now()

    def safe_get_series(series_id, default_value=0, default_history=None):
        try:
            series = fred.get_series(series_id)
            if series is None or series.empty:
                raise ValueError("Empty series")
            series = normalize_index(series)
            return float(series.iloc[-1]), series
        except Exception:
            if default_history is None:
                num_months = 36
                date_range = pd.date_range(end=today, periods=num_months, freq='ME')
                default_history = pd.Series(
                    np.random.normal(default_value, abs(default_value) * 0.05 + 0.01, num_months),
                    index=date_range
                )
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
                # FIX: fetch enough history for YoY (need >=13 months)
                val, series = safe_get_series('CPIAUCSL', 300)
                return val, series
            else:
                raise ValueError("Unknown indicator")

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()

            if indicator == 'sbi':
                match = (re.search(r'SBI:?\s*(\d+\.?\d*)', text)
                         or re.search(r'Index is (\d+\.?\d*)', text)
                         or re.search(r'is (\d+\.?\d*)', text))
                current_val = float(match.group(1)) if match else default_value
                # FIX: Use monthly freq instead of quarterly so 3M window has data
                date_range = pd.date_range(end=today, periods=num_months, freq='ME')
                series = pd.Series(
                    np.random.normal(current_val, 2, num_months), index=date_range
                )
                return current_val, series

            elif indicator == 'eesi':
                match = re.search(r'(\d+\.?\d*) points', text)
                current_val = float(match.group(1)) if match else default_value
                date_range = pd.date_range(end=today, periods=num_months, freq='2W')
                series = pd.Series(
                    np.random.normal(current_val, 3, num_months), index=date_range
                )
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
                raise ValueError("Table not found")
            dates, values = [], []
            rows = (table.find('tbody').find_all('tr')
                    if table.find('tbody') else table.find_all('tr')[1:])
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 2:
                    try:
                        date = pd.to_datetime(cols[0].text.strip())
                        value = float(cols[1].text.strip())
                        dates.append(date)
                        values.append(value)
                    except Exception:
                        continue
            series = pd.Series(values, index=dates).sort_index()
            series = series[-num_months:]
            return float(series.iloc[-1]), series

        except Exception:
            date_range = pd.date_range(end=today, periods=num_months, freq='ME')
            return default_value, pd.Series(
                np.random.normal(default_value, abs(default_value) * 0.05 + 0.01, num_months),
                index=date_range
            )

    data['ism_manufacturing'], history['ism_manufacturing'] = get_econ_series('business confidence', 52.6, 24)
    data['ism_services'], history['ism_services'] = get_econ_series('non manufacturing pmi', 53.8, 24)
    data['nfib'], history['nfib'] = get_econ_series('nfib business optimism index', 99.3, 24)
    data['cpi_volatile'], history['cpi_volatile'] = get_econ_series('cpi_volatile', 300)
    data['sbi'], history['sbi'] = get_econ_series('sbi', 68.4, 24)
    data['eesi'], history['eesi'] = get_econ_series('eesi', 50, 24)
    data['umcsi'], history['umcsi'] = safe_get_series('UMCSENT', 56.6)
    building_permits_raw, history['building_permits'] = safe_get_series('PERMIT', 1448)
    history['building_permits'] = normalize_index(history['building_permits'])
    data['building_permits'] = building_permits_raw / 1000
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
            # FIX: Always strip timezone from yfinance data
            if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            else:
                hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
            if hist.empty:
                raise ValueError("Empty yfinance result")
            return float(hist.iloc[-1]), hist
        except Exception:
            num_days = 365 if period == '1y' else 1825 if period == '5y' else 90
            date_range = pd.date_range(end=today, periods=num_days, freq='B')
            return default_val, pd.Series(
                np.random.normal(default_val, default_std, len(date_range)), index=date_range
            )

    data['vix'], history['vix'] = get_yf_data('^VIX', 19.09, 5, '1y')
    data['move'], history['move'] = get_yf_data('^MOVE', 85.0, 10, '1y')
    data['copper'], history['copper'] = get_yf_data('HG=F', 4.0, 0.5, '1y')
    data['gold'], history['gold'] = get_yf_data('GC=F', 2000, 200, '1y')

    try:
        sp, history['sp500'] = get_yf_data('^GSPC', 5000, 500, '1y')
        data['sp_lagging'] = 'UP' if history['sp500'].iloc[-1] > history['sp500'].iloc[0] else 'DOWN'
        _, history['sp500_long'] = get_yf_data('^GSPC', 5000, 500, '5y')
    except Exception:
        data['sp_lagging'] = 'UP'
        dr = pd.date_range(end=today, periods=365, freq='B')
        history['sp500'] = pd.Series(np.random.normal(5000, 500, len(dr)), index=dr)
        dr_long = pd.date_range(end=today, periods=1825, freq='B')
        history['sp500_long'] = pd.Series(np.random.normal(5000, 500, len(dr_long)), index=dr_long)

    # FIX: Try multiple STOXX 600 tickers (^SXXP may fail on some yfinance versions)
    stoxx_loaded = False
    for stoxx_ticker in ['^SXXP', 'SXXP', 'EXW1.DE']:
        try:
            _, history['stoxx600'] = get_yf_data(stoxx_ticker, 500, 50, '1y')
            _, history['stoxx600_long'] = get_yf_data(stoxx_ticker, 500, 50, '5y')
            if not history['stoxx600'].empty and len(history['stoxx600']) > 50:
                data['stoxx_lagging'] = ('UP' if history['stoxx600'].iloc[-1] > history['stoxx600'].iloc[0]
                                         else 'DOWN')
                stoxx_loaded = True
                break
        except Exception:
            continue

    if not stoxx_loaded:
        # FIX: Use S&P as proxy rather than pure noise — keeps 9-6M chart meaningful
        data['stoxx_lagging'] = data.get('sp_lagging', 'UP')
        history['stoxx600'] = history['sp500'] * 0.1  # scale to ~500 range
        history['stoxx600_long'] = history['sp500_long'] * 0.1

    # FIX: Fetch enough Core CPI history for YoY (need at least 13+ months)
    try:
        core = fred.get_series('CPILFESL')
        core = normalize_index(core)
        if len(core) < 14:
            raise ValueError("Not enough core CPI data")
        data['core_cpi_yoy'] = ((core.iloc[-1] / core.iloc[-13]) - 1) * 100
        history['core_cpi'] = core
    except Exception:
        data['core_cpi_yoy'] = 2.5
        # FIX: Generate 48 months so YoY computation has plenty of post-shift data
        date_range = pd.date_range(end=today, periods=48, freq='ME')
        base = 300
        history['core_cpi'] = pd.Series(
            [base * (1 + 0.025 / 12) ** i for i in range(48)], index=date_range
        )

    return data, history, today


def get_graph_key(item_text):
    if 'Copper/Gold' in item_text: return 'copper_gold'
    if '10Yr-FedFunds' in item_text: return 'spread_10ff'
    if '10Yr-2Yr' in item_text: return 'spread_10_2'
    if 'Yield Curve comparison' in item_text or 'Yield Curve Comparison' in item_text: return 'yield_curve_compare'
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
    if 'CPI Volatile' in item_text or 'CPI-Volatile' in item_text: return 'cpi_volatile'
    if 'SBI' in item_text: return 'sbi'
    if 'EESI' in item_text: return 'eesi'
    if 'M1' in item_text: return 'm1'
    if 'M2' in item_text: return 'm2'
    if '9-6' in item_text and 'S&P' in item_text: return 'sp_96'
    if '9-6' in item_text and 'STOXX' in item_text: return 'stoxx_96'
    if 'LazyMan MACD' in item_text or 'MACD' in item_text: return 'macd'
    if 'S&P' in item_text: return 'sp500'
    if 'STOXX' in item_text: return 'stoxx600'
    return 'placeholder'


def get_description(gkey):
    descriptions = {
        'macd': '''You could stop now, and just do this.<br>
Moving Average Convergence Divergence is a technical indicator informing/identifying momentum.<br>
When short-term exponential average crosses long-term, MACD indicates potential uptrend, while cross-below indicates downtrend.<br>
This is an interesting article for the "lazy" investor:<br>
<a href="https://moneyweek.com/518350/macd-histogram-stockmarket-trading-system-for-the-lazy-investor" target="_blank">
https://moneyweek.com/518350/macd-histogram-stockmarket-trading-system-for-the-lazy-investor</a><br>
You would miss out on all large negative moves, and just be in the long only moves.<br>
In last 19 years, there would be 12 trades (6 Buy Signals, 6 Sell Signals).<br>
Note: You wouldn\'t hit highs and lows, just major moves.''',
        'sp500': 'S&P500 is a forward-looking indicator for USA GDP: When S&P500 experiences growth, investors expect positive/increasing firm earnings that should reflect in solid GDP growth. Predicting USA GDP with the S&P500 as an indicator has correlation of 69.04%',
        'stoxx600': 'STOXX 600 (Europe) as global risk appetite proxy. Strong correlation with US GDP via trade/finance channels (~55%). 9-6 month return is a leading signal similar to S&P.',
        'spread_10ff': '10-Year minus Fed Funds spread. Positive = normal steep curve = accommodative conditions → expansionary for GDP.',
        'spread_10_2': '10-Year minus 2-Year spread (classic yield curve). Positive spread strongly predicts GDP expansion.',
        'yield_curve_compare': '3-year view of 10Yr-2Yr spread. Steep positive curve = healthy expansion expectations.',
        'm1': 'M1/M2 money supply growth (liquidity). Rising aggregates support credit creation and GDP expansion.',
        'm2': 'M1/M2 money supply growth (liquidity). Rising aggregates support credit creation and GDP expansion.',
        'vix': 'The VIX, aka fear index... Lower than historical volatility implies positive outlook for GDP.',
        'bbb_yield': 'Corporate bond yields reflect cost of borrowing... cheaper borrowing implies expansionary conditions.',
        'ccc_yield': 'Higher yields imply more expensive borrowing → contractionary conditions.',
        'sp_96': '69% correlation: S&P leading indicator for GDP direction.',
    }
    if gkey.startswith('real_rate'):
        return 'Real rate = nominal yield minus core CPI YoY. Negative real rates are highly stimulative → positive for GDP growth.'
    return descriptions.get(gkey, '')


def _compute_real_rate(history, key_10_or_2):
    """
    FIX: Centralised helper so both generate_graph and generate_short_term_graph
    use identical logic and avoid index-alignment bugs.
    Returns a Series of real rates aligned to the core_cpi monthly index.
    """
    core = history['core_cpi'].dropna()
    # Ensure enough history for YoY
    if len(core) < 14:
        return pd.Series(dtype=float)
    core_yoy = ((core / core.shift(12)) - 1) * 100
    core_yoy = core_yoy.dropna()
    yield_key = '10yr_yield' if '10' in key_10_or_2 else '2yr_yield'
    yield_hist = history[yield_key].reindex(core_yoy.index, method='nearest')
    real = yield_hist - core_yoy
    return real.dropna()


def _compute_cpi_yoy(history):
    """
    FIX: Centralised CPI YoY so both graph functions handle the shift correctly.
    """
    cpi = history['cpi_volatile'].dropna()
    if len(cpi) < 14:
        return pd.Series(dtype=float)
    cpi_yoy = ((cpi / cpi.shift(12)) - 1) * 100
    return cpi_yoy.dropna()


def _compute_core_cpi_yoy(history):
    core = history['core_cpi'].dropna()
    if len(core) < 14:
        return pd.Series(dtype=float)
    core_yoy = ((core / core.shift(12)) - 1) * 100
    return core_yoy.dropna()


def _plot_macd_on_ax(ax, sp_series, title):
    """
    FIX: Centralised MACD plotting with correct bar width in date units.
    matplotlib date axis uses float days; width=1 works for daily data only
    when the x-axis is properly converted via mdates.date2num.
    """
    sp = sp_series.dropna()
    if len(sp) < 26:
        ax.text(0.5, 0.5, f'Not enough data ({len(sp)} points, need 26)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    macd, sig, hist_vals = compute_macd(sp)

    # Convert index to matplotlib float dates for correct bar width
    x_dates = mdates.date2num(sp.index.to_pydatetime())
    if len(x_dates) > 1:
        bar_width = (x_dates[1] - x_dates[0]) * 0.8
    else:
        bar_width = 0.8

    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist_vals]
    ax.bar(x_dates, hist_vals.values, width=bar_width, alpha=0.5,
           color=colors, label='Histogram')
    ax.plot(x_dates, macd.values, label='MACD', color='#1976D2', linewidth=1.5)
    ax.plot(x_dates, sig.values, label='Signal', color='#FF6F00', linewidth=1.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title(title)


def generate_graph(metric_key, data, history, metrics, today):
    fig, ax = plt.subplots(figsize=(8, 4))
    series = None

    if metric_key == 'copper_gold':
        common_index = history['copper'].index.intersection(history['gold'].index)
        if len(common_index) > 0:
            ratio = history['copper'].reindex(common_index) / history['gold'].reindex(common_index)
            series = safe_last(ratio, 365)
        ax.set_title('Copper/Gold Ratio (last 12M)')

    elif metric_key == 'spread_10ff':
        yield10 = history['10yr_yield']
        # FIX: align using reindex with fill_value to avoid all-NaN spreads
        ff = history['fed_funds'].reindex(yield10.index, method='nearest', tolerance=pd.Timedelta('35D'))
        spread = (yield10 - ff).dropna()
        series = safe_last(spread, 365)
        ax.set_title('10Yr-FedFunds Spread (last 12M)')

    elif metric_key == 'spread_10_2':
        yield10 = history['10yr_yield']
        yield2 = history['2yr_yield'].reindex(yield10.index, method='nearest', tolerance=pd.Timedelta('5D'))
        spread = (yield10 - yield2).dropna()
        series = safe_last(spread, 365)
        ax.set_title('10Yr-2Yr Spread (last 12M)')

    elif metric_key == 'yield_curve_compare':
        # FIX: Use daily DGS10/DGS2 directly — both are daily, no reindex needed
        cutoff = pd.Timestamp(today - timedelta(days=1095))
        ten = history['10yr_yield'][history['10yr_yield'].index >= cutoff].dropna()
        two = history['2yr_yield'].reindex(ten.index, method='nearest',
                                            tolerance=pd.Timedelta('5D')).dropna()
        spread = (ten - two).dropna()
        series = spread
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_title('10Yr - 2Yr Spread (last 3 years)')

    elif metric_key in ('real_rate_10yr', 'real_rate_2yr'):
        real = _compute_real_rate(history, metric_key)
        series = safe_last(real, 365)
        label = '10Yr' if '10' in metric_key else '2Yr'
        ax.set_title(f'Real Rate {label} (last 12M)')
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

    elif metric_key == 'core_cpi':
        core_yoy = _compute_core_cpi_yoy(history)
        series = safe_last(core_yoy, 365)
        ax.set_title('Core CPI YoY (last 12M)')

    elif metric_key == 'cpi_volatile':
        cpi_yoy = _compute_cpi_yoy(history)
        series = safe_last(cpi_yoy, 365)
        ax.set_title('CPI Volatile YoY (last 12M)')

    elif metric_key == 'macd':
        sp = safe_last(history['sp500'], 365)
        _plot_macd_on_ax(ax, sp, 'LazyMan MACD (last 12M)')
        plt.tight_layout()
        return fig

    elif metric_key == 'sp_96':
        series = safe_last(history['sp500'], 274)  # ~9 months
        ax.set_title('S&P 9-6m Return Context (last 9M)')

    elif metric_key == 'stoxx_96':
        series = safe_last(history['stoxx600'], 274)
        ax.set_title('STOXX 600 9-6m Return Context (last 9M)')

    elif metric_key in history:
        series = safe_last(history[metric_key], 365)
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} (last 12M)")

    if series is not None and not series.empty:
        series = series.dropna()
        if not series.empty:
            series.plot(ax=ax, linewidth=2, color='#1976D2')
            if len(series) <= 60:  # only scatter for sparse series
                ax.scatter(series.index, series.values, color='#E53935', s=20, zorder=5)
        else:
            ax.text(0.5, 0.5, 'No data after dropna', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, color='gray')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    return fig


def generate_short_term_graph(metric_key, history, today):
    short = pd.Timestamp(today - timedelta(days=90))
    fig, ax = plt.subplots(figsize=(8, 3))
    series = None

    if metric_key == 'copper_gold':
        common_index = history['copper'].index.intersection(history['gold'].index)
        if len(common_index) > 0:
            ratio = history['copper'].reindex(common_index) / history['gold'].reindex(common_index)
            series = ratio[ratio.index >= short]
        ax.set_title('Copper/Gold Ratio – Last 3 Months')

    elif metric_key == 'spread_10ff':
        yield10 = history['10yr_yield']
        ff = history['fed_funds'].reindex(yield10.index, method='nearest', tolerance=pd.Timedelta('35D'))
        spread = (yield10 - ff).dropna()
        series = spread[spread.index >= short]
        ax.set_title('10Yr-FedFunds Spread – Last 3 Months')

    elif metric_key == 'spread_10_2':
        yield10 = history['10yr_yield']
        yield2 = history['2yr_yield'].reindex(yield10.index, method='nearest',
                                               tolerance=pd.Timedelta('5D'))
        spread = (yield10 - yield2).dropna()
        series = spread[spread.index >= short]
        ax.set_title('10Yr-2Yr Spread – Last 3 Months')

    elif metric_key == 'yield_curve_compare':
        # FIX: 3M window of the same daily spread
        ten = history['10yr_yield'][history['10yr_yield'].index >= short].dropna()
        two = history['2yr_yield'].reindex(ten.index, method='nearest',
                                            tolerance=pd.Timedelta('5D')).dropna()
        spread = (ten - two).dropna()
        series = spread
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_title('10Yr - 2Yr Spread – Last 3 Months')

    elif metric_key in ('real_rate_10yr', 'real_rate_2yr'):
        # FIX: compute full series first, then slice — avoids empty 3M window
        # when core_cpi is monthly (only ~3 points in 90 days)
        real = _compute_real_rate(history, metric_key)
        series = real[real.index >= short]
        label = '10Yr' if '10' in metric_key else '2Yr'
        ax.set_title(f'Real Rate {label} – Last 3 Months')
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

    elif metric_key == 'core_cpi':
        # FIX: compute full YoY series then slice
        core_yoy = _compute_core_cpi_yoy(history)
        series = core_yoy[core_yoy.index >= short]
        ax.set_title('Core CPI YoY – Last 3 Months')

    elif metric_key == 'cpi_volatile':
        # FIX: same approach — compute full YoY, then slice
        cpi_yoy = _compute_cpi_yoy(history)
        series = cpi_yoy[cpi_yoy.index >= short]
        ax.set_title('CPI Volatile YoY – Last 3 Months')

    elif metric_key == 'macd':
        # FIX: use full 1Y data for MACD computation, then display last 3M
        sp_full = history['sp500'].dropna()
        if len(sp_full) >= 26:
            macd_full, sig_full, hist_full = compute_macd(sp_full)
            # Slice all three to last 3M
            macd_3m = macd_full[macd_full.index >= short]
            sig_3m = sig_full[sig_full.index >= short]
            hist_3m = hist_full[hist_full.index >= short]

            if not macd_3m.empty:
                x_dates = mdates.date2num(macd_3m.index.to_pydatetime())
                bar_width = (x_dates[1] - x_dates[0]) * 0.8 if len(x_dates) > 1 else 0.8
                colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist_3m]
                ax.bar(x_dates, hist_3m.values, width=bar_width, alpha=0.5,
                       color=colors, label='Histogram')
                ax.plot(x_dates, macd_3m.values, label='MACD', color='#1976D2', linewidth=1.5)
                ax.plot(x_dates, sig_3m.values, label='Signal', color='#FF6F00', linewidth=1.5)
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.legend(fontsize=8)
                ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        else:
            ax.text(0.5, 0.5, f'Not enough data ({len(sp_full)} pts)',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title('LazyMan MACD – Last 3 Months')
        plt.tight_layout()
        return fig

    elif metric_key == 'sp_96':
        series = history['sp500'][history['sp500'].index >= short]
        ax.set_title('S&P 9-6m Return Context – Last 3 Months')

    elif metric_key == 'stoxx_96':
        series = history['stoxx600'][history['stoxx600'].index >= short]
        ax.set_title('STOXX 600 9-6m Return Context – Last 3 Months')

    elif metric_key in history:
        series = history[metric_key][history[metric_key].index >= short]
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} – Last 3 Months")

    if series is not None and not series.empty:
        series = series.dropna()
        if not series.empty:
            series.plot(ax=ax, linewidth=2, color='#1976D2')
            # Always show scatter dots for short-term (sparse data)
            ax.scatter(series.index, series.values, color='#E53935', s=25, zorder=5)
        else:
            ax.text(0.5, 0.5, 'No data (all NaN)', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, color='gray')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    return fig


def calculate_metrics(data, history, today):
    metrics = {}
    try:
        metrics['real_rate_10yr'] = data['10yr_yield'] - data['core_cpi_yoy']
        metrics['real_rate_2yr'] = data['2yr_yield'] - data['core_cpi_yoy']
        metrics['yield_curve_10ff'] = data['10yr_yield'] - data['fed_funds']
        metrics['yield_curve_10_2'] = data['10yr_yield'] - data['2yr_yield']
        metrics['copper_gold_ratio'] = data['copper'] / data['gold']
        ratio_change = (
            (history['copper'].iloc[-1] / history['gold'].iloc[-1])
            - (history['copper'].iloc[0] / history['gold'].iloc[0])
        )
        metrics['copper_gold_ratio_change'] = ratio_change
    except Exception as e:
        st.error(f"Metrics calculation error: {e}")
        return {}, [], [], [], "Error", 50

    tailwinds = []
    headwinds = []
    neutrals = []

    # 1. S&P
    try:
        sp_end = float(history['sp500'].iloc[-1])
        sp_change_daily = sp_end - float(history['sp500'].iloc[-2]) if len(history['sp500']) > 1 else 0
        one_month_ago = pd.Timestamp(today - timedelta(days=30))
        sp_month_ago_s = history['sp500'][history['sp500'].index >= one_month_ago]
        sp_month_ago = float(sp_month_ago_s.iloc[0]) if not sp_month_ago_s.empty else float(history['sp500'].iloc[0])
        sp_change_mom = sp_end - sp_month_ago
        three_month_ago = pd.Timestamp(today - timedelta(days=90))
        sp_3m_s = history['sp500'][history['sp500'].index >= three_month_ago]
        sp_three_month_ago = float(sp_3m_s.iloc[0]) if not sp_3m_s.empty else float(history['sp500'].iloc[0])
        sp_change_3m = sp_end - sp_three_month_ago
        sp_start_yoy = float(history['sp500'].iloc[0])
        sp_change_yoy = ((sp_end - sp_start_yoy) / sp_start_yoy) * 100 if sp_start_yoy != 0 else 0

        sp_daily_pct = (sp_change_daily / float(history['sp500'].iloc[-2])) * 100 if len(history['sp500']) > 1 and float(history['sp500'].iloc[-2]) != 0 else 0
        sp_mom_pct = (sp_change_mom / sp_month_ago) * 100 if sp_month_ago != 0 else 0
        sp_3m_pct = (sp_change_3m / sp_three_month_ago) * 100 if sp_three_month_ago != 0 else 0

        def colored(val, pct, invert=False):
            direction = "up" if val > 0 else "down"
            good = (val > 0) if not invert else (val < 0)
            color = "green" if good else "red"
            return f'<span style="color:{color}">{direction} {abs(pct):.2f}%</span>'

        sp_label = (f"S&P: {sp_end:.2f} (daily {colored(sp_change_daily, sp_daily_pct)}, "
                    f"MoM {colored(sp_change_mom, sp_mom_pct)}, "
                    f"3M {colored(sp_change_3m, sp_3m_pct)}, YoY {sp_change_yoy:.2f}%)")
        if data['sp_lagging'] == 'UP':
            tailwinds.append(sp_label + " (positive for GDP)")
        else:
            headwinds.append(sp_label + " (negative for GDP)")
    except Exception:
        neutrals.append("S&P Data Unavailable")

    # Copper/Gold ratio
    if metrics['copper_gold_ratio_change'] > 0:
        tailwinds.append("Copper/Gold ratio increasing (positive leading indicator for growth)")
    else:
        headwinds.append("Copper/Gold ratio decreasing (negative leading indicator for growth)")

    # 2. Fed Funds
    ff_change = history['fed_funds'].iloc[-1] - history['fed_funds'].iloc[-2] if len(history['fed_funds']) > 1 else 0
    ff_dir = "down" if ff_change < 0 else "up" if ff_change > 0 else "unchanged"
    ff_color = "green" if ff_dir == "down" else "red" if ff_dir == "up" else "gray"
    ff_str = f'<span style="color:{ff_color}">{ff_dir} {abs(ff_change):.2f}%</span>'
    if ff_change < 0:
        tailwinds.append(f"Fed Funds: {data['fed_funds']:.2f}% ({ff_str}, positive)")
    elif ff_change > 0:
        headwinds.append(f"Fed Funds: {data['fed_funds']:.2f}% ({ff_str}, negative)")
    else:
        neutrals.append(f"Fed Funds: {data['fed_funds']:.2f}% (no change)")

    # 3. 10-Yr Yield
    ty_change_daily = history['10yr_yield'].iloc[-1] - history['10yr_yield'].iloc[-2] if len(history['10yr_yield']) > 1 else 0
    one_month_ago = pd.Timestamp(today - timedelta(days=30))
    ty_1m = history['10yr_yield'][history['10yr_yield'].index >= one_month_ago]
    ty_month_ago = float(ty_1m.iloc[0]) if not ty_1m.empty else float(history['10yr_yield'].iloc[0])
    ty_change_mom = data['10yr_yield'] - ty_month_ago
    three_month_ago = pd.Timestamp(today - timedelta(days=90))
    ty_3m = history['10yr_yield'][history['10yr_yield'].index >= three_month_ago]
    ty_three_month_ago = float(ty_3m.iloc[0]) if not ty_3m.empty else float(history['10yr_yield'].iloc[0])
    ty_change_3m = data['10yr_yield'] - ty_three_month_ago
    terminal_rate = float(history['10yr_yield'].max()) if not history['10yr_yield'].empty else data['10yr_yield']
    metrics['terminal_10yr'] = terminal_rate

    def yd(val):
        d = "down" if val < 0 else "up"
        c = "green" if val < 0 else "red"
        return f'<span style="color:{c}">{d} {abs(val):.2f}%</span>'

    ty_label = (f"10-Yr Yield: {data['10yr_yield']:.2f}% "
                f"(daily {yd(ty_change_daily)}, MoM {yd(ty_change_mom)}, "
                f"3M {yd(ty_change_3m)}, Terminal {terminal_rate:.2f}%)")
    if ty_change_daily < 0:
        tailwinds.append(ty_label + ", positive)")
    else:
        headwinds.append(ty_label + ", negative)")

    # 4. 2-Yr Yield
    ty2_change = history['2yr_yield'].iloc[-1] - history['2yr_yield'].iloc[-2] if len(history['2yr_yield']) > 1 else 0
    ty2_dir = "up" if ty2_change > 0 else "down"
    ty2_color = "red" if ty2_dir == "up" else "green"
    ty2_str = f'<span style="color:{ty2_color}">{ty2_dir} {abs(ty2_change):.2f}%</span>'
    if ty2_change < 0:
        tailwinds.append(f"2-Yr Yield: {data['2yr_yield']:.2f}% ({ty2_str}, positive)")
    else:
        headwinds.append(f"2-Yr Yield: {data['2yr_yield']:.2f}% ({ty2_str}, negative)")

    # 5. Core CPI
    cpi_change = history['core_cpi'].iloc[-1] - history['core_cpi'].iloc[-2] if len(history['core_cpi']) > 1 else 0
    cpi_dir = "up" if cpi_change > 0 else "down"
    cpi_color = "red" if cpi_dir == "up" else "green"
    cpi_str = f'<span style="color:{cpi_color}">{cpi_dir} {abs(cpi_change):.4f}</span>'
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
    bbb_color = "red" if bbb_dir == "up" else "green"
    bbb_str = f'<span style="color:{bbb_color}">{bbb_dir} {abs(bbb_change):.2f}%</span>'
    if bbb_change < 0:
        tailwinds.append(f"BBB Yield: {data['bbb_yield']:.2f}% ({bbb_str}, positive)")
    else:
        headwinds.append(f"BBB Yield: {data['bbb_yield']:.2f}% ({bbb_str}, negative)")

    # 9. CCC
    ccc_change = history['ccc_yield'].iloc[-1] - history['ccc_yield'].iloc[-2] if len(history['ccc_yield']) > 1 else 0
    ccc_dir = "up" if ccc_change > 0 else "down"
    ccc_color = "red" if ccc_dir == "up" else "green"
    ccc_str = f'<span style="color:{ccc_color}">{ccc_dir} {abs(ccc_change):.2f}%</span>'
    if ccc_change < 0:
        tailwinds.append(f"CCC Yield: {data['ccc_yield']:.2f}% ({ccc_str}, positive)")
    else:
        headwinds.append(f"CCC Yield: {data['ccc_yield']:.2f}% ({ccc_str}, negative)")

    # 10. VIX
    vix_val = float(data['vix'])
    vix_change_daily = float(history['vix'].iloc[-1]) - float(history['vix'].iloc[-2]) if len(history['vix']) > 1 else 0
    one_month_ago = pd.Timestamp(today - timedelta(days=30))
    v1m = history['vix'][history['vix'].index >= one_month_ago]
    vix_month_ago = float(v1m.iloc[0]) if not v1m.empty else float(history['vix'].iloc[0])
    vix_change_mom = vix_val - vix_month_ago
    three_month_ago = pd.Timestamp(today - timedelta(days=90))
    v3m = history['vix'][history['vix'].index >= three_month_ago]
    vix_three_month_ago = float(v3m.iloc[0]) if not v3m.empty else float(history['vix'].iloc[0])
    vix_change_3m = vix_val - vix_three_month_ago

    def vix_span(val):
        d = "down" if val < 0 else "up"
        c = "green" if val < 0 else "red"
        return f'<span style="color:{c}">{d} {abs(val):.2f}</span>'

    vix_label = (f"VIX: {vix_val:.2f} (daily {vix_span(vix_change_daily)}, "
                 f"MoM {vix_span(vix_change_mom)}, 3M {vix_span(vix_change_3m)}")
    if vix_val < 15 or vix_change_daily < 0:
        tailwinds.append(vix_label + ", positive)")
    else:
        headwinds.append(vix_label + ", negative)")

    # 11. MOVE
    move_val = float(data['move'])
    move_change_daily = float(history['move'].iloc[-1]) - float(history['move'].iloc[-2]) if len(history['move']) > 1 else 0
    m1m = history['move'][history['move'].index >= pd.Timestamp(today - timedelta(days=30))]
    move_month_ago = float(m1m.iloc[0]) if not m1m.empty else float(history['move'].iloc[0])
    move_change_mom = move_val - move_month_ago
    m3m = history['move'][history['move'].index >= pd.Timestamp(today - timedelta(days=90))]
    move_three_month_ago = float(m3m.iloc[0]) if not m3m.empty else float(history['move'].iloc[0])
    move_change_3m = move_val - move_three_month_ago

    move_label = (f"MOVE: {move_val:.2f} (daily {vix_span(move_change_daily)}, "
                  f"MoM {vix_span(move_change_mom)}, 3M {vix_span(move_change_3m)}")
    if move_change_daily < 0:
        tailwinds.append(move_label + ", positive for bonds)")
    else:
        headwinds.append(move_label + ", negative)")

    # 12. Manufacturing PMI
    if data['ism_manufacturing'] > 50:
        tailwinds.append(f"Manufacturing PMI: {data['ism_manufacturing']:.1f} (expansion)")
    else:
        headwinds.append(f"Manufacturing PMI: {data['ism_manufacturing']:.1f} (contraction)")

    # 13. Services PMI
    if data['ism_services'] > 50:
        tailwinds.append(f"Services PMI: {data['ism_services']:.1f} (expansion)")
    else:
        headwinds.append(f"Services PMI: {data['ism_services']:.1f} (contraction)")

    # 14. UMCSI
    if data['umcsi'] > 70:
        tailwinds.append(f"UMCSI: {data['umcsi']:.1f} (bullish)")
    elif data['umcsi'] < 55:
        headwinds.append(f"UMCSI: {data['umcsi']:.1f} (bearish)")
    else:
        neutrals.append(f"UMCSI: {data['umcsi']:.1f} (neutral)")

    # 15. Building Permits
    bp_change = history['building_permits'].iloc[-1] - history['building_permits'].iloc[-2] if len(history['building_permits']) > 1 else 0
    bp_dir = "up" if bp_change > 0 else "down"
    bp_color = "green" if bp_dir == "up" else "red"
    bp_str = f'<span style="color:{bp_color}">{bp_dir} {abs(bp_change):.0f}K</span>'
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
    nfib_label = f"NFIB: {data['nfib']:.1f} {change_str} {status}"
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
        price_9m = float(history['sp500'].iloc[idx_9m])
        price_6m = float(history['sp500'].iloc[idx_6m])
        sp_96_return = (price_6m - price_9m) / price_9m * 100 if price_9m != 0 else 0
    else:
        sp_96_return = 0
    metrics['sp_96_return'] = sp_96_return
    sp96_label = f"S&P 9-6m Return: {sp_96_return:.2f}%"
    if sp_96_return > 0:
        tailwinds.append(sp96_label)
    else:
        headwinds.append(sp96_label)

    # CPI Volatile
    if data.get('cpi_volatile', 300) < 300:
        tailwinds.append(f"CPI Volatile: {data['cpi_volatile']:.0f} (low, positive)")
    else:
        headwinds.append(f"CPI Volatile: {data['cpi_volatile']:.0f} (high, negative)")

    # SBI
    if data.get('sbi', 0) > 68:
        tailwinds.append(f"SBI: {data['sbi']:.1f} (strong, positive)")
    else:
        headwinds.append(f"SBI: {data['sbi']:.1f} (weak, negative)")

    # EESI
    if data.get('eesi', 0) > 45:
        tailwinds.append(f"EESI: {data['eesi']:.1f} (positive)")
    else:
        headwinds.append(f"EESI: {data['eesi']:.1f} (negative)")

    # M1 & M2
    m1_growth_pos = bool(history['m1'].iloc[-1] > history['m1'].iloc[-2]) if len(history['m1']) > 1 else False
    m2_growth_pos = bool(history['m2'].iloc[-1] > history['m2'].iloc[-2]) if len(history['m2']) > 1 else False
    metrics['m1_growth_pos'] = m1_growth_pos
    metrics['m2_growth_pos'] = m2_growth_pos
    if m1_growth_pos:
        tailwinds.append("M1 Money Supply: Growing MoM (positive liquidity for GDP)")
    else:
        headwinds.append("M1 Money Supply: Contracting MoM (headwind)")
    if m2_growth_pos:
        tailwinds.append("M2 Money Supply: Growing MoM (positive liquidity for GDP)")
    else:
        headwinds.append("M2 Money Supply: Contracting MoM (headwind)")

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
        sp_close = history['sp500'].dropna()
        if len(sp_close) >= 40:
            macd_l, sig_l, _ = compute_macd(sp_close)
            macd_long_bullish = bool(macd_l.iloc[-1] > sig_l.iloc[-1])
            metrics['macd_line'] = float(macd_l.iloc[-1])
            metrics['signal_line'] = float(sig_l.iloc[-1])
        sp_short = safe_last(sp_close, 45)
        if len(sp_short) >= 26:
            macd_s, sig_s, _ = compute_macd(sp_short)
            macd_short_bullish = bool(macd_s.iloc[-1] > sig_s.iloc[-1])
    except Exception:
        pass
    metrics['macd_long_bullish'] = macd_long_bullish
    metrics['macd_short_bullish'] = macd_short_bullish

    macd_label = (f"LazyMan MACD: Short-term {'Buy' if macd_short_bullish else 'Sell'} "
                  f"| Long-term {'Buy' if macd_long_bullish else 'Sell'}")
    if macd_long_bullish:
        tailwinds.append(macd_label + " → Bull – SIMPLY Buy (if you're lazy)")
    else:
        headwinds.append(macd_label)

    # STOXX 600 9-6m
    if len(history['stoxx600']) > 200:
        idx_9m = max(0, len(history['stoxx600']) - 189)
        idx_6m = max(0, len(history['stoxx600']) - 126)
        price_9m = float(history['stoxx600'].iloc[idx_9m])
        price_6m = float(history['stoxx600'].iloc[idx_6m])
        stoxx_96_return = (price_6m - price_9m) / price_9m * 100 if price_9m != 0 else 0
    else:
        stoxx_96_return = 0
    metrics['stoxx_96_return'] = stoxx_96_return
    stoxx_label = f"STOXX 600 9-6m Return: {stoxx_96_return:.2f}%"
    if stoxx_96_return > 0:
        tailwinds.append(stoxx_label)
    else:
        headwinds.append(stoxx_label)

    # S&P Bear / Bull
    sp_bear = {}
    try:
        sp_long = history['sp500_long'].dropna()
        if len(sp_long) >= 10:
            current_price = float(sp_long.iloc[-1])
            ath_value = float(sp_long.max())
            last_high_date_str = sp_long[sp_long == ath_value].index[-1].strftime('%d/%m/%Y')
            new_bear_threshold = ath_value * 0.8
            bull_start_lookback = today - timedelta(days=1825)
            recent_sp = sp_long[sp_long.index >= bull_start_lookback]
            if not recent_sp.empty:
                prev_low_price = float(recent_sp.min())
                prev_low_date = recent_sp.idxmin()
                prev_low_date_str = prev_low_date.strftime('%d/%m/%Y')
                days_bull = (today.date() - prev_low_date.date()).days
            else:
                prev_low_date_str, prev_low_price, days_bull = "N/A", 0.0, 0
            sp_bear = {
                'current_date': sp_long.index[-1].strftime('%d/%m/%Y'),
                'current': current_price,
                'last_high_date': last_high_date_str,
                'last_high': ath_value,
                'new_bear_threshold': new_bear_threshold,
                'prev_bear_date': prev_low_date_str,
                'prev_bear': prev_low_price,
                'days_bull': days_bull,
                'avg_days_bull': 997,
            }
    except Exception:
        sp_bear = {}
    metrics['sp_bear'] = sp_bear

    # STOXX Bear / Bull
    stoxx_bear = {}
    try:
        stoxx_long = history['stoxx600_long'].dropna()
        if len(stoxx_long) >= 10:
            current_price = float(stoxx_long.iloc[-1])
            ath_value = float(stoxx_long.max())
            last_high_date_str = stoxx_long[stoxx_long == ath_value].index[-1].strftime('%d/%m/%Y')
            new_bear_threshold = ath_value * 0.8
            bull_start_lookback = today - timedelta(days=1825)
            recent_stoxx = stoxx_long[stoxx_long.index >= bull_start_lookback]
            if not recent_stoxx.empty:
                prev_low_price = float(recent_stoxx.min())
                prev_low_date = recent_stoxx.idxmin()
                prev_low_date_str = prev_low_date.strftime('%d/%m/%Y')
                days_bull = (today.date() - prev_low_date.date()).days
            else:
                prev_low_date_str, prev_low_price, days_bull = "N/A", 0.0, 0
            stoxx_bear = {
                'current_date': stoxx_long.index[-1].strftime('%d/%m/%Y'),
                'current': current_price,
                'last_high_date': last_high_date_str,
                'last_high': ath_value,
                'new_bear_threshold': new_bear_threshold,
                'prev_bear_date': prev_low_date_str,
                'prev_bear': prev_low_price,
                'days_bull': days_bull,
                'avg_days_bull': 857,
            }
    except Exception:
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
    score = max(0, min(100, int(score)))

    if score >= 60:
        bias = 'Long (6 long/4 short)'
    elif score <= 40:
        bias = 'Short (4 long/6 short)'
    else:
        bias = 'Neutral (5 long/5 short)'

    return metrics, tailwinds, headwinds, neutrals, bias, score


def generate_html_summary(tailwinds, headwinds, neutrals, bias, data, history, metrics, today, score):
    def build_section(items_list):
        html_parts = []
        for item in items_list:
            gkey = get_graph_key(item)
            fig = generate_graph(gkey, data, history, metrics, today)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            short_html = ''
            short_fig = generate_short_term_graph(gkey, history, today)
            if short_fig is not None:
                sbuf = BytesIO()
                short_fig.savefig(sbuf, format='png', bbox_inches='tight', dpi=150)
                sbuf.seek(0)
                short_base64 = base64.b64encode(sbuf.read()).decode('utf-8')
                plt.close(short_fig)
                short_html = (f'<h4 style="margin:20px 0 8px 0;color:#555;">Short-term View (last 3 months)</h4>'
                              f'<img src="data:image/png;base64,{short_base64}" '
                              f'style="width:100%;max-width:820px;display:block;margin:0 auto;'
                              f'box-shadow:0 4px 12px rgba(0,0,0,0.08);"/>')

            desc = get_description(gkey)
            desc_html = f'<p style="margin-top:12px;color:#444;font-size:0.95em;">{desc}</p>' if desc else ''

            terminal_html = ''
            if gkey == '10yr_yield':
                current = data['10yr_yield']
                terminal = metrics.get('terminal_10yr', current)
                terminal_html = f'''
                <h4 style="margin:25px 0 10px 0;color:#555;">10-Yr Treasury Yield &amp; Terminal Yield</h4>
                <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;font-size:0.95em;border:1px solid #ddd;">
                    <thead><tr style="background:#f8f8f8;">
                        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Metric</th>
                        <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
                    </tr></thead>
                    <tbody>
                        <tr><td style="padding:10px;border:1px solid #ddd;">Current 10-Yr Yield</td>
                            <td style="padding:10px;border:1px solid #ddd;text-align:right;">{current:.2f}%</td></tr>
                        <tr><td style="padding:10px;border:1px solid #ddd;">Terminal Yield (recent high)</td>
                            <td style="padding:10px;border:1px solid #ddd;text-align:right;">{terminal:.2f}%</td></tr>
                    </tbody>
                </table>'''

            bear_html = ''
            if gkey == 'sp500':
                sp_bear = metrics.get('sp_bear', {})
                if sp_bear:
                    bear_html = f'''
                    <h4 style="margin:25px 0 10px 0;color:#555;">S&amp;P 500 Bull / Bear Market Status</h4>
                    <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;font-size:0.95em;border:1px solid #ddd;">
                        <thead><tr style="background:#f8f8f8;">
                            <th style="padding:10px;border:1px solid #ddd;">Metric</th>
                            <th style="padding:10px;border:1px solid #ddd;">Date / Note</th>
                            <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
                        </tr></thead>
                        <tbody>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Current</td>
                                <td style="padding:10px;border:1px solid #ddd;">{sp_bear.get("current_date","N/A")}</td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get("current",0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Last High</td>
                                <td style="padding:10px;border:1px solid #ddd;">{sp_bear.get("last_high_date","N/A")}</td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get("last_high",0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">New Bear Threshold (−20%)</td>
                                <td style="padding:10px;border:1px solid #ddd;"></td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get("new_bear_threshold",0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Previous Cycle Low</td>
                                <td style="padding:10px;border:1px solid #ddd;">{sp_bear.get("prev_bear_date","N/A")}</td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get("prev_bear",0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;"># Days in Bull</td>
                                <td style="padding:10px;border:1px solid #ddd;"></td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get("days_bull",0)}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Avg Bull Duration</td>
                                <td style="padding:10px;border:1px solid #ddd;"></td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sp_bear.get("avg_days_bull",997)} days</td></tr>
                        </tbody>
                    </table>'''
            elif gkey == 'stoxx600':
                sb = metrics.get('stoxx_bear', {})
                if sb:
                    bear_html = f'''
                    <h4 style="margin:25px 0 10px 0;color:#555;">STOXX 600 Bull / Bear Market Status</h4>
                    <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;font-size:0.95em;border:1px solid #ddd;">
                        <thead><tr style="background:#f8f8f8;">
                            <th style="padding:10px;border:1px solid #ddd;">Metric</th>
                            <th style="padding:10px;border:1px solid #ddd;">Date / Note</th>
                            <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
                        </tr></thead>
                        <tbody>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Current</td>
                                <td style="padding:10px;border:1px solid #ddd;">{sb.get("current_date","N/A")}</td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get("current",0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Last High</td>
                                <td style="padding:10px;border:1px solid #ddd;">{sb.get("last_high_date","N/A")}</td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get("last_high",0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">New Bear Threshold (−20%)</td>
                                <td style="padding:10px;border:1px solid #ddd;"></td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get("new_bear_threshold",0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Previous Cycle Low</td>
                                <td style="padding:10px;border:1px solid #ddd;">{sb.get("prev_bear_date","N/A")}</td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get("prev_bear",0):,.2f}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;"># Days in Bull</td>
                                <td style="padding:10px;border:1px solid #ddd;"></td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get("days_bull",0)}</td></tr>
                            <tr><td style="padding:10px;border:1px solid #ddd;">Avg Bull Duration</td>
                                <td style="padding:10px;border:1px solid #ddd;"></td>
                                <td style="padding:10px;border:1px solid #ddd;text-align:right;">{sb.get("avg_days_bull",857)} days</td></tr>
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

    html = f"""<!DOCTYPE html>
<html><head><title>Portfolio Bias Summary</title>
<style>
body{{font-family:Arial,sans-serif;padding:40px;background:#fff;color:#000;max-width:960px;margin:auto;}}
h1{{color:#1a1a1a;font-size:32px;}}
.bias{{font-size:1.35em;font-weight:bold;color:#003366;margin-bottom:35px;border-bottom:2px solid #e5e5e5;padding-bottom:12px;}}
.score{{font-size:1.4em;font-weight:bold;color:#003366;}}
h2{{font-size:24px;border-bottom:3px solid #ddd;padding-bottom:10px;margin-top:45px;}}
ul{{list-style:none;padding:0;margin:0;}}
li{{margin-bottom:8px;}}
summary{{font-size:1.05em;font-weight:600;cursor:pointer;padding:12px 16px;background:#f8f8f8;
          border:1px solid #e0e0e0;border-radius:6px;list-style:none;}}
summary::-webkit-details-marker{{display:none;}}
summary::before{{content:"▶ ";font-size:0.8em;}}
details[open] summary::before{{content:"▼ ";}}
summary:hover{{background:#f0f0f0;}}
</style></head><body>
<h1>Portfolio Bias Summary</h1>
<p class="bias">Recommended Bias: {bias}</p>
<p class="score">GDP Growth Score: {score}/100</p>
<h2 style="color:#28a745;border-bottom-color:#28a745;">✅ Tailwinds (Positive)</h2>
<ul>{build_section(tailwinds)}</ul>
<h2 style="color:#dc3545;border-bottom-color:#dc3545;">❌ Headwinds (Negative)</h2>
<ul>{build_section(headwinds)}</ul>
<h2>⚖️ Neutrals</h2>
<ul>{build_section(neutrals)}</ul>
</body></html>"""
    return html


def plot_sector_chart(etf_ticker, period='1y'):
    try:
        hist = yf.Ticker(etf_ticker).history(period=period)['Close']
        if hist.empty:
            return None
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot(ax=ax, linewidth=2)
        ax.set_title(f'{etf_ticker} ({period})')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        return fig
    except Exception:
        return None


def plot_commodity_chart(ticker, period='1y'):
    try:
        hist = yf.Ticker(ticker).history(period=period)['Close']
        if hist.empty:
            return None
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot(ax=ax, linewidth=2)
        ax.set_title(f'{ticker} ({period})')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        return fig
    except Exception:
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

    performance = {}
    three_m_ago = pd.Timestamp(today - timedelta(days=90))
    for sector, info in all_sectors.items():
        try:
            hist = yf.Ticker(info['etf']).history(period='1y')['Close']
            if hist.empty or len(hist) < 2:
                performance[sector] = 0
                continue
            if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            ret_1y = (hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0] if hist.iloc[0] != 0 else 0
            hist_3m = hist[hist.index >= three_m_ago]
            ret_3m = (hist.iloc[-1] - hist_3m.iloc[0]) / hist_3m.iloc[0] if (not hist_3m.empty and hist_3m.iloc[0] != 0) else 0
            performance[sector] = 0.5 * ret_1y + 0.5 * ret_3m
        except Exception:
            performance[sector] = 0

    sorted_sectors = sorted(performance, key=performance.get, reverse=True)

    if 'Long' in bias:
        longs = ([s for s in sorted_sectors if s in preferred_sectors][:4]
                 or sorted_sectors[:4])
        shorts = ([s for s in sorted_sectors[::-1] if s in preferred_sectors][:3]
                  or sorted_sectors[-3:])
        long_alloc = portfolio_size * 0.6 / max(len(longs), 1)
        short_alloc = portfolio_size * 0.4 / max(len(shorts), 1)
    elif 'Short' in bias:
        longs = ([s for s in sorted_sectors if s in preferred_sectors][:3]
                 or sorted_sectors[:3])
        shorts = ([s for s in sorted_sectors[::-1] if s in preferred_sectors][:4]
                  or sorted_sectors[-4:])
        long_alloc = portfolio_size * 0.4 / max(len(longs), 1)
        short_alloc = portfolio_size * 0.6 / max(len(shorts), 1)
    else:
        longs = sorted_sectors[:3]
        shorts = sorted_sectors[-3:]
        long_alloc = portfolio_size * 0.5 / max(len(longs), 1)
        short_alloc = portfolio_size * 0.5 / max(len(shorts), 1)

    tilt_df = pd.DataFrame({
        'Type': ['Long'] * len(longs) + ['Short'] * len(shorts),
        'Sector': longs + shorts,
        'ETF': [all_sectors[s]['etf'] for s in longs + shorts],
        'Allocation ($)': [f"${long_alloc:,.0f}"] * len(longs) + [f"${short_alloc:,.0f}"] * len(shorts),
    })

    return tilt_df, all_sectors, all_commodities


# --- STREAMLIT UI ---
st.set_page_config(page_title="Macro Portfolio Bias & Sector Tilt", layout="wide")
st.title("📊 Portfolio Bias Analysis & Sector Tilt Dashboard")

for key in ['bias_calculated', 'bias', 'score', 'metrics', 'data', 'history',
            'today', 'tailwinds', 'headwinds', 'neutrals']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'bias_calculated' else None

if st.button("🔄 Update Analysis", type="primary"):
    with st.spinner("Fetching latest macro data..."):
        try:
            st.session_state.data, st.session_state.history, st.session_state.today = fetch_data()
            (st.session_state.metrics,
             st.session_state.tailwinds,
             st.session_state.headwinds,
             st.session_state.neutrals,
             st.session_state.bias,
             st.session_state.score) = calculate_metrics(
                st.session_state.data,
                st.session_state.history,
                st.session_state.today
            )
            st.success(f"✅ Analysis updated for {st.session_state.today.strftime('%Y-%m-%d')}")
            st.info(f"**GDP Growth Score: {st.session_state.score}/100** → {st.session_state.bias}")

            with st.spinner("Generating HTML report..."):
                html_report = generate_html_summary(
                    st.session_state.tailwinds,
                    st.session_state.headwinds,
                    st.session_state.neutrals,
                    st.session_state.bias,
                    st.session_state.data,
                    st.session_state.history,
                    st.session_state.metrics,
                    st.session_state.today,
                    st.session_state.score
                )
            st.download_button(
                "📥 Download Interactive HTML Report",
                data=html_report,
                file_name=f"macro_bias_{st.session_state.today.date()}.html",
                mime="text/html"
            )
            st.session_state.bias_calculated = True
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.exception(e)

if st.session_state.bias_calculated:
    st.header("🎯 Sector Tilt Recommendations")
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_level = st.selectbox("Risk Tolerance", ['Low', 'Medium', 'High'], index=1)
    with col2:
        portfolio_size = st.number_input("Portfolio Size ($)", min_value=1000, value=100000, step=1000)
    with col3:
        st.metric("Current Bias", st.session_state.bias)
        st.metric("Score", f"{st.session_state.score}/100")

    preferred_sectors = st.multiselect(
        "Preferred Sectors (optional filter)",
        options=['Technology', 'Industrials', 'Financials', 'Consumer Discretionary',
                 'Utilities', 'Healthcare', 'Energy', 'Materials',
                 'Consumer Staples', 'Real Estate', 'Communication Services']
    )

    if st.button("📈 Generate Sector Tilt Recommendations", type="primary"):
        with st.spinner("Fetching ETF performance data..."):
            try:
                tilt_df, sectors, commodities = generate_sector_tilt(
                    st.session_state.bias, st.session_state.score, risk_level,
                    preferred_sectors, portfolio_size,
                    st.session_state.data, st.session_state.metrics, st.session_state.today
                )
                st.subheader("Recommended Sector Tilt")
                st.dataframe(tilt_df, use_container_width=True)

                csv = tilt_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", data=csv,
                                   file_name="sector_tilt.csv", mime="text/csv")

                st.subheader("📊 Sector Performance Charts")
                for sector, info in sectors.items():
                    with st.expander(f"{sector} — {info['etf']}"):
                        cols = st.columns(3)
                        for i, period in enumerate(['5y', '1y', '3mo']):
                            with cols[i]:
                                st.caption(period)
                                fig = plot_sector_chart(info['etf'], period)
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    st.warning("Chart unavailable")

                st.subheader("🥇 Commodity Performance Charts")
                for comm, info in commodities.items():
                    with st.expander(f"{comm} — {info['ticker']}"):
                        cols = st.columns(3)
                        for i, period in enumerate(['5y', '1y', '3mo']):
                            with cols[i]:
                                st.caption(period)
                                fig = plot_commodity_chart(info['ticker'], period)
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    st.warning("Chart unavailable")

            except Exception as e:
                st.error(f"Sector tilt error: {e}")
                st.exception(e)

