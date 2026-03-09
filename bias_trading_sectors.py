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
    """Replaces .last('NM') — uses explicit timestamp filtering."""
    if series is None or series.empty:
        return series
    cutoff = series.index[-1] - pd.Timedelta(days=window_days)
    return series[series.index >= cutoff]


def normalize_index(series):
    """Strip tz and freq metadata from FRED series index."""
    if series is None or series.empty:
        return series
    idx = pd.DatetimeIndex(series.index).tz_localize(None)
    return pd.Series(series.values, index=idx)


def _is_monthly(series):
    """
    Returns True if the series has monthly (or lower) frequency.
    Detected by checking median gap between observations.
    """
    if series is None or len(series) < 3:
        return False
    gaps = pd.Series(series.index).diff().dropna()
    median_gap = gaps.median()
    return median_gap >= pd.Timedelta(days=20)


def _apply_axis_format(ax, series):
    """
    FIX: Auto-select x-axis date formatter and locator based on
    whether the series is monthly or daily/sub-monthly.
    Monthly series get MonthLocator + '%b %Y'.
    Daily series get AutoDateFormatter.
    """
    if _is_monthly(series):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')


def _short_term_window(series, days=90):
    """
    FIX: For monthly series, use a 180-day window instead of 90
    to guarantee >= 5 visible data points in the short-term chart.
    For daily series, use the requested days.
    """
    if _is_monthly(series):
        window = 185  # ~6 months → 5-6 monthly points
    else:
        window = days
    return safe_last(series, window)


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
                num_months = 48
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
                    if (len(ths) == 2
                            and ths[0].text.strip() == 'Date'
                            and ths[1].text.strip() == 'Value'):
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
    data['ism_services'], history['ism_services']           = get_econ_series('non manufacturing pmi', 53.8, 24)
    data['nfib'], history['nfib']                           = get_econ_series('nfib business optimism index', 99.3, 24)
    data['cpi_volatile'], history['cpi_volatile']           = get_econ_series('cpi_volatile', 300)
    data['sbi'], history['sbi']                             = get_econ_series('sbi', 68.4, 24)
    data['eesi'], history['eesi']                           = get_econ_series('eesi', 50, 24)
    data['umcsi'], history['umcsi']                         = safe_get_series('UMCSENT', 56.6)

    building_permits_raw, history['building_permits'] = safe_get_series('PERMIT', 1448)
    history['building_permits'] = normalize_index(history['building_permits'])
    data['building_permits'] = building_permits_raw / 1000

    data['fed_funds'], history['fed_funds']   = safe_get_series('FEDFUNDS', 3.64)
    data['10yr_yield'], history['10yr_yield'] = safe_get_series('DGS10', 4.086)
    data['2yr_yield'], history['2yr_yield']   = safe_get_series('DGS2', 3.48)
    data['bbb_yield'], history['bbb_yield']   = safe_get_series('BAMLC0A4CBBBEY', 4.93)
    data['ccc_yield'], history['ccc_yield']   = safe_get_series('BAMLH0A3HYCEY', 12.44)
    data['m1'], history['m1']                 = safe_get_series('M1SL', 19100)
    data['m2'], history['m2']                 = safe_get_series('M2SL', 22400)

    def get_yf_data(ticker, default_val, default_std, period='1y'):
        try:
            hist = yf.Ticker(ticker).history(period=period)['Close']
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
                np.random.normal(default_val, default_std, len(date_range)),
                index=date_range
            )

    data['vix'], history['vix']     = get_yf_data('^VIX', 19.09, 5, '1y')
    data['move'], history['move']   = get_yf_data('^MOVE', 85.0, 10, '1y')
    data['copper'], history['copper'] = get_yf_data('HG=F', 4.0, 0.5, '1y')
    data['gold'], history['gold']   = get_yf_data('GC=F', 2000, 200, '1y')

    try:
        _, history['sp500']      = get_yf_data('^GSPC', 5000, 500, '1y')
        _, history['sp500_long'] = get_yf_data('^GSPC', 5000, 500, '5y')
        data['sp_lagging'] = 'UP' if history['sp500'].iloc[-1] > history['sp500'].iloc[0] else 'DOWN'
    except Exception:
        data['sp_lagging'] = 'UP'
        dr = pd.date_range(end=today, periods=365, freq='B')
        history['sp500'] = pd.Series(np.random.normal(5000, 500, len(dr)), index=dr)
        dr_long = pd.date_range(end=today, periods=1825, freq='B')
        history['sp500_long'] = pd.Series(np.random.normal(5000, 500, len(dr_long)), index=dr_long)

    # FIX: Try multiple tickers for STOXX 600 — EXW1.DE is most reliable
    stoxx_loaded = False
    for stoxx_ticker in ['EXW1.DE', '^STOXX', 'FEZ', 'EXSA.DE']:
        try:
            _, hist_1y   = get_yf_data(stoxx_ticker, 500, 50, '1y')
            _, hist_5y   = get_yf_data(stoxx_ticker, 500, 50, '5y')
            if not hist_1y.empty and len(hist_1y) > 100:
                history['stoxx600']      = hist_1y
                history['stoxx600_long'] = hist_5y
                data['stoxx_lagging'] = 'UP' if hist_1y.iloc[-1] > hist_1y.iloc[0] else 'DOWN'
                stoxx_loaded = True
                break
        except Exception:
            continue

    if not stoxx_loaded:
        # Last resort: use S&P scaled — still better than random noise
        data['stoxx_lagging'] = data.get('sp_lagging', 'UP')
        sp_last = float(history['sp500'].iloc[-1])
        scale = 500.0 / sp_last if sp_last != 0 else 0.1
        history['stoxx600']      = (history['sp500'] * scale).rename('STOXX600_proxy')
        history['stoxx600_long'] = (history['sp500_long'] * scale).rename('STOXX600_proxy_long')

    # Core CPI — fetch ≥48 months for reliable YoY
    try:
        core = fred.get_series('CPILFESL')
        core = normalize_index(core)
        if len(core) < 14:
            raise ValueError("Not enough core CPI data")
        data['core_cpi_yoy'] = ((core.iloc[-1] / core.iloc[-13]) - 1) * 100
        history['core_cpi'] = core
    except Exception:
        data['core_cpi_yoy'] = 2.5
        date_range = pd.date_range(end=today, periods=48, freq='ME')
        base = 300.0
        history['core_cpi'] = pd.Series(
            [base * (1 + 0.025 / 12) ** i for i in range(48)],
            index=date_range
        )

    return data, history, today


# ---------------------------------------------------------------------------
# Centralised computation helpers
# ---------------------------------------------------------------------------

def _compute_real_rate(history, key_10_or_2):
    """
    Compute real rate series aligned to core_cpi monthly index.
    Returns a pd.Series; may be sparse (monthly cadence).
    """
    core = history['core_cpi'].dropna()
    if len(core) < 14:
        return pd.Series(dtype=float)
    core_yoy = ((core / core.shift(12)) - 1) * 100
    core_yoy = core_yoy.dropna()
    yield_key = '10yr_yield' if '10' in key_10_or_2 else '2yr_yield'
    yield_hist = history[yield_key].reindex(core_yoy.index, method='nearest',
                                             tolerance=pd.Timedelta('35D'))
    real = (yield_hist - core_yoy).dropna()
    return real


def _compute_core_cpi_yoy(history):
    core = history['core_cpi'].dropna()
    if len(core) < 14:
        return pd.Series(dtype=float)
    return ((core / core.shift(12)) - 1) * 100


def _compute_cpi_yoy(history):
    cpi = history['cpi_volatile'].dropna()
    if len(cpi) < 14:
        return pd.Series(dtype=float)
    return ((cpi / cpi.shift(12)) - 1) * 100


def _plot_macd_on_ax(ax, sp_series, title):
    """Plot MACD with correct bar width in matplotlib date float units."""
    sp = sp_series.dropna()
    if len(sp) < 26:
        ax.text(0.5, 0.5, f'Not enough data ({len(sp)} pts, need ≥26)',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title(title)
        return
    macd, sig, hist_vals = compute_macd(sp)
    x_dates = mdates.date2num(sp.index.to_pydatetime())
    bar_width = (x_dates[1] - x_dates[0]) * 0.8 if len(x_dates) > 1 else 0.8
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist_vals]
    ax.bar(x_dates, hist_vals.values, width=bar_width, alpha=0.5, color=colors, label='Histogram')
    ax.plot(x_dates, macd.values,  label='MACD',   color='#1976D2', linewidth=1.5)
    ax.plot(x_dates, sig.values,   label='Signal', color='#FF6F00', linewidth=1.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title(title)


def _finalise_ax(ax, series):
    """Apply smart axis formatter and tighten layout."""
    if series is not None and not series.empty:
        _apply_axis_format(ax, series)
    plt.tight_layout()


def _plot_series(ax, series, title, hline=None):
    """
    Render a series on ax with appropriate scatter density and axis format.
    hline: optional float — draws a horizontal reference line.
    """
    ax.set_title(title)
    if series is None or series.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return
    s = series.dropna()
    if s.empty:
        ax.text(0.5, 0.5, 'All values NaN', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return
    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', linewidth=1)
    s.plot(ax=ax, linewidth=2, color='#1976D2')
    # Always show scatter dots — helps when there are few points
    ax.scatter(s.index, s.values, color='#E53935', s=30, zorder=5)
    _apply_axis_format(ax, s)


# ---------------------------------------------------------------------------
# Main graph — 12M (or appropriate long window)
# ---------------------------------------------------------------------------

def generate_graph(metric_key, data, history, metrics, today):
    fig, ax = plt.subplots(figsize=(8, 4))
    series = None
    hline = None

    if metric_key == 'copper_gold':
        common = history['copper'].index.intersection(history['gold'].index)
        if len(common) > 0:
            ratio = history['copper'].reindex(common) / history['gold'].reindex(common)
            series = safe_last(ratio, 365)
        ax.set_title('Copper/Gold Ratio (last 12M)')

    elif metric_key == 'spread_10ff':
        y10 = history['10yr_yield']
        ff  = history['fed_funds'].reindex(y10.index, method='nearest',
                                           tolerance=pd.Timedelta('35D'))
        spread = (y10 - ff).dropna()
        series = safe_last(spread, 365)
        hline = 0
        ax.set_title('10Yr-FedFunds Spread (last 12M)')

    elif metric_key == 'spread_10_2':
        y10 = history['10yr_yield']
        y2  = history['2yr_yield'].reindex(y10.index, method='nearest',
                                           tolerance=pd.Timedelta('5D'))
        spread = (y10 - y2).dropna()
        series = safe_last(spread, 365)
        hline = 0
        ax.set_title('10Yr-2Yr Spread (last 12M)')

    elif metric_key == 'yield_curve_compare':
        cutoff = pd.Timestamp(today - timedelta(days=1095))
        ten    = history['10yr_yield'][history['10yr_yield'].index >= cutoff].dropna()
        two    = history['2yr_yield'].reindex(ten.index, method='nearest',
                                              tolerance=pd.Timedelta('5D')).dropna()
        series = (ten - two).dropna()
        hline  = 0
        ax.set_title('10Yr - 2Yr Spread (last 3 years)')

    elif metric_key in ('real_rate_10yr', 'real_rate_2yr'):
        series = safe_last(_compute_real_rate(history, metric_key), 365)
        hline  = 0
        lbl    = '10Yr' if '10' in metric_key else '2Yr'
        ax.set_title(f'Real Rate {lbl} (last 12M)')

    elif metric_key == 'core_cpi':
        series = safe_last(_compute_core_cpi_yoy(history).dropna(), 365)
        ax.set_title('Core CPI YoY % (last 12M)')

    elif metric_key == 'cpi_volatile':
        series = safe_last(_compute_cpi_yoy(history).dropna(), 365)
        ax.set_title('CPI Volatile YoY % (last 12M)')

    elif metric_key == 'macd':
        sp = safe_last(history['sp500'], 365)
        _plot_macd_on_ax(ax, sp, 'LazyMan MACD (last 12M)')
        plt.tight_layout()
        return fig

    # FIX: sp_96 top chart shows the 9M→6M AGO window (the signal window)
    elif metric_key == 'sp_96':
        sp = history['sp500']
        if len(sp) > 200:
            idx_9m = max(0, len(sp) - 189)
            idx_6m = max(0, len(sp) - 126)
            series = sp.iloc[idx_9m: idx_6m + 1]
        else:
            series = safe_last(sp, 274)
        ax.set_title('S&P 500 – 9M-to-6M-Ago Window (leading signal)')

    # FIX: stoxx_96 top chart shows the 9M→6M AGO window
    elif metric_key == 'stoxx_96':
        stoxx = history['stoxx600']
        if len(stoxx) > 200:
            idx_9m = max(0, len(stoxx) - 189)
            idx_6m = max(0, len(stoxx) - 126)
            series = stoxx.iloc[idx_9m: idx_6m + 1]
        else:
            series = safe_last(stoxx, 274)
        ax.set_title('STOXX 600 – 9M-to-6M-Ago Window (leading signal)')

    elif metric_key in history:
        series = safe_last(history[metric_key], 365)
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} (last 12M)")

    _plot_series(ax, series, ax.get_title(), hline=hline)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Short-term graph — last 3M (or ~6M for monthly series)
# ---------------------------------------------------------------------------

def generate_short_term_graph(metric_key, history, today):
    fig, ax = plt.subplots(figsize=(8, 3))
    series = None
    hline  = None

    if metric_key == 'copper_gold':
        common = history['copper'].index.intersection(history['gold'].index)
        if len(common) > 0:
            ratio  = history['copper'].reindex(common) / history['gold'].reindex(common)
            series = _short_term_window(ratio)
        ax.set_title('Copper/Gold Ratio – Recent View')

    elif metric_key == 'spread_10ff':
        y10    = history['10yr_yield']
        ff     = history['fed_funds'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('35D'))
        spread = (y10 - ff).dropna()
        series = _short_term_window(spread)
        hline  = 0
        ax.set_title('10Yr-FedFunds Spread – Recent View')

    elif metric_key == 'spread_10_2':
        y10    = history['10yr_yield']
        y2     = history['2yr_yield'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('5D'))
        spread = (y10 - y2).dropna()
        series = _short_term_window(spread)
        hline  = 0
        ax.set_title('10Yr-2Yr Spread – Recent View')

    elif metric_key == 'yield_curve_compare':
        short  = pd.Timestamp(today - timedelta(days=90))
        ten    = history['10yr_yield'][history['10yr_yield'].index >= short].dropna()
        two    = history['2yr_yield'].reindex(ten.index, method='nearest',
                                              tolerance=pd.Timedelta('5D')).dropna()
        series = (ten - two).dropna()
        hline  = 0
        ax.set_title('10Yr-2Yr Spread – Last 3 Months')

    elif metric_key in ('real_rate_10yr', 'real_rate_2yr'):
        # FIX: compute full series first, then use _short_term_window
        # so monthly series gets 6M window instead of 90-day (2-3 points)
        full   = _compute_real_rate(history, metric_key).dropna()
        series = _short_term_window(full)
        hline  = 0
        lbl    = '10Yr' if '10' in metric_key else '2Yr'
        ax.set_title(f'Real Rate {lbl} – Recent View')

    elif metric_key == 'core_cpi':
        # FIX: compute full YoY first, then window
        full   = _compute_core_cpi_yoy(history).dropna()
        series = _short_term_window(full)
        ax.set_title('Core CPI YoY % – Recent View')

    elif metric_key == 'cpi_volatile':
        # FIX: compute full YoY first, then window
        full   = _compute_cpi_yoy(history).dropna()
        series = _short_term_window(full)
        ax.set_title('CPI Volatile YoY % – Recent View')

    elif metric_key == 'macd':
        # FIX: compute MACD on full 1Y data, display last 3M slice
        sp_full = history['sp500'].dropna()
        if len(sp_full) >= 26:
            macd_full, sig_full, hist_full = compute_macd(sp_full)
            short   = pd.Timestamp(today - timedelta(days=90))
            m3      = macd_full[macd_full.index >= short]
            s3      = sig_full[sig_full.index >= short]
            h3      = hist_full[hist_full.index >= short]
            if not m3.empty:
                x = mdates.date2num(m3.index.to_pydatetime())
                bw = (x[1] - x[0]) * 0.8 if len(x) > 1 else 0.8
                ax.bar(x, h3.values, width=bw, alpha=0.5,
                       color=['#26a69a' if v >= 0 else '#ef5350' for v in h3],
                       label='Histogram')
                ax.plot(x, m3.values, color='#1976D2', linewidth=1.5, label='MACD')
                ax.plot(x, s3.values, color='#FF6F00', linewidth=1.5, label='Signal')
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.legend(fontsize=8)
                ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        else:
            ax.text(0.5, 0.5, f'Not enough data ({len(sp_full)} pts)',
                    ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title('LazyMan MACD – Last 3 Months')
        plt.tight_layout()
        return fig

    # FIX: sp_96 bottom chart = last 3M of S&P price (current momentum)
    elif metric_key == 'sp_96':
        series = _short_term_window(history['sp500'])
        ax.set_title('S&P 500 – Last 3 Months (current momentum)')

    # FIX: stoxx_96 bottom chart = last 3M of STOXX price (current momentum)
    elif metric_key == 'stoxx_96':
        series = _short_term_window(history['stoxx600'])
        ax.set_title('STOXX 600 – Last 3 Months (current momentum)')

    # FIX: building_permits — monthly series, use _short_term_window
    elif metric_key == 'building_permits':
        series = _short_term_window(history['building_permits'])
        ax.set_title('Building Permits – Recent View (~6M)')

    # FIX: m1/m2 — monthly, use _short_term_window
    elif metric_key in ('m1', 'm2'):
        series = _short_term_window(history[metric_key])
        ax.set_title(f"{metric_key.upper()} Money Supply – Recent View (~6M)")

    elif metric_key in history:
        series = _short_term_window(history[metric_key])
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} – Recent View")

    _plot_series(ax, series, ax.get_title(), hline=hline)
    plt.tight_layout()
    return fig


def get_graph_key(item_text):
    if 'Copper/Gold' in item_text:                                          return 'copper_gold'
    if '10Yr-FedFunds' in item_text:                                        return 'spread_10ff'
    if '10Yr-2Yr' in item_text:                                             return 'spread_10_2'
    if 'Yield Curve comparison' in item_text or 'Yield Curve Comparison' in item_text: return 'yield_curve_compare'
    if 'Real Rate' in item_text and '10' in item_text:                      return 'real_rate_10yr'
    if 'Real Rate' in item_text and '2' in item_text:                       return 'real_rate_2yr'
    if 'Fed Funds' in item_text:                                            return 'fed_funds'
    if '10-Yr Yield' in item_text or '10-Yr' in item_text:                 return '10yr_yield'
    if '2-Yr Yield' in item_text or '2-Yr' in item_text:                   return '2yr_yield'
    if 'Core CPI' in item_text:                                             return 'core_cpi'
    if 'BBB Yield' in item_text:                                            return 'bbb_yield'
    if 'CCC Yield' in item_text:                                            return 'ccc_yield'
    if 'VIX' in item_text:                                                  return 'vix'
    if 'MOVE' in item_text:                                                 return 'move'
    if 'Manufacturing PMI' in item_text:                                    return 'ism_manufacturing'
    if 'Services PMI' in item_text:                                         return 'ism_services'
    if 'UMCSI' in item_text:                                                return 'umcsi'
    if 'Building Permits' in item_text:                                     return 'building_permits'
    if 'NFIB' in item_text:                                                 return 'nfib'
    if 'CPI Volatile' in item_text or 'CPI-Volatile' in item_text:         return 'cpi_volatile'
    if 'SBI' in item_text:                                                  return 'sbi'
    if 'EESI' in item_text:                                                 return 'eesi'
    if 'M1' in item_text:                                                   return 'm1'
    if 'M2' in item_text:                                                   return 'm2'
    if '9-6' in item_text and 'S&P' in item_text:                          return 'sp_96'
    if '9-6' in item_text and 'STOXX' in item_text:                        return 'stoxx_96'
    if 'LazyMan MACD' in item_text or 'MACD' in item_text:                 return 'macd'
    if 'S&P' in item_text:                                                  return 'sp500'
    if 'STOXX' in item_text:                                                return 'stoxx600'
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
        'sp500':   'S&P500 is a forward-looking indicator for USA GDP: When S&P500 experiences growth, investors expect positive/increasing firm earnings that should reflect in solid GDP growth. Predicting USA GDP with the S&P500 as an indicator has correlation of 69.04%',
        'stoxx600':'STOXX 600 (Europe) as global risk appetite proxy. Strong correlation with US GDP via trade/finance channels (~55%). 9-6 month return is a leading signal similar to S&P.',
        'spread_10ff': '10-Year minus Fed Funds spread. Positive = normal steep curve = accommodative conditions → expansionary for GDP.',
        'spread_10_2': '10-Year minus 2-Year spread (classic yield curve). Positive spread strongly predicts GDP expansion.',
        'yield_curve_compare': '3-year view of 10Yr-2Yr spread. Steep positive curve = healthy expansion expectations.',
        'm1': 'M1/M2 money supply growth (liquidity). Rising aggregates support credit creation and GDP expansion.',
        'm2': 'M1/M2 money supply growth (liquidity). Rising aggregates support credit creation and GDP expansion.',
        'vix': 'The VIX, aka fear index. Lower than historical volatility implies positive outlook for GDP.',
        'bbb_yield': 'Corporate bond yields reflect cost of borrowing. Cheaper borrowing implies expansionary conditions.',
        'ccc_yield': 'Higher yields imply more expensive borrowing → contractionary conditions.',
        'sp_96': '69% correlation: S&P 9-to-6-months-ago return is a leading indicator for GDP direction. Top chart = signal window. Bottom chart = current momentum.',
        'stoxx_96': 'STOXX 600 9-to-6-months-ago return as leading indicator. Top chart = signal window. Bottom chart = current momentum.',
        'core_cpi': 'Core CPI YoY — excludes food & energy. Falling trend = disinflationary, reduces pressure on Fed, positive for growth.',
        'cpi_volatile': 'Headline CPI YoY — includes food & energy (volatile components). Useful for tracking total inflation pressure.',
        'building_permits': 'Building permits lead housing starts by ~1-2 months and are a key leading indicator for economic activity.',
    }
    if gkey.startswith('real_rate'):
        return 'Real rate = nominal yield minus core CPI YoY. Negative real rates are highly stimulative → positive for GDP growth.'
    return descriptions.get(gkey, '')


def calculate_metrics(data, history, today):
    metrics = {}
    try:
        metrics['real_rate_10yr']        = data['10yr_yield'] - data['core_cpi_yoy']
        metrics['real_rate_2yr']         = data['2yr_yield']  - data['core_cpi_yoy']
        metrics['yield_curve_10ff']      = data['10yr_yield'] - data['fed_funds']
        metrics['yield_curve_10_2']      = data['10yr_yield'] - data['2yr_yield']
        metrics['copper_gold_ratio']     = data['copper'] / data['gold']
        metrics['copper_gold_ratio_change'] = (
            (history['copper'].iloc[-1] / history['gold'].iloc[-1])
            - (history['copper'].iloc[0]  / history['gold'].iloc[0])
        )
    except Exception as e:
        st.error(f"Metrics calculation error: {e}")
        return {}, [], [], [], "Error", 50

    tailwinds, headwinds, neutrals = [], [], []

    # 1. S&P
    try:
        sp_end        = float(history['sp500'].iloc[-1])
        sp_prev       = float(history['sp500'].iloc[-2]) if len(history['sp500']) > 1 else sp_end
        sp_change_daily = sp_end - sp_prev

        def _ago(ser, days):
            cut = pd.Timestamp(today - timedelta(days=days))
            sub = ser[ser.index >= cut]
            return float(sub.iloc[0]) if not sub.empty else float(ser.iloc[0])

        sp_month_ago        = _ago(history['sp500'], 30)
        sp_three_month_ago  = _ago(history['sp500'], 90)
        sp_start_yoy        = float(history['sp500'].iloc[0])

        sp_daily_pct = (sp_change_daily / sp_prev) * 100 if sp_prev != 0 else 0
        sp_mom_pct   = (sp_end - sp_month_ago) / sp_month_ago * 100 if sp_month_ago != 0 else 0
        sp_3m_pct    = (sp_end - sp_three_month_ago) / sp_three_month_ago * 100 if sp_three_month_ago != 0 else 0
        sp_yoy_pct   = (sp_end - sp_start_yoy) / sp_start_yoy * 100 if sp_start_yoy != 0 else 0

        def _cs(val, pct, up_good=True):
            d = "up" if val >= 0 else "down"
            good = (val >= 0) if up_good else (val < 0)
            c = "green" if good else "red"
            return f'<span style="color:{c}">{d} {abs(pct):.2f}%</span>'

        sp_label = (f"S&P: {sp_end:.2f} (daily {_cs(sp_change_daily, sp_daily_pct)}, "
                    f"MoM {_cs(sp_end-sp_month_ago, sp_mom_pct)}, "
                    f"3M {_cs(sp_end-sp_three_month_ago, sp_3m_pct)}, "
                    f"YoY {sp_yoy_pct:.2f}%)")
        if data['sp_lagging'] == 'UP':
            tailwinds.append(sp_label + " (positive for GDP)")
        else:
            headwinds.append(sp_label + " (negative for GDP)")
    except Exception:
        neutrals.append("S&P Data Unavailable")

    # Copper/Gold
    if metrics['copper_gold_ratio_change'] > 0:
        tailwinds.append("Copper/Gold ratio increasing (positive leading indicator for growth)")
    else:
        headwinds.append("Copper/Gold ratio decreasing (negative leading indicator for growth)")

    def _yd(series_key, label, current_val, up_bad=False):
        """Generic helper for simple up/down yield-style series."""
        try:
            h = history[series_key]
            chg = float(h.iloc[-1]) - float(h.iloc[-2]) if len(h) > 1 else 0
            d = "up" if chg > 0 else "down" if chg < 0 else "unchanged"
            good = (chg < 0) if up_bad else (chg > 0)
            c = "green" if good else "red"
            s = f'<span style="color:{c}">{d} {abs(chg):.3f}</span>'
            return chg, f"{label}: {current_val:.2f}% ({s})"
        except Exception:
            return 0, f"{label}: {current_val:.2f}%"

    # 2. Fed Funds
    ff_chg, ff_lbl = _yd('fed_funds', 'Fed Funds', data['fed_funds'], up_bad=True)
    if ff_chg < 0:
        tailwinds.append(ff_lbl + ", positive)")
    elif ff_chg > 0:
        headwinds.append(ff_lbl + ", negative)")
    else:
        neutrals.append(ff_lbl + ", no change)")

    # 3. 10-Yr Yield (detailed)
    ty_change_daily = (history['10yr_yield'].iloc[-1] - history['10yr_yield'].iloc[-2]
                       if len(history['10yr_yield']) > 1 else 0)
    terminal_rate = float(history['10yr_yield'].max())
    metrics['terminal_10yr'] = terminal_rate

    def _ago_v(key, days):
        h = history[key]
        cut = pd.Timestamp(today - timedelta(days=days))
        sub = h[h.index >= cut]
        return float(sub.iloc[0]) if not sub.empty else float(h.iloc[0])

    ty_mom = data['10yr_yield'] - _ago_v('10yr_yield', 30)
    ty_3m  = data['10yr_yield'] - _ago_v('10yr_yield', 90)

    def _yd_span(val):
        d = "down" if val < 0 else "up"
        c = "green" if val < 0 else "red"
        return f'<span style="color:{c}">{d} {abs(val):.3f}%</span>'

    ty_label = (f"10-Yr Yield: {data['10yr_yield']:.2f}% "
                f"(daily {_yd_span(ty_change_daily)}, MoM {_yd_span(ty_mom)}, "
                f"3M {_yd_span(ty_3m)}, Terminal {terminal_rate:.2f}%)")
    if ty_change_daily < 0:
        tailwinds.append(ty_label + ", positive)")
    else:
        headwinds.append(ty_label + ", negative)")

    # 4. 2-Yr Yield
    ty2_chg, ty2_lbl = _yd('2yr_yield', '2-Yr Yield', data['2yr_yield'], up_bad=True)
    if ty2_chg < 0:
        tailwinds.append(ty2_lbl + ", positive)")
    else:
        headwinds.append(ty2_lbl + ", negative)")

    # 5. Core CPI
    cpi_chg, cpi_lbl = _yd('core_cpi', 'Core CPI (level)', data['core_cpi_yoy'], up_bad=True)
    full_lbl = f"Core CPI YoY: {data['core_cpi_yoy']:.2f}%"
    if cpi_chg < 0:
        tailwinds.append(full_lbl + " (falling, positive)")
    else:
        headwinds.append(full_lbl + " (rising, negative)")

    # 6-7. Real Rates
    for rk, rl in [('real_rate_10yr', '10-Yr'), ('real_rate_2yr', '2-Yr')]:
        rv = metrics[rk]
        lbl = f"Real Rate ({rl}): {rv:.2f}%"
        if rv < 0:
            tailwinds.append(lbl + " (negative real rate, stimulative, positive for GDP)")
        else:
            headwinds.append(lbl + " (positive real rate, restrictive, negative for GDP)")

    # 8-9. BBB / CCC
    for yk, lbl_str, ub in [('bbb_yield', 'BBB Yield', True), ('ccc_yield', 'CCC Yield', True)]:
        chg, lbl = _yd(yk, lbl_str, data[yk], up_bad=ub)
        if chg < 0:
            tailwinds.append(lbl + ", positive)")
        else:
            headwinds.append(lbl + ", negative)")

    # 10. VIX
    vix_val = float(data['vix'])
    vix_chg = (float(history['vix'].iloc[-1]) - float(history['vix'].iloc[-2])
               if len(history['vix']) > 1 else 0)
    vix_mom = vix_val - _ago_v('vix', 30)
    vix_3m  = vix_val - _ago_v('vix', 90)

    def _vs(val): 
        d, c = ("down", "green") if val < 0 else ("up", "red")
        return f'<span style="color:{c}">{d} {abs(val):.2f}</span>'

    vix_label = (f"VIX: {vix_val:.2f} (daily {_vs(vix_chg)}, "
                 f"MoM {_vs(vix_mom)}, 3M {_vs(vix_3m)}")
    if vix_val < 15 or vix_chg < 0:
        tailwinds.append(vix_label + ", positive)")
    else:
        headwinds.append(vix_label + ", negative)")

    # 11. MOVE
    move_val = float(data['move'])
    move_chg = (float(history['move'].iloc[-1]) - float(history['move'].iloc[-2])
                if len(history['move']) > 1 else 0)
    move_mom = move_val - _ago_v('move', 30)
    move_3m  = move_val - _ago_v('move', 90)
    move_label = (f"MOVE: {move_val:.2f} (daily {_vs(move_chg)}, "
                  f"MoM {_vs(move_mom)}, 3M {_vs(move_3m)}")
    if move_chg < 0:
        tailwinds.append(move_label + ", positive for bonds)")
    else:
        headwinds.append(move_label + ", negative)")

    # 12. Manufacturing PMI
    v = data['ism_manufacturing']
    (tailwinds if v > 50 else headwinds).append(
        f"Manufacturing PMI: {v:.1f} ({'expansion' if v > 50 else 'contraction'})")

    # 13. Services PMI
    v = data['ism_services']
    (tailwinds if v > 50 else headwinds).append(
        f"Services PMI: {v:.1f} ({'expansion' if v > 50 else 'contraction'})")

    # 14. UMCSI
    v = data['umcsi']
    if v > 70:   tailwinds.append(f"UMCSI: {v:.1f} (bullish)")
    elif v < 55: headwinds.append(f"UMCSI: {v:.1f} (bearish)")
    else:        neutrals.append(f"UMCSI: {v:.1f} (neutral)")

    # 15. Building Permits
    bp_chg = (history['building_permits'].iloc[-1] - history['building_permits'].iloc[-2]
              if len(history['building_permits']) > 1 else 0)
    bp_dir = "up" if bp_chg > 0 else "down"
    bp_c   = "green" if bp_chg > 0 else "red"
    bp_str = f'<span style="color:{bp_c}">{bp_dir} {abs(bp_chg):.0f}K</span>'
    (tailwinds if bp_chg > 0 else headwinds).append(
        f"Building Permits: {data['building_permits']:.2f}M ({bp_str}, "
        f"{'positive' if bp_chg > 0 else 'negative'})")

    # 16. NFIB
    nfib_chg = (history['nfib'].iloc[-1] - history['nfib'].iloc[-2]
                if len(history['nfib']) > 1 else 0)
    pct_chg  = nfib_chg / history['nfib'].iloc[-2] * 100 if (len(history['nfib']) > 1 and history['nfib'].iloc[-2] != 0) else 0
    prev_m   = history['nfib'].index[-2].strftime('%b') if len(history['nfib']) > 1 else 'Prev'
    curr_m   = history['nfib'].index[-1].strftime('%b')
    direction = "Up" if nfib_chg > 0 else "Down" if nfib_chg < 0 else "Unchanged"
    status   = "(strong)" if data['nfib'] > 100 else "(weak)" if data['nfib'] < 95 else "(neutral)"
    nfib_lbl = f"NFIB: {data['nfib']:.1f} {direction} {abs(nfib_chg):.1f} ({pct_chg:.2f}%) ({prev_m}→{curr_m}) {status}"
    if   data['nfib'] > 100: tailwinds.append(nfib_lbl)
    elif data['nfib'] < 95:  headwinds.append(nfib_lbl)
    else:                     neutrals.append(nfib_lbl)

    # 17. S&P 9-6m Return
    sp = history['sp500']
    if len(sp) > 200:
        p9 = float(sp.iloc[max(0, len(sp) - 189)])
        p6 = float(sp.iloc[max(0, len(sp) - 126)])
        sp_96 = (p6 - p9) / p9 * 100 if p9 != 0 else 0
    else:
        sp_96 = 0
    metrics['sp_96_return'] = sp_96
    (tailwinds if sp_96 > 0 else headwinds).append(f"S&P 9-6m Return: {sp_96:.2f}%")

    # CPI Volatile
    v = data.get('cpi_volatile', 300)
    (tailwinds if v < 300 else headwinds).append(
        f"CPI Volatile: {v:.0f} ({'low, positive' if v < 300 else 'high, negative'})")

    # SBI
    v = data.get('sbi', 0)
    (tailwinds if v > 68 else headwinds).append(
        f"SBI: {v:.1f} ({'strong, positive' if v > 68 else 'weak, negative'})")

    # EESI
    v = data.get('eesi', 0)
    (tailwinds if v > 45 else headwinds).append(
        f"EESI: {v:.1f} ({'positive' if v > 45 else 'negative'})")

    # M1 & M2
    for mk in ['m1', 'm2']:
        h = history[mk]
        growing = bool(h.iloc[-1] > h.iloc[-2]) if len(h) > 1 else False
        metrics[f'{mk}_growth_pos'] = growing
        lbl = mk.upper()
        (tailwinds if growing else headwinds).append(
            f"{lbl} Money Supply: {'Growing' if growing else 'Contracting'} MoM "
            f"({'positive liquidity for GDP' if growing else 'headwind'})")

    # Spreads
    ff_sp = metrics.get('yield_curve_10ff', 0)
    (tailwinds if ff_sp > 0 else headwinds).append(
        f"10Yr-FedFunds Spread: {ff_sp:.2f}% "
        f"({'positive, expansionary' if ff_sp > 0 else 'negative, contractionary'})")

    t2_sp = metrics.get('yield_curve_10_2', 0)
    (tailwinds if t2_sp > 0 else headwinds).append(
        f"10Yr-2Yr Spread: {t2_sp:.2f}% "
        f"({'positive, expansionary' if t2_sp > 0 else 'flat/inverted, contractionary'})")

    yc_pos = t2_sp > 0
    (tailwinds if yc_pos else headwinds).append(
        f"Yield Curve Comparison (3yr): {'Steep/Positive' if yc_pos else 'Flat/Inverted'}")

    # MACD LazyMan
    macd_long_bullish = macd_short_bullish = False
    try:
        sp_close = history['sp500'].dropna()
        if len(sp_close) >= 40:
            ml, sl, _ = compute_macd(sp_close)
            macd_long_bullish = bool(ml.iloc[-1] > sl.iloc[-1])
            metrics['macd_line']   = float(ml.iloc[-1])
            metrics['signal_line'] = float(sl.iloc[-1])
        sp_short = safe_last(sp_close, 45)
        if len(sp_short) >= 26:
            ms, ss, _ = compute_macd(sp_short)
            macd_short_bullish = bool(ms.iloc[-1] > ss.iloc[-1])
    except Exception:
        pass
    metrics['macd_long_bullish']  = macd_long_bullish
    metrics['macd_short_bullish'] = macd_short_bullish
    macd_lbl = (f"LazyMan MACD: Short-term {'Buy' if macd_short_bullish else 'Sell'} "
                f"| Long-term {'Buy' if macd_long_bullish else 'Sell'}")
    if macd_long_bullish:
        tailwinds.append(macd_lbl + " → Bull – SIMPLY Buy (if you're lazy)")
    else:
        headwinds.append(macd_lbl)

    # STOXX 600 9-6m
    st600 = history['stoxx600']
    if len(st600) > 200:
        p9 = float(st600.iloc[max(0, len(st600) - 189)])
        p6 = float(st600.iloc[max(0, len(st600) - 126)])
        stoxx_96 = (p6 - p9) / p9 * 100 if p9 != 0 else 0
    else:
        stoxx_96 = 0
    metrics['stoxx_96_return'] = stoxx_96
    (tailwinds if stoxx_96 > 0 else headwinds).append(
        f"STOXX 600 9-6m Return: {stoxx_96:.2f}%")

    # S&P Bear/Bull
    sp_bear = {}
    try:
        sp_long = history['sp500_long'].dropna()
        if len(sp_long) >= 10:
            cur   = float(sp_long.iloc[-1])
            ath   = float(sp_long.max())
            nhigh = sp_long[sp_long == ath].index[-1]
            lback = today - timedelta(days=1825)
            recent = sp_long[sp_long.index >= lback]
            if not recent.empty:
                prev_low = float(recent.min())
                pld      = recent.idxmin()
                days_bull = (today.date() - pld.date()).days
            else:
                prev_low, pld, days_bull = 0.0, sp_long.index[0], 0
            sp_bear = dict(
                current_date=sp_long.index[-1].strftime('%d/%m/%Y'),
                current=cur, last_high_date=nhigh.strftime('%d/%m/%Y'),
                last_high=ath, new_bear_threshold=ath * 0.8,
                prev_bear_date=pld.strftime('%d/%m/%Y'),
                prev_bear=prev_low, days_bull=days_bull, avg_days_bull=997)
    except Exception:
        pass
    metrics['sp_bear'] = sp_bear

    # STOXX Bear/Bull
    stoxx_bear = {}
    try:
        sl = history['stoxx600_long'].dropna()
        if len(sl) >= 10:
            cur  = float(sl.iloc[-1])
            ath  = float(sl.max())
            nhigh = sl[sl == ath].index[-1]
            lback = today - timedelta(days=1825)
            rec   = sl[sl.index >= lback]
            if not rec.empty:
                prev_low = float(rec.min())
                pld      = rec.idxmin()
                days_bull = (today.date() - pld.date()).days
            else:
                prev_low, pld, days_bull = 0.0, sl.index[0], 0
            stoxx_bear = dict(
                current_date=sl.index[-1].strftime('%d/%m/%Y'),
                current=cur, last_high_date=nhigh.strftime('%d/%m/%Y'),
                last_high=ath, new_bear_threshold=ath * 0.8,
                prev_bear_date=pld.strftime('%d/%m/%Y'),
                prev_bear=prev_low, days_bull=days_bull, avg_days_bull=857)
    except Exception:
        pass
    metrics['stoxx_bear'] = stoxx_bear

    # Scoring
    score  = 0
    score += min(max(metrics.get('sp_96_return', 0) / 5 * 18, 0), 18) if metrics.get('sp_96_return', 0) > 0 else 0
    score += 12 if data.get('sp_lagging') == 'UP' else 0
    score += 15 if metrics.get('yield_curve_10_2', 0) > 0 else 0
    score += 12 if metrics.get('yield_curve_10ff', 0) > 0 else 0
    score += 10 if metrics.get('macd_long_bullish', False) else 0
    score += 8  if metrics.get('stoxx_96_return', 0) > 0 else 0
    score += 10 if metrics.get('real_rate_10yr', 0) < 0 else 0
    score += 8  if metrics.get('real_rate_2yr', 0) < 0 else 0
    score += max(8 - data.get('vix', 20) / 5, 0) if data.get('vix', 20) < 25 else 0
    score += 8  if data['ism_manufacturing'] > 50 else 0
    score += 7  if data['ism_services'] > 50 else 0
    score += 6  if data['umcsi'] > 60 else 0
    score += 5  if data.get('building_permits', 0) > 1.4 else 0
    score += 5  if data.get('sbi', 0) > 68 else 0
    score += 4  if data.get('cpi_volatile', 300) < 300 else 0
    score += 4  if data.get('eesi', 50) > 45 else 0
    score += 5  if data.get('nfib', 99) > 100 else 0
    score += 5  if metrics.get('m1_growth_pos', False) else 0
    score += 5  if metrics.get('m2_growth_pos', False) else 0
    score += 8  if metrics.get('copper_gold_ratio_change', 0) > 0 else 0
    score  = max(0, min(100, int(score)))

    bias = ('Long (6 long/4 short)' if score >= 60
            else 'Short (4 long/6 short)' if score <= 40
            else 'Neutral (5 long/5 short)')

    return metrics, tailwinds, headwinds, neutrals, bias, score


def generate_html_summary(tailwinds, headwinds, neutrals, bias, data, history, metrics, today, score):
    def build_section(items_list):
        html_parts = []
        for item in items_list:
            gkey = get_graph_key(item)
            fig  = generate_graph(gkey, data, history, metrics, today)
            buf  = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)

            short_html = ''
            sfig = generate_short_term_graph(gkey, history, today)
            if sfig is not None:
                sbuf = BytesIO()
                sfig.savefig(sbuf, format='png', bbox_inches='tight', dpi=150)
                sbuf.seek(0)
                s64 = base64.b64encode(sbuf.read()).decode()
                plt.close(sfig)
                short_html = (
                    '<h4 style="margin:20px 0 8px 0;color:#555;">'
                    'Short-term View (recent months)</h4>'
                    f'<img src="data:image/png;base64,{s64}" '
                    'style="width:100%;max-width:820px;display:block;margin:0 auto;'
                    'box-shadow:0 4px 12px rgba(0,0,0,0.08);"/>'
                )

            desc = get_description(gkey)
            desc_html = f'<p style="margin-top:12px;color:#444;font-size:0.95em;">{desc}</p>' if desc else ''

            terminal_html = ''
            if gkey == '10yr_yield':
                cur, term = data['10yr_yield'], metrics.get('terminal_10yr', data['10yr_yield'])
                terminal_html = f'''
                <h4 style="margin:25px 0 10px;color:#555;">10-Yr Treasury: Current vs Terminal</h4>
                <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;
                              font-size:0.95em;border:1px solid #ddd;">
                  <thead><tr style="background:#f8f8f8;">
                    <th style="padding:10px;border:1px solid #ddd;text-align:left;">Metric</th>
                    <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
                  </tr></thead>
                  <tbody>
                    <tr><td style="padding:10px;border:1px solid #ddd;">Current 10-Yr Yield</td>
                        <td style="padding:10px;border:1px solid #ddd;text-align:right;">{cur:.2f}%</td></tr>
                    <tr><td style="padding:10px;border:1px solid #ddd;">Terminal Yield (recent high)</td>
                        <td style="padding:10px;border:1px solid #ddd;text-align:right;">{term:.2f}%</td></tr>
                  </tbody>
                </table>'''

            bear_html = ''
            for bkey, btitle in [('sp500', 'S&amp;P 500'), ('stoxx600', 'STOXX 600')]:
                if gkey == bkey:
                    b = metrics.get('sp_bear' if bkey == 'sp500' else 'stoxx_bear', {})
                    if b:
                        bear_html = f'''
                        <h4 style="margin:25px 0 10px;color:#555;">{btitle} Bull/Bear Market Status</h4>
                        <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;
                                      font-size:0.95em;border:1px solid #ddd;">
                          <thead><tr style="background:#f8f8f8;">
                            <th style="padding:10px;border:1px solid #ddd;">Metric</th>
                            <th style="padding:10px;border:1px solid #ddd;">Date</th>
                            <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
                          </tr></thead>
                          <tbody>
                            <tr><td style="padding:8px;border:1px solid #ddd;">Current</td>
                                <td style="padding:8px;border:1px solid #ddd;">{b.get("current_date","")}</td>
                                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{b.get("current",0):,.2f}</td></tr>
                            <tr><td style="padding:8px;border:1px solid #ddd;">All-Time High</td>
                                <td style="padding:8px;border:1px solid #ddd;">{b.get("last_high_date","")}</td>
                                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{b.get("last_high",0):,.2f}</td></tr>
                            <tr><td style="padding:8px;border:1px solid #ddd;">New Bear Threshold (−20%)</td>
                                <td style="padding:8px;border:1px solid #ddd;"></td>
                                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{b.get("new_bear_threshold",0):,.2f}</td></tr>
                            <tr><td style="padding:8px;border:1px solid #ddd;">Prev Cycle Low</td>
                                <td style="padding:8px;border:1px solid #ddd;">{b.get("prev_bear_date","")}</td>
                                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{b.get("prev_bear",0):,.2f}</td></tr>
                            <tr><td style="padding:8px;border:1px solid #ddd;"># Days in Bull</td>
                                <td style="padding:8px;border:1px solid #ddd;"></td>
                                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{b.get("days_bull",0)}</td></tr>
                            <tr><td style="padding:8px;border:1px solid #ddd;">Avg Bull Duration</td>
                                <td style="padding:8px;border:1px solid #ddd;"></td>
                                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{b.get("avg_days_bull",0)} days</td></tr>
                          </tbody>
                        </table>'''

            html_parts.append(f'''
<li>
  <details>
    <summary>{item}</summary>
    <div style="padding:18px;background:#fafafa;border:1px solid #e5e5e5;
                border-top:none;border-radius:0 0 6px 6px;">
      <img src="data:image/png;base64,{img64}"
           style="width:100%;max-width:820px;display:block;margin:0 auto;
                  box-shadow:0 4px 12px rgba(0,0,0,0.08);"/>
      {short_html}{desc_html}{terminal_html}{bear_html}
    </div>
  </details>
</li>''')
        return ''.join(html_parts)

    return f"""<!DOCTYPE html>
<html><head><title>Portfolio Bias Summary</title>
<style>
body{{font-family:Arial,sans-serif;padding:40px;background:#fff;color:#000;
     max-width:960px;margin:auto;}}
h1{{color:#1a1a1a;font-size:32px;}}
.bias{{font-size:1.35em;font-weight:bold;color:#003366;margin-bottom:35px;
       border-bottom:2px solid #e5e5e5;padding-bottom:12px;}}
.score{{font-size:1.4em;font-weight:bold;color:#003366;}}
h2{{font-size:24px;border-bottom:3px solid #ddd;padding-bottom:10px;margin-top:45px;}}
ul{{list-style:none;padding:0;margin:0;}} li{{margin-bottom:8px;}}
summary{{font-size:1.05em;font-weight:600;cursor:pointer;padding:12px 16px;
          background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;
          list-style:none;}}
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
        'Technology':             {'etf': 'XLK',  'type': 'cyclical'},
        'Industrials':            {'etf': 'XLI',  'type': 'cyclical'},
        'Financials':             {'etf': 'XLF',  'type': 'cyclical'},
        'Consumer Discretionary': {'etf': 'XLY',  'type': 'cyclical'},
        'Materials':              {'etf': 'XLB',  'type': 'cyclical'},
        'Energy':                 {'etf': 'XLE',  'type': 'cyclical'},
        'Healthcare':             {'etf': 'XLV',  'type': 'defensive'},
        'Utilities':              {'etf': 'XLU',  'type': 'defensive'},
        'Consumer Staples':       {'etf': 'XLP',  'type': 'defensive'},
        'Real Estate':            {'etf': 'XLRE', 'type': 'defensive'},
        'Communication Services': {'etf': 'XLC',  'type': 'mixed'},
    }
    all_commodities = {
        'Gold':   {'ticker': 'GLD'},
        'Oil':    {'ticker': 'USO'},
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
            ret_1y  = (hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0] if hist.iloc[0] != 0 else 0
            h3m = hist[hist.index >= three_m_ago]
            ret_3m  = (hist.iloc[-1] - h3m.iloc[0]) / h3m.iloc[0] if (not h3m.empty and h3m.iloc[0] != 0) else 0
            performance[sector] = 0.5 * ret_1y + 0.5 * ret_3m
        except Exception:
            performance[sector] = 0

    ss = sorted(performance, key=performance.get, reverse=True)

    if 'Long' in bias:
        longs  = ([s for s in ss if s in preferred_sectors][:4] or ss[:4])
        shorts = ([s for s in ss[::-1] if s in preferred_sectors][:3] or ss[-3:])
        la, sa = portfolio_size * 0.6 / max(len(longs), 1), portfolio_size * 0.4 / max(len(shorts), 1)
    elif 'Short' in bias:
        longs  = ([s for s in ss if s in preferred_sectors][:3] or ss[:3])
        shorts = ([s for s in ss[::-1] if s in preferred_sectors][:4] or ss[-4:])
        la, sa = portfolio_size * 0.4 / max(len(longs), 1), portfolio_size * 0.6 / max(len(shorts), 1)
    else:
        longs, shorts = ss[:3], ss[-3:]
        la = sa = portfolio_size * 0.5 / 3

    tilt_df = pd.DataFrame({
        'Type':          ['Long'] * len(longs)  + ['Short'] * len(shorts),
        'Sector':        longs + shorts,
        'ETF':           [all_sectors[s]['etf'] for s in longs + shorts],
        'Allocation ($)':[f"${la:,.0f}"] * len(longs) + [f"${sa:,.0f}"] * len(shorts),
    })
    return tilt_df, all_sectors, all_commodities


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Macro Portfolio Bias & Sector Tilt", layout="wide")
st.title("📊 Portfolio Bias Analysis & Sector Tilt Dashboard")

for key in ['bias_calculated', 'bias', 'score', 'metrics', 'data',
            'history', 'today', 'tailwinds', 'headwinds', 'neutrals']:
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
                st.session_state.today)

            st.success(f"✅ Analysis updated for {st.session_state.today.strftime('%Y-%m-%d')}")
            st.info(f"**GDP Growth Score: {st.session_state.score}/100** → {st.session_state.bias}")

            with st.spinner("Generating HTML report (this may take ~30s)..."):
                html_report = generate_html_summary(
                    st.session_state.tailwinds, st.session_state.headwinds,
                    st.session_state.neutrals, st.session_state.bias,
                    st.session_state.data, st.session_state.history,
                    st.session_state.metrics, st.session_state.today,
                    st.session_state.score)

            st.download_button(
                "📥 Download Interactive HTML Report", data=html_report,
                file_name=f"macro_bias_{st.session_state.today.date()}.html",
                mime="text/html")
            st.session_state.bias_calculated = True
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.exception(e)

if st.session_state.bias_calculated:
    st.header("🎯 Sector Tilt Recommendations")
    c1, c2, c3 = st.columns(3)
    with c1: risk_level     = st.selectbox("Risk Tolerance", ['Low', 'Medium', 'High'], index=1)
    with c2: portfolio_size = st.number_input("Portfolio Size ($)", min_value=1000, value=100000, step=1000)
    with c3:
        st.metric("Bias",  st.session_state.bias)
        st.metric("Score", f"{st.session_state.score}/100")

    preferred_sectors = st.multiselect(
        "Preferred Sectors (optional)",
        options=['Technology', 'Industrials', 'Financials', 'Consumer Discretionary',
                 'Utilities', 'Healthcare', 'Energy', 'Materials',
                 'Consumer Staples', 'Real Estate', 'Communication Services'])

    if st.button("📈 Generate Sector Tilt Recommendations", type="primary"):
        with st.spinner("Fetching ETF performance data..."):
            try:
                tilt_df, sectors, commodities = generate_sector_tilt(
                    st.session_state.bias, st.session_state.score, risk_level,
                    preferred_sectors, portfolio_size,
                    st.session_state.data, st.session_state.metrics,
                    st.session_state.today)

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
                                    st.pyplot(fig); plt.close(fig)
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
                                    st.pyplot(fig); plt.close(fig)
                                else:
                                    st.warning("Chart unavailable")
            except Exception as e:
                st.error(f"Sector tilt error: {e}")
                st.exception(e)


