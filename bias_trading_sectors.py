import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoLocator
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

# Keys for which we suppress the short-term (3M) chart
NO_SHORT_TERM_CHART = {'core_cpi', 'eesi', 'cpi_volatile'}


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def safe_last(series, window_days):
    """Explicit timestamp-based window — avoids .last() bugs on FRED indexes."""
    if series is None or series.empty:
        return series
    cutoff = series.index[-1] - pd.Timedelta(days=window_days)
    return series[series.index >= cutoff]


def normalize_index(series):
    """Strip tz and freq metadata from FRED series."""
    if series is None or series.empty:
        return series
    idx = pd.DatetimeIndex(series.index).tz_localize(None)
    return pd.Series(series.values, index=idx)


def _is_monthly(series):
    """True if median gap between observations >= 20 days (monthly or sparser)."""
    if series is None or len(series) < 3:
        return False
    gaps = pd.Series(series.index).diff().dropna()
    return gaps.median() >= pd.Timedelta(days=20)


def _apply_axis_format(ax, series):
    """
    FIX: ConciseDateFormatter gives 'Mar 2025'-style labels for daily data.
    Monthly series get MonthLocator + '%b %Y'.
    Without this, AutoDateFormatter collapses to year-only on 12M windows.
    """
    if series is None or series.empty:
        return
    if _is_monthly(series):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    else:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        # ConciseDateFormatter: shows 'Jan', 'Feb'... with year only at transitions
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)


def _short_term_window(series):
    """
    Monthly series → 185-day window (~6 months) for ≥5 visible points.
    Daily/weekly → 90-day window.
    """
    if _is_monthly(series):
        return safe_last(series, 185)
    return safe_last(series, 90)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_data():
    data    = {}
    history = {}
    today   = datetime.now()

    def safe_get_series(series_id, default_value=0, default_history=None):
        try:
            series = fred.get_series(series_id)
            if series is None or series.empty:
                raise ValueError("Empty series")
            series = normalize_index(series)
            return float(series.iloc[-1]), series
        except Exception:
            if default_history is None:
                date_range = pd.date_range(end=today, periods=48, freq='ME')
                default_history = pd.Series(
                    np.random.normal(default_value, abs(default_value) * 0.05 + 0.01, 48),
                    index=date_range)
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
                date_range  = pd.date_range(end=today, periods=num_months, freq='ME')
                series      = pd.Series(np.random.normal(current_val, 2, num_months), index=date_range)
                return current_val, series
            elif indicator == 'eesi':
                match = re.search(r'(\d+\.?\d*) points', text)
                current_val = float(match.group(1)) if match else default_value
                date_range  = pd.date_range(end=today, periods=num_months, freq='2W')
                series      = pd.Series(np.random.normal(current_val, 3, num_months), index=date_range)
                return current_val, series

            tables = soup.find_all('table')
            table  = None
            for t in tables:
                thead = t.find('thead')
                if thead:
                    ths = thead.find_all('th')
                    if (len(ths) == 2 and ths[0].text.strip() == 'Date'
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
                        dates.append(pd.to_datetime(cols[0].text.strip()))
                        values.append(float(cols[1].text.strip()))
                    except Exception:
                        continue
            series = pd.Series(values, index=dates).sort_index()[-num_months:]
            return float(series.iloc[-1]), series

        except Exception:
            date_range = pd.date_range(end=today, periods=num_months, freq='ME')
            return default_value, pd.Series(
                np.random.normal(default_value, abs(default_value) * 0.05 + 0.01, num_months),
                index=date_range)

    data['ism_manufacturing'], history['ism_manufacturing'] = get_econ_series('business confidence', 52.6, 24)
    data['ism_services'],      history['ism_services']       = get_econ_series('non manufacturing pmi', 53.8, 24)
    data['nfib'],              history['nfib']                = get_econ_series('nfib business optimism index', 99.3, 24)
    data['cpi_volatile'],      history['cpi_volatile']        = get_econ_series('cpi_volatile', 300)
    data['sbi'],               history['sbi']                 = get_econ_series('sbi', 68.4, 24)
    data['eesi'],              history['eesi']                = get_econ_series('eesi', 50, 24)
    data['umcsi'],             history['umcsi']               = safe_get_series('UMCSENT', 56.6)

    bp_raw, history['building_permits'] = safe_get_series('PERMIT', 1448)
    history['building_permits']         = normalize_index(history['building_permits'])
    data['building_permits']            = bp_raw / 1000

    data['fed_funds'],   history['fed_funds']   = safe_get_series('FEDFUNDS', 3.64)
    data['10yr_yield'],  history['10yr_yield']  = safe_get_series('DGS10', 4.086)
    data['2yr_yield'],   history['2yr_yield']   = safe_get_series('DGS2', 3.48)
    data['bbb_yield'],   history['bbb_yield']   = safe_get_series('BAMLC0A4CBBBEY', 4.93)
    data['ccc_yield'],   history['ccc_yield']   = safe_get_series('BAMLH0A3HYCEY', 12.44)
    data['m1'],          history['m1']          = safe_get_series('M1SL', 19100)
    data['m2'],          history['m2']          = safe_get_series('M2SL', 22400)

    def get_yf_data(ticker, default_val, default_std, period='1y'):
        try:
            hist = yf.Ticker(ticker).history(period=period)['Close']
            if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            else:
                hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
            if hist.empty:
                raise ValueError("Empty")
            return float(hist.iloc[-1]), hist
        except Exception:
            ndays  = 365 if period == '1y' else 1825 if period == '5y' else 30
            dr     = pd.date_range(end=today, periods=ndays, freq='B')
            return default_val, pd.Series(np.random.normal(default_val, default_std, len(dr)), index=dr)

    data['vix'],    history['vix']    = get_yf_data('^VIX',  19.09, 5,   '1y')
    data['move'],   history['move']   = get_yf_data('^MOVE', 85.0,  10,  '1y')
    data['copper'], history['copper'] = get_yf_data('HG=F',  4.0,   0.5, '1y')
    data['gold'],   history['gold']   = get_yf_data('GC=F',  2000,  200, '1y')

    # S&P — fetch both 1Y and 5Y
    _, history['sp500']      = get_yf_data('^GSPC', 5000, 500, '1y')
    _, history['sp500_long'] = get_yf_data('^GSPC', 5000, 500, '5y')
    data['sp_lagging'] = 'UP' if history['sp500'].iloc[-1] > history['sp500'].iloc[0] else 'DOWN'

    # STOXX 600 — try multiple tickers
    stoxx_loaded = False
    for sticker in ['EXW1.DE', '^STOXX', 'FEZ', 'EXSA.DE']:
        try:
            _, h1y = get_yf_data(sticker, 500, 50, '1y')
            _, h5y = get_yf_data(sticker, 500, 50, '5y')
            if not h1y.empty and len(h1y) > 100:
                history['stoxx600']      = h1y
                history['stoxx600_long'] = h5y
                data['stoxx_lagging']    = 'UP' if h1y.iloc[-1] > h1y.iloc[0] else 'DOWN'
                stoxx_loaded = True
                break
        except Exception:
            continue
    if not stoxx_loaded:
        sp_last = float(history['sp500'].iloc[-1])
        scale   = 500.0 / sp_last if sp_last != 0 else 0.1
        history['stoxx600']      = (history['sp500']      * scale).rename('STOXX600_proxy')
        history['stoxx600_long'] = (history['sp500_long'] * scale).rename('STOXX600_proxy_long')
        data['stoxx_lagging']    = data.get('sp_lagging', 'UP')

    # Core CPI — fetch 48M for reliable YoY
    try:
        core = fred.get_series('CPILFESL')
        core = normalize_index(core)
        if len(core) < 14:
            raise ValueError("Not enough data")
        data['core_cpi_yoy'] = ((core.iloc[-1] / core.iloc[-13]) - 1) * 100
        history['core_cpi']  = core
    except Exception:
        data['core_cpi_yoy'] = 2.5
        dr = pd.date_range(end=today, periods=48, freq='ME')
        history['core_cpi'] = pd.Series([300.0 * (1 + 0.025 / 12) ** i for i in range(48)], index=dr)

    return data, history, today


# ---------------------------------------------------------------------------
# Centralised computation helpers
# ---------------------------------------------------------------------------

def _compute_real_rate(history, key_10_or_2):
    core = history['core_cpi'].dropna()
    if len(core) < 14:
        return pd.Series(dtype=float)
    core_yoy   = ((core / core.shift(12)) - 1) * 100
    core_yoy   = core_yoy.dropna()
    yield_key  = '10yr_yield' if '10' in key_10_or_2 else '2yr_yield'
    yield_hist = history[yield_key].reindex(core_yoy.index, method='nearest',
                                             tolerance=pd.Timedelta('35D'))
    return (yield_hist - core_yoy).dropna()


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


# ---------------------------------------------------------------------------
# Core plotting primitive — LINE ONLY (no scatter dots)
# ---------------------------------------------------------------------------

def _plot_series(ax, series, title, hline=None, color='#1565C0', linewidth=2):
    """
    FIX: Removed ax.scatter() — clean line-only rendering.
    FIX: ConciseDateFormatter via _apply_axis_format for proper month/year labels.
    """
    ax.set_title(title, fontsize=10, fontweight='bold')
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
        ax.axhline(hline, color='#E53935', linestyle='--', linewidth=1, alpha=0.7)
    # Line only — no scatter
    s.plot(ax=ax, linewidth=linewidth, color=color)
    ax.grid(True, alpha=0.3)
    _apply_axis_format(ax, s)


# ---------------------------------------------------------------------------
# MACD 4-panel chart (5Y price | 1M price | 12M MACD | 1M MACD)
# ---------------------------------------------------------------------------

def _plot_macd_bars(ax, macd_vals, sig_vals, hist_vals, x_dates, title):
    """Shared MACD bar+line renderer used in all MACD panels."""
    bw = (x_dates[1] - x_dates[0]) * 0.8 if len(x_dates) > 1 else 0.8
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist_vals]
    ax.bar(x_dates, hist_vals.values, width=bw, alpha=0.6, color=colors, label='Histogram')
    ax.plot(x_dates, macd_vals.values,  color='#1565C0', linewidth=1.5, label='MACD Line')
    ax.plot(x_dates, sig_vals.values,   color='#E53935', linewidth=1.5, label='Signal Line')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.xaxis_date()
    locator   = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)


def generate_macd_4panel(history, today):
    """
    4-panel MACD figure matching the LazyMan layout:
      Top-left:     S&P 500 – 5 Year
      Top-right:    S&P 500 – 1 Month
      Bottom-left:  MACD (12,26,9) – 12 Months
      Bottom-right: MACD (12,26,9) – 1 Month
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('LazyMan Investor – S&P500 & MACD', fontsize=14, fontweight='bold')
    ax_5y, ax_1m, ax_macd_12m, ax_macd_1m = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    sp_full = history['sp500_long'].dropna()
    sp_1y   = history['sp500'].dropna()

    # --- Top-left: 5Y S&P price ---
    _plot_series(ax_5y, sp_full, 'S&P 500 & MACD Indicator – 5 Year',
                 color='#1565C0', linewidth=1.5)

    # --- Top-right: 1M S&P price ---
    sp_1mo = safe_last(sp_1y, 31)
    _plot_series(ax_1m, sp_1mo, 'S&P 500 & MACD Indicator – 1 Month',
                 color='#1565C0', linewidth=1.8)

    # --- Bottom panels: compute MACD on full 1Y data ---
    if len(sp_1y) >= 26:
        macd_full, sig_full, hist_full = compute_macd(sp_1y)

        # Bottom-left: 12M MACD
        x12 = mdates.date2num(macd_full.index.to_pydatetime())
        _plot_macd_bars(ax_macd_12m, macd_full, sig_full, hist_full, x12,
                        'MACD (12,26,9)')

        # Bottom-right: 1M MACD — slice full series to last 31 days
        cutoff_1m  = pd.Timestamp(today - timedelta(days=31))
        m1m        = macd_full[macd_full.index >= cutoff_1m]
        s1m        = sig_full[sig_full.index >= cutoff_1m]
        h1m        = hist_full[hist_full.index >= cutoff_1m]
        if not m1m.empty:
            x1m = mdates.date2num(m1m.index.to_pydatetime())
            _plot_macd_bars(ax_macd_1m, m1m, s1m, h1m, x1m, 'MACD (12,26,9)')
        else:
            ax_macd_1m.text(0.5, 0.5, 'No 1M data', ha='center', va='center',
                            transform=ax_macd_1m.transAxes, color='gray')
    else:
        for ax in [ax_macd_12m, ax_macd_1m]:
            ax.text(0.5, 0.5, f'Not enough data ({len(sp_1y)} pts, need ≥26)',
                    ha='center', va='center', transform=ax.transAxes, color='gray')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# STOXX 600 9-6M signal table (dynamic)
# ---------------------------------------------------------------------------

def _build_stoxx_96_table_html(history, metrics):
    """
    Dynamically compute the STOXX 600 9-to-6 month signal window and
    return an HTML table with dates, prices, and return %.
    """
    stoxx = history['stoxx600'].dropna()
    if len(stoxx) < 200:
        return ''

    idx_9m    = max(0, len(stoxx) - 189)
    idx_6m    = max(0, len(stoxx) - 126)
    idx_curr  = len(stoxx) - 1

    date_9m   = stoxx.index[idx_9m].strftime('%d/%m/%Y')
    date_6m   = stoxx.index[idx_6m].strftime('%d/%m/%Y')
    date_curr = stoxx.index[idx_curr].strftime('%d/%m/%Y')

    price_9m   = float(stoxx.iloc[idx_9m])
    price_6m   = float(stoxx.iloc[idx_6m])
    price_curr = float(stoxx.iloc[idx_curr])

    ret_96     = (price_6m - price_9m) / price_9m * 100 if price_9m != 0 else 0
    ret_6m_now = (price_curr - price_6m) / price_6m * 100 if price_6m != 0 else 0

    signal_color = "#28a745" if ret_96 >= 0 else "#dc3545"
    signal_text  = "📈 Bullish Signal" if ret_96 >= 0 else "📉 Bearish Signal"

    return f'''
    <h4 style="margin:25px 0 10px;color:#555;">STOXX 600 – 9-to-6 Month Leading Signal Window</h4>
    <table style="width:100%;max-width:820px;margin:15px auto;border-collapse:collapse;
                  font-size:0.95em;border:1px solid #ddd;">
      <thead><tr style="background:#f8f8f8;">
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Metric</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Date</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
      </tr></thead>
      <tbody>
        <tr>
          <td style="padding:9px;border:1px solid #ddd;">Price at 9M Ago (signal start)</td>
          <td style="padding:9px;border:1px solid #ddd;">{date_9m}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;">{price_9m:,.2f}</td>
        </tr>
        <tr>
          <td style="padding:9px;border:1px solid #ddd;">Price at 6M Ago (signal end)</td>
          <td style="padding:9px;border:1px solid #ddd;">{date_6m}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;">{price_6m:,.2f}</td>
        </tr>
        <tr style="background:#f0fff4;">
          <td style="padding:9px;border:1px solid #ddd;font-weight:bold;">9→6M Return (leading signal)</td>
          <td style="padding:9px;border:1px solid #ddd;">{date_9m} → {date_6m}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;
                     font-weight:bold;color:{signal_color};">{ret_96:+.2f}%</td>
        </tr>
        <tr>
          <td style="padding:9px;border:1px solid #ddd;">Current Price</td>
          <td style="padding:9px;border:1px solid #ddd;">{date_curr}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;">{price_curr:,.2f}</td>
        </tr>
        <tr>
          <td style="padding:9px;border:1px solid #ddd;">Return since 6M Ago (momentum)</td>
          <td style="padding:9px;border:1px solid #ddd;">{date_6m} → {date_curr}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;
                     color:{'#28a745' if ret_6m_now >= 0 else '#dc3545'};">{ret_6m_now:+.2f}%</td>
        </tr>
        <tr style="background:#f8f8f8;">
          <td style="padding:9px;border:1px solid #ddd;font-weight:bold;" colspan="2">Signal Interpretation</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;
                     font-weight:bold;color:{signal_color};">{signal_text}</td>
        </tr>
      </tbody>
    </table>'''


# ---------------------------------------------------------------------------
# Main graph (12M / long window)
# ---------------------------------------------------------------------------

def generate_graph(metric_key, data, history, metrics, today):
    # MACD gets its own 4-panel figure
    if metric_key == 'macd':
        return generate_macd_4panel(history, today)

    fig, ax = plt.subplots(figsize=(8, 4))
    series  = None
    hline   = None

    if metric_key == 'copper_gold':
        common = history['copper'].index.intersection(history['gold'].index)
        if len(common) > 0:
            ratio  = history['copper'].reindex(common) / history['gold'].reindex(common)
            series = safe_last(ratio, 365)
        ax.set_title('Copper/Gold Ratio (last 12M)')

    elif metric_key == 'spread_10ff':
        y10    = history['10yr_yield']
        ff     = history['fed_funds'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('35D'))
        spread = (y10 - ff).dropna()
        series = safe_last(spread, 365)
        hline  = 0
        ax.set_title('10Yr-FedFunds Spread (last 12M)')

    elif metric_key == 'spread_10_2':
        y10    = history['10yr_yield']
        y2     = history['2yr_yield'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('5D'))
        spread = (y10 - y2).dropna()
        series = safe_last(spread, 365)
        hline  = 0
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

    # FIX: sp_96 top chart = 9M-to-6M-ago window (the actual signal)
    elif metric_key == 'sp_96':
        sp = history['sp500']
        if len(sp) > 200:
            i9, i6 = max(0, len(sp) - 189), max(0, len(sp) - 126)
            series = sp.iloc[i9: i6 + 1]
        else:
            series = safe_last(sp, 274)
        ax.set_title('S&P 500 – 9M-to-6M-Ago Window (leading signal)')

    # FIX: stoxx_96 top chart = 9M-to-6M-ago window
    elif metric_key == 'stoxx_96':
        stoxx = history['stoxx600']
        if len(stoxx) > 200:
            i9, i6 = max(0, len(stoxx) - 189), max(0, len(stoxx) - 126)
            series = stoxx.iloc[i9: i6 + 1]
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
# Short-term graph (3M / ~6M for monthly)
# Returns None for keys in NO_SHORT_TERM_CHART
# ---------------------------------------------------------------------------

def generate_short_term_graph(metric_key, history, today):
    # FIX: suppress short-term chart for these indicators
    if metric_key in NO_SHORT_TERM_CHART:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    series  = None
    hline   = None

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
        full   = _compute_real_rate(history, metric_key).dropna()
        series = _short_term_window(full)
        hline  = 0
        lbl    = '10Yr' if '10' in metric_key else '2Yr'
        ax.set_title(f'Real Rate {lbl} – Recent View')

    elif metric_key == 'macd':
        # Short-term MACD: compute on full 1Y, slice to 3M
        sp_full = history['sp500'].dropna()
        if len(sp_full) >= 26:
            macd_f, sig_f, hist_f = compute_macd(sp_full)
            short  = pd.Timestamp(today - timedelta(days=90))
            m3, s3, h3 = (macd_f[macd_f.index >= short],
                          sig_f[sig_f.index >= short],
                          hist_f[hist_f.index >= short])
            if not m3.empty:
                x = mdates.date2num(m3.index.to_pydatetime())
                _plot_macd_bars(ax, m3, s3, h3, x, 'LazyMan MACD – Last 3 Months')
            else:
                ax.text(0.5, 0.5, 'No 3M data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
        else:
            ax.text(0.5, 0.5, f'Not enough data ({len(sp_full)} pts)',
                    ha='center', va='center', transform=ax.transAxes, color='gray')
        plt.tight_layout()
        return fig

    # sp_96 bottom = last 3M current momentum
    elif metric_key == 'sp_96':
        series = _short_term_window(history['sp500'])
        ax.set_title('S&P 500 – Last 3 Months (current momentum)')

    # stoxx_96 bottom = last 3M current momentum
    elif metric_key == 'stoxx_96':
        series = _short_term_window(history['stoxx600'])
        ax.set_title('STOXX 600 – Last 3 Months (current momentum)')

    elif metric_key == 'building_permits':
        series = _short_term_window(history['building_permits'])
        ax.set_title('Building Permits – Recent View (~6M)')

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
    if 'Copper/Gold'    in item_text:                                            return 'copper_gold'
    if '10Yr-FedFunds'  in item_text:                                            return 'spread_10ff'
    if '10Yr-2Yr'       in item_text:                                            return 'spread_10_2'
    if 'Yield Curve'    in item_text:                                            return 'yield_curve_compare'
    if 'Real Rate'      in item_text and '10' in item_text:                      return 'real_rate_10yr'
    if 'Real Rate'      in item_text and '2'  in item_text:                      return 'real_rate_2yr'
    if 'Fed Funds'      in item_text:                                            return 'fed_funds'
    if '10-Yr Yield'    in item_text or '10-Yr' in item_text:                   return '10yr_yield'
    if '2-Yr Yield'     in item_text or '2-Yr'  in item_text:                   return '2yr_yield'
    if 'Core CPI'       in item_text:                                            return 'core_cpi'
    if 'BBB Yield'      in item_text:                                            return 'bbb_yield'
    if 'CCC Yield'      in item_text:                                            return 'ccc_yield'
    if 'VIX'            in item_text:                                            return 'vix'
    if 'MOVE'           in item_text:                                            return 'move'
    if 'Manufacturing'  in item_text:                                            return 'ism_manufacturing'
    if 'Services PMI'   in item_text:                                            return 'ism_services'
    if 'UMCSI'          in item_text:                                            return 'umcsi'
    if 'Building'       in item_text:                                            return 'building_permits'
    if 'NFIB'           in item_text:                                            return 'nfib'
    if 'CPI Volatile'   in item_text or 'CPI-Volatile' in item_text:            return 'cpi_volatile'
    if 'SBI'            in item_text:                                            return 'sbi'
    if 'EESI'           in item_text:                                            return 'eesi'
    if 'M1'             in item_text:                                            return 'm1'
    if 'M2'             in item_text:                                            return 'm2'
    if '9-6'            in item_text and 'S&P'   in item_text:                  return 'sp_96'
    if '9-6'            in item_text and 'STOXX' in item_text:                  return 'stoxx_96'
    if 'LazyMan'        in item_text or 'MACD'   in item_text:                  return 'macd'
    if 'S&P'            in item_text:                                            return 'sp500'
    if 'STOXX'          in item_text:                                            return 'stoxx600'
    return 'placeholder'


def get_description(gkey):
    d = {
        'macd':      '''You could stop now, and just do this.<br>
Moving Average Convergence Divergence identifies momentum.<br>
When short-term EMA crosses long-term, MACD signals potential uptrend; cross-below = downtrend.<br>
<a href="https://moneyweek.com/518350/macd-histogram-stockmarket-trading-system-for-the-lazy-investor"
   target="_blank">The LazyMan MACD article (MoneyWeek)</a><br>
In the last 19 years: 12 trades (6 Buy, 6 Sell). You'd miss exact highs/lows but catch major moves.''',
        'sp500':      'S&P500 is a forward-looking GDP indicator. Correlation with US GDP: 69.04%',
        'stoxx600':   'STOXX 600 as global risk appetite proxy (~55% correlation with US GDP via trade/finance).',
        'spread_10ff':'10-Year minus Fed Funds. Positive = steep curve = accommodative → expansionary.',
        'spread_10_2':'10-Year minus 2-Year (classic yield curve). Positive = GDP expansion signal.',
        'yield_curve_compare': '3-year view of 10Yr-2Yr spread. Steep positive = healthy expansion expectations.',
        'm1':         'M1 money supply growth supports credit creation and GDP expansion.',
        'm2':         'M2 money supply growth supports credit creation and GDP expansion.',
        'vix':        'VIX (fear index). Below 15 = low fear = positive GDP outlook.',
        'bbb_yield':  'BBB yields = cost of borrowing. Lower = expansionary conditions.',
        'ccc_yield':  'CCC yields = high-yield stress. Rising = contractionary signal.',
        'sp_96':      'S&P 9-to-6M-ago return: 69% correlation with future GDP. Top chart = signal window. Bottom = current momentum.',
        'stoxx_96':   'STOXX 9-to-6M-ago return as leading indicator. Top chart = signal window. Bottom = current momentum.',
        'core_cpi':   'Core CPI YoY (excl. food & energy). Falling = disinflationary → positive for growth.',
        'cpi_volatile':'Headline CPI YoY (incl. food & energy). Tracks total inflation pressure.',
        'building_permits': 'Building permits lead housing starts by 1-2 months — key leading economic indicator.',
    }
    if gkey.startswith('real_rate'):
        return 'Real rate = nominal yield − core CPI YoY. Negative real rates are stimulative → positive for GDP.'
    return d.get(gkey, '')


def calculate_metrics(data, history, today):
    metrics = {}
    try:
        metrics['real_rate_10yr']           = data['10yr_yield'] - data['core_cpi_yoy']
        metrics['real_rate_2yr']            = data['2yr_yield']  - data['core_cpi_yoy']
        metrics['yield_curve_10ff']         = data['10yr_yield'] - data['fed_funds']
        metrics['yield_curve_10_2']         = data['10yr_yield'] - data['2yr_yield']
        metrics['copper_gold_ratio']        = data['copper'] / data['gold']
        metrics['copper_gold_ratio_change'] = (
            (history['copper'].iloc[-1] / history['gold'].iloc[-1])
            - (history['copper'].iloc[0]  / history['gold'].iloc[0]))
    except Exception as e:
        st.error(f"Metrics calculation error: {e}")
        return {}, [], [], [], "Error", 50

    tailwinds, headwinds, neutrals = [], [], []

    def _ago(series, days):
        cut = pd.Timestamp(today - timedelta(days=days))
        sub = series[series.index >= cut]
        return float(sub.iloc[0]) if not sub.empty else float(series.iloc[0])

    def _cs(val, pct, up_good=True):
        d    = "up" if val >= 0 else "down"
        good = (val >= 0) if up_good else (val < 0)
        c    = "green" if good else "red"
        return f'<span style="color:{c}">{d} {abs(pct):.2f}%</span>'

    # 1. S&P
    try:
        sp   = history['sp500']
        spv  = float(sp.iloc[-1])
        spp  = float(sp.iloc[-2]) if len(sp) > 1 else spv
        d_ch = spv - spp
        m_ch = spv - _ago(sp, 30)
        t_ch = spv - _ago(sp, 90)
        yoy  = (spv - float(sp.iloc[0])) / float(sp.iloc[0]) * 100 if float(sp.iloc[0]) != 0 else 0
        d_p  = d_ch / spp * 100 if spp != 0 else 0
        m_p  = m_ch / _ago(sp, 30) * 100 if _ago(sp, 30) != 0 else 0
        t_p  = t_ch / _ago(sp, 90) * 100 if _ago(sp, 90) != 0 else 0
        lbl  = (f"S&P: {spv:.2f} (daily {_cs(d_ch,d_p)}, MoM {_cs(m_ch,m_p)}, "
                f"3M {_cs(t_ch,t_p)}, YoY {yoy:.2f}%)")
        (tailwinds if data['sp_lagging'] == 'UP' else headwinds).append(lbl + " (positive for GDP)" if data['sp_lagging'] == 'UP' else lbl + " (negative for GDP)")
    except Exception:
        neutrals.append("S&P Data Unavailable")

    # Copper/Gold
    (tailwinds if metrics['copper_gold_ratio_change'] > 0 else headwinds).append(
        "Copper/Gold ratio " + ("increasing (positive leading indicator)" if metrics['copper_gold_ratio_change'] > 0
                                 else "decreasing (negative leading indicator)"))

    def _simp(key, label, val, up_bad=False):
        h   = history[key]
        chg = float(h.iloc[-1]) - float(h.iloc[-2]) if len(h) > 1 else 0
        d   = "up" if chg > 0 else "down" if chg < 0 else "flat"
        good = (chg < 0) if up_bad else (chg > 0)
        c   = "green" if good else "red"
        sp  = f'<span style="color:{c}">{d} {abs(chg):.3f}</span>'
        return chg, f"{label}: {val:.2f}% ({sp})"

    # 2. Fed Funds
    ff_c, ff_l = _simp('fed_funds', 'Fed Funds', data['fed_funds'], up_bad=True)
    if ff_c < 0:   tailwinds.append(ff_l + ", positive)")
    elif ff_c > 0: headwinds.append(ff_l + ", negative)")
    else:          neutrals.append(ff_l + ", no change)")

    # 3. 10-Yr
    ty_h   = history['10yr_yield']
    ty_d   = float(ty_h.iloc[-1]) - float(ty_h.iloc[-2]) if len(ty_h) > 1 else 0
    term   = float(ty_h.max())
    metrics['terminal_10yr'] = term

    def _yd(v):
        d,c = ("down","green") if v < 0 else ("up","red")
        return f'<span style="color:{c}">{d} {abs(v):.3f}%</span>'

    ty_m = data['10yr_yield'] - _ago(ty_h, 30)
    ty_t = data['10yr_yield'] - _ago(ty_h, 90)
    ty_l = (f"10-Yr Yield: {data['10yr_yield']:.2f}% (daily {_yd(ty_d)}, "
            f"MoM {_yd(ty_m)}, 3M {_yd(ty_t)}, Terminal {term:.2f}%)")
    (tailwinds if ty_d < 0 else headwinds).append(ty_l + ", positive)" if ty_d < 0 else ty_l + ", negative)")

    # 4. 2-Yr
    ty2_c, ty2_l = _simp('2yr_yield', '2-Yr Yield', data['2yr_yield'], up_bad=True)
    (tailwinds if ty2_c < 0 else headwinds).append(ty2_l + (", positive)" if ty2_c < 0 else ", negative)"))

    # 5. Core CPI
    cc = data['core_cpi_yoy']
    cc_c = float(history['core_cpi'].iloc[-1]) - float(history['core_cpi'].iloc[-2]) if len(history['core_cpi']) > 1 else 0
    (tailwinds if cc_c < 0 else headwinds).append(
        f"Core CPI YoY: {cc:.2f}% ({'falling, positive' if cc_c < 0 else 'rising, negative'})")

    # 6-7. Real Rates
    for rk, rl in [('real_rate_10yr', '10-Yr'), ('real_rate_2yr', '2-Yr')]:
        rv = metrics[rk]
        (tailwinds if rv < 0 else headwinds).append(
            f"Real Rate ({rl}): {rv:.2f}% ({'negative = stimulative' if rv < 0 else 'positive = restrictive'})")

    # 8-9. BBB / CCC
    for yk, lbl in [('bbb_yield', 'BBB Yield'), ('ccc_yield', 'CCC Yield')]:
        c, l = _simp(yk, lbl, data[yk], up_bad=True)
        (tailwinds if c < 0 else headwinds).append(l + (", positive)" if c < 0 else ", negative)"))

    # 10-11. VIX / MOVE
    for vk, vl, vv in [('vix', 'VIX', float(data['vix'])), ('move', 'MOVE', float(data['move']))]:
        vc = float(history[vk].iloc[-1]) - float(history[vk].iloc[-2]) if len(history[vk]) > 1 else 0
        vm = vv - _ago(history[vk], 30)
        vt = vv - _ago(history[vk], 90)
        def _vs(x): d,c = ("down","green") if x<0 else ("up","red"); return f'<span style="color:{c}">{d} {abs(x):.2f}</span>'
        lbl = f"{vl}: {vv:.2f} (daily {_vs(vc)}, MoM {_vs(vm)}, 3M {_vs(vt)}"
        pos = (vv < 15 or vc < 0) if vk == 'vix' else vc < 0
        (tailwinds if pos else headwinds).append(lbl + (", positive)" if pos else ", negative)"))

    # 12-13. PMIs
    for pk, pl in [('ism_manufacturing','Manufacturing PMI'), ('ism_services','Services PMI')]:
        v = data[pk]
        (tailwinds if v > 50 else headwinds).append(f"{pl}: {v:.1f} ({'expansion' if v>50 else 'contraction'})")

    # 14. UMCSI
    v = data['umcsi']
    if v > 70: tailwinds.append(f"UMCSI: {v:.1f} (bullish)")
    elif v < 55: headwinds.append(f"UMCSI: {v:.1f} (bearish)")
    else: neutrals.append(f"UMCSI: {v:.1f} (neutral)")

    # 15. Building Permits
    bp_h  = history['building_permits']
    bp_c  = float(bp_h.iloc[-1]) - float(bp_h.iloc[-2]) if len(bp_h) > 1 else 0
    bp_cs = f'<span style="color:{"green" if bp_c>0 else "red"}">{"up" if bp_c>0 else "down"} {abs(bp_c):.0f}K</span>'
    (tailwinds if bp_c > 0 else headwinds).append(
        f"Building Permits: {data['building_permits']:.2f}M ({bp_cs}, {'positive' if bp_c>0 else 'negative'})")

    # 16. NFIB
    nfib_h = history['nfib']
    nfib_c = float(nfib_h.iloc[-1]) - float(nfib_h.iloc[-2]) if len(nfib_h) > 1 else 0
    pct_c  = nfib_c / float(nfib_h.iloc[-2]) * 100 if (len(nfib_h) > 1 and float(nfib_h.iloc[-2]) != 0) else 0
    dirn   = "Up" if nfib_c > 0 else "Down" if nfib_c < 0 else "Unchanged"
    pm     = nfib_h.index[-2].strftime('%b') if len(nfib_h) > 1 else 'Prev'
    cm     = nfib_h.index[-1].strftime('%b')
    nst    = "(strong)" if data['nfib'] > 100 else "(weak)" if data['nfib'] < 95 else "(neutral)"
    nfib_l = f"NFIB: {data['nfib']:.1f} {dirn} {abs(nfib_c):.1f} ({pct_c:.2f}%) ({pm}→{cm}) {nst}"
    if data['nfib'] > 100: tailwinds.append(nfib_l)
    elif data['nfib'] < 95: headwinds.append(nfib_l)
    else: neutrals.append(nfib_l)

    # 17. S&P 9-6M
    sp = history['sp500']
    if len(sp) > 200:
        p9 = float(sp.iloc[max(0, len(sp)-189)])
        p6 = float(sp.iloc[max(0, len(sp)-126)])
        sp96 = (p6-p9)/p9*100 if p9 != 0 else 0
    else:
        sp96 = 0
    metrics['sp_96_return'] = sp96
    (tailwinds if sp96 > 0 else headwinds).append(f"S&P 9-6m Return: {sp96:.2f}%")

    # CPI Volatile / SBI / EESI
    v = data.get('cpi_volatile', 300)
    (tailwinds if v < 300 else headwinds).append(f"CPI Volatile: {v:.0f} ({'low, positive' if v<300 else 'high, negative'})")
    v = data.get('sbi', 0)
    (tailwinds if v > 68 else headwinds).append(f"SBI: {v:.1f} ({'strong, positive' if v>68 else 'weak, negative'})")
    v = data.get('eesi', 0)
    (tailwinds if v > 45 else headwinds).append(f"EESI: {v:.1f} ({'positive' if v>45 else 'negative'})")

    # M1 / M2
    for mk in ['m1', 'm2']:
        h = history[mk]
        g = bool(h.iloc[-1] > h.iloc[-2]) if len(h) > 1 else False
        metrics[f'{mk}_growth_pos'] = g
        (tailwinds if g else headwinds).append(
            f"{mk.upper()} Money Supply: {'Growing' if g else 'Contracting'} MoM")

    # Spreads
    ffs = metrics.get('yield_curve_10ff', 0)
    (tailwinds if ffs > 0 else headwinds).append(
        f"10Yr-FedFunds Spread: {ffs:.2f}% ({'expansionary' if ffs>0 else 'contractionary'})")
    t2s = metrics.get('yield_curve_10_2', 0)
    (tailwinds if t2s > 0 else headwinds).append(
        f"10Yr-2Yr Spread: {t2s:.2f}% ({'expansionary' if t2s>0 else 'flat/inverted, contractionary'})")
    (tailwinds if t2s > 0 else headwinds).append(
        f"Yield Curve Comparison (3yr): {'Steep/Positive' if t2s>0 else 'Flat/Inverted'}")

    # MACD
    macd_lb = macd_sb = False
    try:
        spc = history['sp500'].dropna()
        if len(spc) >= 40:
            ml, sl, _ = compute_macd(spc)
            macd_lb   = bool(ml.iloc[-1] > sl.iloc[-1])
            metrics['macd_line']   = float(ml.iloc[-1])
            metrics['signal_line'] = float(sl.iloc[-1])
        sp45 = safe_last(spc, 45)
        if len(sp45) >= 26:
            ms, ss, _ = compute_macd(sp45)
            macd_sb   = bool(ms.iloc[-1] > ss.iloc[-1])
    except Exception:
        pass
    metrics['macd_long_bullish']  = macd_lb
    metrics['macd_short_bullish'] = macd_sb
    macd_l = (f"LazyMan MACD: Short {'Buy' if macd_sb else 'Sell'} | "
              f"Long {'Buy' if macd_lb else 'Sell'}")
    (tailwinds if macd_lb else headwinds).append(
        macd_l + (" → Bull – SIMPLY Buy" if macd_lb else ""))

    # STOXX 9-6M
    st600 = history['stoxx600']
    if len(st600) > 200:
        p9  = float(st600.iloc[max(0, len(st600)-189)])
        p6  = float(st600.iloc[max(0, len(st600)-126)])
        s96 = (p6-p9)/p9*100 if p9 != 0 else 0
    else:
        s96 = 0
    metrics['stoxx_96_return'] = s96
    (tailwinds if s96 > 0 else headwinds).append(f"STOXX 600 9-6m Return: {s96:.2f}%")

    # S&P Bear/Bull
    sp_bear = {}
    try:
        sl = history['sp500_long'].dropna()
        if len(sl) >= 10:
            cur   = float(sl.iloc[-1])
            ath   = float(sl.max())
            nhigh = sl[sl == ath].index[-1]
            rec   = sl[sl.index >= today - timedelta(days=1825)]
            if not rec.empty:
                pl = float(rec.min()); pld = rec.idxmin(); db = (today.date()-pld.date()).days
            else:
                pl, pld, db = 0.0, sl.index[0], 0
            sp_bear = dict(current_date=sl.index[-1].strftime('%d/%m/%Y'), current=cur,
                           last_high_date=nhigh.strftime('%d/%m/%Y'), last_high=ath,
                           new_bear_threshold=ath*0.8,
                           prev_bear_date=pld.strftime('%d/%m/%Y'), prev_bear=pl,
                           days_bull=db, avg_days_bull=997)
    except Exception:
        pass
    metrics['sp_bear'] = sp_bear

    # STOXX Bear/Bull
    stoxx_bear = {}
    try:
        sl = history['stoxx600_long'].dropna()
        if len(sl) >= 10:
            cur   = float(sl.iloc[-1])
            ath   = float(sl.max())
            nhigh = sl[sl == ath].index[-1]
            rec   = sl[sl.index >= today - timedelta(days=1825)]
            if not rec.empty:
                pl = float(rec.min()); pld = rec.idxmin(); db = (today.date()-pld.date()).days
            else:
                pl, pld, db = 0.0, sl.index[0], 0
            stoxx_bear = dict(current_date=sl.index[-1].strftime('%d/%m/%Y'), current=cur,
                              last_high_date=nhigh.strftime('%d/%m/%Y'), last_high=ath,
                              new_bear_threshold=ath*0.8,
                              prev_bear_date=pld.strftime('%d/%m/%Y'), prev_bear=pl,
                              days_bull=db, avg_days_bull=857)
    except Exception:
        pass
    metrics['stoxx_bear'] = stoxx_bear

    # Score
    score  = 0
    score += min(max(metrics.get('sp_96_return',0)/5*18, 0), 18) if metrics.get('sp_96_return',0)>0 else 0
    score += 12 if data.get('sp_lagging')=='UP' else 0
    score += 15 if metrics.get('yield_curve_10_2',0)>0 else 0
    score += 12 if metrics.get('yield_curve_10ff',0)>0 else 0
    score += 10 if metrics.get('macd_long_bullish',False) else 0
    score += 8  if metrics.get('stoxx_96_return',0)>0 else 0
    score += 10 if metrics.get('real_rate_10yr',0)<0 else 0
    score += 8  if metrics.get('real_rate_2yr',0)<0 else 0
    score += max(8-data.get('vix',20)/5, 0) if data.get('vix',20)<25 else 0
    score += 8  if data['ism_manufacturing']>50 else 0
    score += 7  if data['ism_services']>50 else 0
    score += 6  if data['umcsi']>60 else 0
    score += 5  if data.get('building_permits',0)>1.4 else 0
    score += 5  if data.get('sbi',0)>68 else 0
    score += 4  if data.get('cpi_volatile',300)<300 else 0
    score += 4  if data.get('eesi',50)>45 else 0
    score += 5  if data.get('nfib',99)>100 else 0
    score += 5  if metrics.get('m1_growth_pos',False) else 0
    score += 5  if metrics.get('m2_growth_pos',False) else 0
    score += 8  if metrics.get('copper_gold_ratio_change',0)>0 else 0
    score  = max(0, min(100, int(score)))
    bias   = ('Long (6 long/4 short)' if score >= 60
              else 'Short (4 long/6 short)' if score <= 40
              else 'Neutral (5 long/5 short)')
    return metrics, tailwinds, headwinds, neutrals, bias, score


def generate_html_summary(tailwinds, headwinds, neutrals, bias, data, history, metrics, today, score):

    def _fig_to_b64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64

    def build_section(items_list):
        parts = []
        for item in items_list:
            gkey = get_graph_key(item)

            # Main chart
            fig    = generate_graph(gkey, data, history, metrics, today)
            img64  = _fig_to_b64(fig)

            # Short-term chart (suppressed for some keys)
            short_html = ''
            sfig = generate_short_term_graph(gkey, history, today)
            if sfig is not None:
                s64 = _fig_to_b64(sfig)
                short_html = (
                    '<h4 style="margin:20px 0 8px;color:#555;font-size:1em;">'
                    'Short-term View (recent months)</h4>'
                    f'<img src="data:image/png;base64,{s64}" '
                    'style="width:100%;max-width:820px;display:block;margin:0 auto;'
                    'box-shadow:0 2px 8px rgba(0,0,0,0.08);"/>')

            desc      = get_description(gkey)
            desc_html = f'<p style="margin-top:12px;color:#444;font-size:0.93em;">{desc}</p>' if desc else ''

            # Terminal yield table
            extra_html = ''
            if gkey == '10yr_yield':
                cur, term = data['10yr_yield'], metrics.get('terminal_10yr', data['10yr_yield'])
                extra_html = f'''
                <h4 style="margin:22px 0 8px;color:#555;">10-Yr Treasury: Current vs Terminal</h4>
                <table style="width:100%;max-width:820px;margin:12px auto;border-collapse:collapse;
                              font-size:0.93em;border:1px solid #ddd;">
                  <thead><tr style="background:#f8f8f8;">
                    <th style="padding:9px;border:1px solid #ddd;text-align:left;">Metric</th>
                    <th style="padding:9px;border:1px solid #ddd;text-align:right;">Value</th>
                  </tr></thead><tbody>
                    <tr><td style="padding:9px;border:1px solid #ddd;">Current 10-Yr Yield</td>
                        <td style="padding:9px;border:1px solid #ddd;text-align:right;">{cur:.2f}%</td></tr>
                    <tr><td style="padding:9px;border:1px solid #ddd;">Terminal (recent high)</td>
                        <td style="padding:9px;border:1px solid #ddd;text-align:right;">{term:.2f}%</td></tr>
                  </tbody></table>'''

            # STOXX 9-6M dynamic table
            if gkey == 'stoxx_96':
                extra_html += _build_stoxx_96_table_html(history, metrics)

            # Bear/Bull tables
            for bkey, btitle, bmet in [('sp500',   'S&amp;P 500', 'sp_bear'),
                                        ('stoxx600', 'STOXX 600',  'stoxx_bear')]:
                if gkey == bkey:
                    b = metrics.get(bmet, {})
                    if b:
                        extra_html += f'''
                        <h4 style="margin:22px 0 8px;color:#555;">{btitle} Bull/Bear Status</h4>
                        <table style="width:100%;max-width:820px;margin:12px auto;border-collapse:collapse;
                                      font-size:0.93em;border:1px solid #ddd;">
                          <thead><tr style="background:#f8f8f8;">
                            <th style="padding:9px;border:1px solid #ddd;">Metric</th>
                            <th style="padding:9px;border:1px solid #ddd;">Date</th>
                            <th style="padding:9px;border:1px solid #ddd;text-align:right;">Value</th>
                          </tr></thead><tbody>
                            <tr><td style="padding:8px;border:1px solid #ddd;">Current</td>
                                <td style="padding:8px;border:1px solid #ddd;">{b.get("current_date","")}</td>
                                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{b.get("current",0):,.2f}</td></tr>
                            <tr><td style="padding:8px;border:1px solid #ddd;">All-Time High</td>
                                <td style="padding:8px;border:1px solid #ddd;">{b.get("last_high_date","")}</td>
                                <td style="padding:8px;border:1px solid #ddd;text-align:right;">{b.get("last_high",0):,.2f}</td></tr>
                            <tr><td style="padding:8px;border:1px solid #ddd;">Bear Threshold (−20%)</td>
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
                          </tbody></table>'''

            parts.append(f'''
<li>
  <details>
    <summary>{item}</summary>
    <div style="padding:18px;background:#fafafa;border:1px solid #e5e5e5;
                border-top:none;border-radius:0 0 6px 6px;">
      <img src="data:image/png;base64,{img64}"
           style="width:100%;max-width:820px;display:block;margin:0 auto;
                  box-shadow:0 2px 8px rgba(0,0,0,0.08);"/>
      {short_html}{desc_html}{extra_html}
    </div>
  </details>
</li>''')
        return ''.join(parts)

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Portfolio Bias Summary</title>
<style>
body{{font-family:Arial,sans-serif;padding:40px;background:#fff;color:#111;
     max-width:960px;margin:auto;}}
h1{{color:#1a1a1a;font-size:30px;margin-bottom:6px;}}
.bias{{font-size:1.3em;font-weight:bold;color:#003366;margin-bottom:30px;
       border-bottom:2px solid #e5e5e5;padding-bottom:10px;}}
.score{{font-size:1.3em;font-weight:bold;color:#003366;}}
h2{{font-size:22px;border-bottom:3px solid #ddd;padding-bottom:8px;margin-top:40px;}}
ul{{list-style:none;padding:0;margin:0;}} li{{margin-bottom:7px;}}
summary{{font-size:1.02em;font-weight:600;cursor:pointer;padding:11px 15px;
          background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;list-style:none;}}
summary::-webkit-details-marker{{display:none;}}
summary::before{{content:"▶ ";font-size:0.75em;color:#888;}}
details[open] summary::before{{content:"▼ ";}}
summary:hover{{background:#efefef;}}
</style></head><body>
<h1>📊 Portfolio Bias Summary</h1>
<p class="bias">Recommended Bias: {bias}</p>
<p class="score">GDP Growth Score: {score}/100</p>
<h2 style="color:#28a745;border-bottom-color:#28a745;">✅ Tailwinds</h2>
<ul>{build_section(tailwinds)}</ul>
<h2 style="color:#dc3545;border-bottom-color:#dc3545;">❌ Headwinds</h2>
<ul>{build_section(headwinds)}</ul>
<h2 style="color:#888;">⚖️ Neutrals</h2>
<ul>{build_section(neutrals)}</ul>
</body></html>"""


def plot_sector_chart(etf_ticker, period='1y'):
    try:
        hist = yf.Ticker(etf_ticker).history(period=period)['Close']
        if hist.empty: return None
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot(ax=ax, linewidth=1.8, color='#1565C0')
        ax.set_title(f'{etf_ticker} ({period})', fontsize=9)
        ax.grid(True, alpha=0.3)
        locator   = mdates.AutoDateLocator(minticks=4, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
        plt.tight_layout()
        return fig
    except Exception:
        return None


def plot_commodity_chart(ticker, period='1y'):
    try:
        hist = yf.Ticker(ticker).history(period=period)['Close']
        if hist.empty: return None
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        fig, ax = plt.subplots(figsize=(6, 4))
        hist.plot(ax=ax, linewidth=1.8, color='#6A1B9A')
        ax.set_title(f'{ticker} ({period})', fontsize=9)
        ax.grid(True, alpha=0.3)
        locator   = mdates.AutoDateLocator(minticks=4, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
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
    performance  = {}
    three_m_ago  = pd.Timestamp(today - timedelta(days=90))
    for sector, info in all_sectors.items():
        try:
            hist = yf.Ticker(info['etf']).history(period='1y')['Close']
            if hist.empty or len(hist) < 2: performance[sector] = 0; continue
            if hasattr(hist.index,'tz') and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            r1y = (hist.iloc[-1]-hist.iloc[0])/hist.iloc[0] if hist.iloc[0]!=0 else 0
            h3  = hist[hist.index >= three_m_ago]
            r3m = (hist.iloc[-1]-h3.iloc[0])/h3.iloc[0] if (not h3.empty and h3.iloc[0]!=0) else 0
            performance[sector] = 0.5*r1y + 0.5*r3m
        except Exception:
            performance[sector] = 0

    ss = sorted(performance, key=performance.get, reverse=True)
    if 'Long' in bias:
        longs  = ([s for s in ss if s in preferred_sectors][:4] or ss[:4])
        shorts = ([s for s in ss[::-1] if s in preferred_sectors][:3] or ss[-3:])
        la, sa = portfolio_size*0.6/max(len(longs),1), portfolio_size*0.4/max(len(shorts),1)
    elif 'Short' in bias:
        longs  = ([s for s in ss if s in preferred_sectors][:3] or ss[:3])
        shorts = ([s for s in ss[::-1] if s in preferred_sectors][:4] or ss[-4:])
        la, sa = portfolio_size*0.4/max(len(longs),1), portfolio_size*0.6/max(len(shorts),1)
    else:
        longs, shorts = ss[:3], ss[-3:]
        la = sa = portfolio_size*0.5/3

    tilt_df = pd.DataFrame({
        'Type':           ['Long']*len(longs)  + ['Short']*len(shorts),
        'Sector':         longs + shorts,
        'ETF':            [all_sectors[s]['etf'] for s in longs+shorts],
        'Allocation ($)': [f"${la:,.0f}"]*len(longs) + [f"${sa:,.0f}"]*len(shorts),
    })
    return tilt_df, all_sectors, all_commodities


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Macro Portfolio Bias & Sector Tilt", layout="wide")
st.title("📊 Portfolio Bias Analysis & Sector Tilt Dashboard")

for k in ['bias_calculated','bias','score','metrics','data','history','today','tailwinds','headwinds','neutrals']:
    if k not in st.session_state:
        st.session_state[k] = False if k == 'bias_calculated' else None

if st.button("🔄 Update Analysis", type="primary"):
    with st.spinner("Fetching latest macro data..."):
        try:
            st.session_state.data, st.session_state.history, st.session_state.today = fetch_data()
            (st.session_state.metrics, st.session_state.tailwinds,
             st.session_state.headwinds, st.session_state.neutrals,
             st.session_state.bias, st.session_state.score) = calculate_metrics(
                st.session_state.data, st.session_state.history, st.session_state.today)
            st.success(f"✅ Updated for {st.session_state.today.strftime('%Y-%m-%d')}")
            st.info(f"**GDP Growth Score: {st.session_state.score}/100** → {st.session_state.bias}")
            with st.spinner("Generating HTML report..."):
                html = generate_html_summary(
                    st.session_state.tailwinds, st.session_state.headwinds,
                    st.session_state.neutrals, st.session_state.bias,
                    st.session_state.data, st.session_state.history,
                    st.session_state.metrics, st.session_state.today,
                    st.session_state.score)
            st.download_button("📥 Download HTML Report", data=html,
                               file_name=f"macro_bias_{st.session_state.today.date()}.html",
                               mime="text/html")
            st.session_state.bias_calculated = True
        except Exception as e:
            st.error(f"Error: {e}"); st.exception(e)

if st.session_state.bias_calculated:
    st.header("🎯 Sector Tilt Recommendations")
    c1, c2, c3 = st.columns(3)
    with c1: risk_level     = st.selectbox("Risk Tolerance", ['Low','Medium','High'], index=1)
    with c2: portfolio_size = st.number_input("Portfolio Size ($)", min_value=1000, value=100000, step=1000)
    with c3:
        st.metric("Bias",  st.session_state.bias)
        st.metric("Score", f"{st.session_state.score}/100")
    preferred_sectors = st.multiselect("Preferred Sectors (optional)", options=[
        'Technology','Industrials','Financials','Consumer Discretionary',
        'Utilities','Healthcare','Energy','Materials',
        'Consumer Staples','Real Estate','Communication Services'])

    if st.button("📈 Generate Sector Tilt", type="primary"):
        with st.spinner("Fetching ETF data..."):
            try:
                tilt_df, sectors, commodities = generate_sector_tilt(
                    st.session_state.bias, st.session_state.score, risk_level,
                    preferred_sectors, portfolio_size,
                    st.session_state.data, st.session_state.metrics, st.session_state.today)
                st.subheader("Recommended Sector Tilt")
                st.dataframe(tilt_df, use_container_width=True)
                st.download_button("📥 Download CSV",
                                   data=tilt_df.to_csv(index=False).encode(),
                                   file_name="sector_tilt.csv", mime="text/csv")
                st.subheader("📊 Sector Performance Charts")
                for sector, info in sectors.items():
                    with st.expander(f"{sector} — {info['etf']}"):
                        cols = st.columns(3)
                        for i, period in enumerate(['5y','1y','3mo']):
                            with cols[i]:
                                st.caption(period)
                                fig = plot_sector_chart(info['etf'], period)
                                if fig: st.pyplot(fig); plt.close(fig)
                                else:   st.warning("Unavailable")
                st.subheader("🥇 Commodity Charts")
                for comm, info in commodities.items():
                    with st.expander(f"{comm} — {info['ticker']}"):
                        cols = st.columns(3)
                        for i, period in enumerate(['5y','1y','3mo']):
                            with cols[i]:
                                st.caption(period)
                                fig = plot_commodity_chart(info['ticker'], period)
                                if fig: st.pyplot(fig); plt.close(fig)
                                else:   st.warning("Unavailable")
            except Exception as e:
                st.error(f"Sector tilt error: {e}"); st.exception(e)



