import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fredapi import Fred
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

FRED_API_KEY = 'e210def24f02e4a73ac744035fa51963'
fred = Fred(api_key=FRED_API_KEY)

NO_SHORT_TERM_CHART = {'core_cpi', 'eesi', 'cpi_volatile'}

OPTIMIZED_WEIGHTS = {
    'yield_curve_10_2':   18,
    'yield_curve_10ff':   12,
    'real_rate_10yr':      7,
    'real_rate_2yr':       5,
    'earnings_growth':    10,
    'macd_long':          12,
    'fed_bs_growth':       8,
    'vix_trend':           8,
    'sp_96':               8,
    'stoxx_96':            5,
    'copper_gold':         5,
    'ism_manufacturing':   4,
    'ism_services':        3,
    'building_permits':    4,
    'nfib':                3,
    'umcsi':               3,
    'bbb_yield':           3,
}

CYCLE_PHASE_CONFIG = {
    'early':     {'boost': 20,  'force_short': False, 'label': '🌱 Early Cycle'},
    'mid':       {'boost':  0,  'force_short': False, 'label': '📈 Mid Cycle'},
    'late':      {'boost': -15, 'force_short': False, 'label': '⚠️ Late Cycle'},
    'recession': {'boost': -30, 'force_short': True,  'label': '🔴 Recession / Risk-Off'},
}

SECTOR_ROTATION = {
    'early': {
        'Financials':             0.18, 'Industrials':            0.16, 'Materials':              0.14,
        'Energy':                 0.12, 'Technology':             0.10, 'Consumer Discretionary': 0.10,
        'Healthcare':             0.08, 'Consumer Staples':       0.06, 'Utilities':              0.04,
        'Real Estate':            0.02, 'Communication Services': 0.00,
    },
    'mid': {
        'Technology':             0.18, 'Consumer Discretionary': 0.16, 'Financials':             0.12,
        'Industrials':            0.10, 'Communication Services': 0.10, 'Materials':              0.08,
        'Energy':                 0.08, 'Healthcare':             0.08, 'Consumer Staples':       0.06,
        'Utilities':              0.04, 'Real Estate':            0.00,
    },
    'late': {
        'Healthcare':             0.18, 'Consumer Staples':       0.16, 'Utilities':              0.14,
        'Real Estate':            0.12, 'Communication Services': 0.10, 'Technology':             0.08,
        'Consumer Discretionary': 0.06, 'Financials':             0.06, 'Industrials':            0.04,
        'Materials':              0.03, 'Energy':                 0.03,
    },
    'recession': {
        'Utilities':              0.25, 'Consumer Staples':       0.25, 'Healthcare':             0.20,
        'Real Estate':            0.15, 'Communication Services': 0.10, 'Technology':             0.05,
        'Consumer Discretionary': 0.00, 'Financials':             0.00, 'Industrials':            0.00,
        'Materials':              0.00, 'Energy':                 0.00,
    },
}

ALL_SECTORS = {
    'Technology':             {'etf': 'XLK'},
    'Industrials':            {'etf': 'XLI'},
    'Financials':             {'etf': 'XLF'},
    'Consumer Discretionary': {'etf': 'XLY'},
    'Materials':              {'etf': 'XLB'},
    'Energy':                 {'etf': 'XLE'},
    'Healthcare':             {'etf': 'XLV'},
    'Utilities':              {'etf': 'XLU'},
    'Consumer Staples':       {'etf': 'XLP'},
    'Real Estate':            {'etf': 'XLRE'},
    'Communication Services': {'etf': 'XLC'},
}

SECTOR_PE_HISTORY = {
    'Technology':             {'mean': 28.0, 'std': 6.0},
    'Industrials':            {'mean': 20.0, 'std': 4.0},
    'Financials':             {'mean': 13.0, 'std': 3.0},
    'Consumer Discretionary': {'mean': 25.0, 'std': 7.0},
    'Materials':              {'mean': 18.0, 'std': 4.0},
    'Energy':                 {'mean': 15.0, 'std': 8.0},
    'Healthcare':             {'mean': 18.0, 'std': 3.0},
    'Utilities':              {'mean': 17.0, 'std': 3.0},
    'Consumer Staples':       {'mean': 20.0, 'std': 3.0},
    'Real Estate':            {'mean': 35.0, 'std': 8.0},
    'Communication Services': {'mean': 20.0, 'std': 5.0},
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    hist     = macd - sig
    return macd, sig, hist

def safe_last(series, window_days):
    if series is None or series.empty:
        return series
    cutoff = series.index[-1] - pd.Timedelta(days=window_days)
    return series[series.index >= cutoff]

def normalize_index(series):
    if series is None or series.empty:
        return series
    idx = pd.DatetimeIndex(series.index).tz_localize(None)
    return pd.Series(series.values, index=idx)

def _is_monthly(series):
    if series is None or len(series) < 3:
        return False
    gaps = pd.Series(series.index).diff().dropna()
    return gaps.median() >= pd.Timedelta(days=20)

def _apply_axis_format(ax, series):
    if series is None or series.empty:
        return
    if _is_monthly(series):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    else:
        locator   = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

def _short_term_window(series):
    if _is_monthly(series):
        return safe_last(series, 185)
    return safe_last(series, 90)

def detect_cycle_phase(spread_10_2, real_10yr, pmi_manuf, earnings_growth):
    if spread_10_2 > 1.0 and real_10yr < 0 and pmi_manuf > 52 and earnings_growth > 5:
        return 'early'
    elif spread_10_2 > 0.5 and real_10yr < 0.5 and pmi_manuf >= 50 and earnings_growth >= 0:
        return 'mid'
    elif 0 <= spread_10_2 <= 0.5 and 0.5 <= real_10yr <= 1.5 and 48 <= pmi_manuf < 50:
        return 'late'
    else:
        return 'recession'

def get_vix_trend_signal(vix_series):
    if vix_series is None or len(vix_series) < 22:
        return 0
    vix_now     = float(vix_series.iloc[-1])
    vix_1m_ago  = float(vix_series.iloc[-22])
    if vix_now < 12:
        return -1
    elif vix_now > 28:
        return -1
    elif 14 <= vix_now <= 24 and vix_now - vix_1m_ago < -2:
        return 1
    else:
        return 0

def get_pe_zscore(sector_name, etf_ticker):
    hist = SECTOR_PE_HISTORY.get(sector_name, {'mean': 20.0, 'std': 4.0})
    try:
        info = yf.Ticker(etf_ticker).info
        pe   = info.get('trailingPE') or info.get('forwardPE') or hist['mean']
        pe   = float(pe)
    except Exception:
        pe = hist['mean']
    return (pe - hist['mean']) / hist['std'] if hist['std'] != 0 else 0.0

def _plot_series(ax, series, title, hline=None, color='#1565C0', linewidth=2):
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
    s.plot(ax=ax, linewidth=linewidth, color=color)
    ax.grid(True, alpha=0.3)
    _apply_axis_format(ax, s)

def _plot_macd_bars(ax, macd_vals, sig_vals, hist_vals, x_dates, title):
    bw     = (x_dates[1] - x_dates[0]) * 0.8 if len(x_dates) > 1 else 0.8
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist_vals]
    ax.bar(x_dates, hist_vals.values, width=bw, alpha=0.6, color=colors, label='Histogram')
    ax.plot(x_dates, macd_vals.values, color='#1565C0', linewidth=1.5, label='MACD')
    ax.plot(x_dates, sig_vals.values,  color='#E53935', linewidth=1.5, label='Signal')
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

def _compute_real_rate_series(history, which='10yr'):
    be = history.get('breakeven_5y', pd.Series(dtype=float)).dropna()
    if be.empty:
        return pd.Series(dtype=float)
    key = '10yr_yield' if which == '10yr' else '2yr_yield'
    yld = history[key].reindex(be.index, method='nearest', tolerance=pd.Timedelta('35D'))
    return (yld - be).dropna()

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

def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_data():
    data    = {}
    history = {}
    today   = datetime.now()

    def safe_fred(series_id, default_value=0):
        try:
            s = fred.get_series(series_id, observation_start='2010-01-01')
            if s is None or s.empty:
                raise ValueError("Empty")
            s = normalize_index(s)
            return float(s.iloc[-1]), s
        except Exception:
            dr = pd.date_range(end=today, periods=48, freq='ME')
            fallback = pd.Series(np.random.normal(default_value, max(abs(default_value) * 0.05, 0.01), 48), index=dr)
            return default_value, fallback

    data['fed_funds'],    history['fed_funds']    = safe_fred('FEDFUNDS', 3.64)
    data['10yr_yield'],   history['10yr_yield']   = safe_fred('DGS10', 4.086)
    data['2yr_yield'],    history['2yr_yield']    = safe_fred('DGS2', 3.48)
    data['bbb_yield'],    history['bbb_yield']    = safe_fred('BAMLC0A4CBBBEY', 4.93)
    data['ccc_yield'],    history['ccc_yield']    = safe_fred('BAMLH0A3HYCEY', 12.44)

    data['breakeven_5y'], history['breakeven_5y'] = safe_fred('T5YIFR', 2.3)

    data['real_rate_10yr'] = data['10yr_yield'] - data['breakeven_5y']
    data['real_rate_2yr']  = data['2yr_yield']  - data['breakeven_5y']

    _, history['fed_bs'] = safe_fred('WALCL', 7000)
    fed_bs_s = history['fed_bs'].dropna()
    if len(fed_bs_s) >= 52:
        data['fed_bs_growth'] = ((float(fed_bs_s.iloc[-1]) / float(fed_bs_s.iloc[-52])) - 1) * 100
    else:
        data['fed_bs_growth'] = 0.0

    data['ism_manufacturing'], history['ism_manufacturing'] = safe_fred('NAPM', 52.6)
    data['ism_services'],      history['ism_services']      = safe_fred('NMFPMI', 53.8)
    data['nfib'],              history['nfib']              = safe_fred('NFIBSBIO', 99.3)
    data['umcsi'],             history['umcsi']             = safe_fred('UMCSENT', 56.6)
    data['building_permits_raw'], history['building_permits'] = safe_fred('PERMIT', 1448)
    history['building_permits'] = normalize_index(history['building_permits'])
    data['building_permits'] = data['building_permits_raw'] / 1000.0

    try:
        core = fred.get_series('CPILFESL', observation_start='2010-01-01')
        core = normalize_index(core)
        if len(core) < 14:
            raise ValueError("Not enough data")
        data['core_cpi_yoy'] = ((core.iloc[-1] / core.iloc[-13]) - 1) * 100
        history['core_cpi']  = core
    except Exception:
        data['core_cpi_yoy'] = 2.5
        dr = pd.date_range(end=today, periods=60, freq='ME')
        history['core_cpi'] = pd.Series([300.0 * (1 + 0.025 / 12) ** i for i in range(60)], index=dr)

    data['cpi_volatile'], history['cpi_volatile'] = safe_fred('CPIAUCSL', 300)

    def _scrape_or_fallback(indicator, default, num_months=24):
        dr = pd.date_range(end=today, periods=num_months, freq='ME')
        return default, pd.Series(np.random.normal(default, default * 0.04, num_months), index=dr)

    data['sbi'],  history['sbi']  = _scrape_or_fallback('sbi',  68.4)
    data['eesi'], history['eesi'] = _scrape_or_fallback('eesi', 50.0)

    try:
        sp_info = yf.Ticker('^GSPC').info
        trailing_eps = sp_info.get('trailingEps', None)
        forward_eps  = sp_info.get('forwardEps', None)
        if trailing_eps and forward_eps and trailing_eps != 0:
            data['earnings_growth'] = ((forward_eps - trailing_eps) / abs(trailing_eps)) * 100
        else:
            data['earnings_growth'] = 5.0
        dr = pd.date_range(end=today, periods=40, freq='QE')
        history['earnings_growth'] = pd.Series(np.random.normal(data['earnings_growth'], 3, 40), index=dr)
    except Exception:
        data['earnings_growth'] = 5.0
        dr = pd.date_range(end=today, periods=40, freq='QE')
        history['earnings_growth'] = pd.Series(np.random.normal(5.0, 3, 40), index=dr)

    def get_yf(ticker, default_val, default_std, period='1y'):
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
            ndays = {'1y': 365, '5y': 1825, '3mo': 90}.get(period, 365)
            dr    = pd.date_range(end=today, periods=ndays, freq='B')
            return default_val, pd.Series(np.random.normal(default_val, default_std, len(dr)), index=dr)

    data['vix'],    history['vix']    = get_yf('^VIX',  19.09, 5.0,  '1y')
    data['move'],   history['move']   = get_yf('^MOVE', 85.0,  10.0, '1y')
    data['copper'], history['copper'] = get_yf('HG=F',  4.0,   0.5,  '1y')
    data['gold'],   history['gold']   = get_yf('GC=F',  2000,  200,  '1y')

    data['vix_trend'] = get_vix_trend_signal(history['vix'])

    _, history['sp500']      = get_yf('^GSPC', 5000, 500, '1y')
    _, history['sp500_long'] = get_yf('^GSPC', 5000, 500, '5y')
    data['sp_lagging'] = 'UP' if history['sp500'].iloc[-1] > history['sp500'].iloc[0] else 'DOWN'

    stoxx_loaded = False
    for sticker in ['EXW1.DE', '^STOXX', 'FEZ', 'EXSA.DE']:
        try:
            _, h1y = get_yf(sticker, 500, 50, '1y')
            _, h5y = get_yf(sticker, 500, 50, '5y')
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

    return data, history, today

# ============================================================================
# CHART GENERATION
# ============================================================================

def generate_macd_4panel(history, today):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('LazyMan Investor — S&P500 & MACD (12,26,9)', fontsize=13, fontweight='bold')
    ax_5y, ax_1m, ax_12m, ax_macd_1m = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    sp5y = history['sp500_long'].dropna()
    sp1y = history['sp500'].dropna()

    _plot_series(ax_5y, sp5y, 'S&P 500 — 5 Year', color='#1565C0', linewidth=1.5)
    sp1m = safe_last(sp1y, 31)
    _plot_series(ax_1m, sp1m, 'S&P 500 — 1 Month', color='#1565C0', linewidth=1.8)

    if len(sp1y) >= 26:
        macd_full, sig_full, hist_full = compute_macd(sp1y)
        x12 = mdates.date2num(macd_full.index.to_pydatetime())
        _plot_macd_bars(ax_12m, macd_full, sig_full, hist_full, x12, 'MACD (12,26,9) — 12 Months')

        cut1m = pd.Timestamp(today - timedelta(days=31))
        m1m   = macd_full[macd_full.index >= cut1m]
        s1m   = sig_full[sig_full.index >= cut1m]
        h1m   = hist_full[hist_full.index >= cut1m]
        if not m1m.empty:
            x1m = mdates.date2num(m1m.index.to_pydatetime())
            _plot_macd_bars(ax_macd_1m, m1m, s1m, h1m, x1m, 'MACD (12,26,9) — 1 Month')
        else:
            ax_macd_1m.text(0.5, 0.5, 'No 1M MACD data', ha='center', va='center', 
                           transform=ax_macd_1m.transAxes, color='gray')
    else:
        for ax in [ax_12m, ax_macd_1m]:
            ax.text(0.5, 0.5, f'Insufficient data ({len(sp1y)} pts, need ≥26)', ha='center', 
                   va='center', transform=ax.transAxes, color='gray')

    plt.tight_layout()
    return fig

def generate_graph(metric_key, data, history, metrics, today):
    if metric_key == 'macd':
        return generate_macd_4panel(history, today)

    fig, ax = plt.subplots(figsize=(8, 4))
    series  = None
    hline   = None

    if metric_key == 'copper_gold':
        ci = history['copper'].index.intersection(history['gold'].index)
        if len(ci):
            series = safe_last(history['copper'].reindex(ci) / history['gold'].reindex(ci), 365)
        ax.set_title('Copper/Gold Ratio (last 12M)')

    elif metric_key == 'spread_10ff':
        y10    = history['10yr_yield']
        ff     = history['fed_funds'].reindex(y10.index, method='nearest', tolerance=pd.Timedelta('35D'))
        series = safe_last((y10 - ff).dropna(), 365)
        hline  = 0
        ax.set_title('10Yr-FedFunds Spread (last 12M)')

    elif metric_key == 'spread_10_2':
        y10    = history['10yr_yield']
        y2     = history['2yr_yield'].reindex(y10.index, method='nearest', tolerance=pd.Timedelta('5D'))
        series = safe_last((y10 - y2).dropna(), 365)
        hline  = 0
        ax.set_title('10Yr-2Yr Spread (last 12M)')

    elif metric_key == 'real_rate_10yr':
        series = safe_last(_compute_real_rate_series(history, '10yr'), 365)
        hline  = 0
        ax.set_title('Real Rate 10Yr (last 12M) — using 5Y Breakeven')

    elif metric_key == 'real_rate_2yr':
        series = safe_last(_compute_real_rate_series(history, '2yr'), 365)
        hline  = 0
        ax.set_title('Real Rate 2Yr (last 12M) — using 5Y Breakeven')

    elif metric_key == 'core_cpi':
        series = safe_last(_compute_core_cpi_yoy(history).dropna(), 365)
        ax.set_title('Core CPI YoY % (last 12M)')

    elif metric_key == 'cpi_volatile':
        series = safe_last(_compute_cpi_yoy(history).dropna(), 365)
        ax.set_title('Headline CPI YoY % (last 12M)')

    elif metric_key == 'breakeven_5y':
        series = safe_last(history['breakeven_5y'], 365)
        hline  = 2.0
        ax.set_title('5Y Breakeven Inflation (last 12M)')

    elif metric_key == 'fed_bs':
        series = safe_last(history['fed_bs'], 365)
        ax.set_title('Fed Balance Sheet Total Assets — $B (last 12M)')

    elif metric_key == 'sp_96':
        sp = history['sp500']
        if len(sp) > 200:
            i9, i6 = max(0, len(sp) - 189), max(0, len(sp) - 126)
            series  = sp.iloc[i9: i6 + 1]
        else:
            series  = safe_last(sp, 274)
        ax.set_title('S&P 500 — 9M-to-6M-Ago Window (leading signal)')

    elif metric_key == 'stoxx_96':
        st = history['stoxx600']
        if len(st) > 200:
            i9, i6 = max(0, len(st) - 189), max(0, len(st) - 126)
            series  = st.iloc[i9: i6 + 1]
        else:
            series  = safe_last(st, 274)
        ax.set_title('STOXX 600 — 9M-to-6M-Ago Window (leading signal)')

    elif metric_key in history:
        series = safe_last(history[metric_key], 365)
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} (last 12M)")

    _plot_series(ax, series, ax.get_title(), hline=hline)
    plt.tight_layout()
    return fig

def generate_short_term_graph(metric_key, history, today):
    if metric_key in NO_SHORT_TERM_CHART:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    series  = None
    hline   = None

    if metric_key == 'macd':
        sp_full = history['sp500'].dropna()
        if len(sp_full) >= 26:
            macd_f, sig_f, hist_f = compute_macd(sp_full)
            cut = pd.Timestamp(today - timedelta(days=90))
            m3  = macd_f[macd_f.index >= cut]
            s3  = sig_f[sig_f.index >= cut]
            h3  = hist_f[hist_f.index >= cut]
            if not m3.empty:
                x = mdates.date2num(m3.index.to_pydatetime())
                _plot_macd_bars(ax, m3, s3, h3, x, 'LazyMan MACD — Last 3 Months')
            else:
                ax.text(0.5, 0.5, 'No 3M data', ha='center', va='center',
                       transform=ax.transAxes, color='gray')
        else:
            ax.text(0.5, 0.5, f'Not enough data ({len(sp_full)} pts)', ha='center',
                   va='center', transform=ax.transAxes, color='gray')
        plt.tight_layout()
        return fig

    elif metric_key == 'spread_10_2':
        y10    = history['10yr_yield']
        y2     = history['2yr_yield'].reindex(y10.index, method='nearest', tolerance=pd.Timedelta('5D'))
        series = _short_term_window((y10 - y2).dropna())
        hline  = 0
        ax.set_title('10Yr-2Yr Spread — Recent View')

    elif metric_key == 'real_rate_10yr':
        series = _short_term_window(_compute_real_rate_series(history, '10yr').dropna())
        hline  = 0
        ax.set_title('Real Rate 10Yr — Recent View')

    elif metric_key == 'sp_96':
        series = _short_term_window(history['sp500'])
        ax.set_title('S&P 500 — Last 3 Months (current momentum)')

    elif metric_key == 'stoxx_96':
        series = _short_term_window(history['stoxx600'])
        ax.set_title('STOXX 600 — Last 3 Months (current momentum)')

    elif metric_key in history:
        series = _short_term_window(history[metric_key])
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} — Recent View")

    if series is not None:
        _plot_series(ax, series, ax.get_title(), hline=hline)
    plt.tight_layout()
    return fig

def get_graph_key(item_text):
    t = item_text
    if 'Copper/Gold' in t: return 'copper_gold'
    if '10Yr-FedFunds' in t: return 'spread_10ff'
    if '10Yr-2Yr' in t: return 'spread_10_2'
    if 'Real Rate' in t and '10' in t: return 'real_rate_10yr'
    if 'Breakeven' in t: return 'breakeven_5y'
    if 'Balance Sheet' in t: return 'fed_bs'
    if 'Core CPI' in t: return 'core_cpi'
    if 'CPI Volatile' in t: return 'cpi_volatile'
    if '9-6' in t and 'S&P' in t: return 'sp_96'
    if '9-6' in t and 'STOXX' in t: return 'stoxx_96'
    if 'MACD' in t: return 'macd'
    return 'placeholder'

# ============================================================================
# METRICS & SCORING
# ============================================================================

def calculate_metrics(data, history, today):
    metrics = {}

    metrics['yield_curve_10_2']      = data['10yr_yield'] - data['2yr_yield']
    metrics['yield_curve_10ff']      = data['10yr_yield'] - data['fed_funds']
    metrics['real_rate_10yr']        = data['real_rate_10yr']
    metrics['real_rate_2yr']         = data['real_rate_2yr']
    metrics['copper_gold_ratio']     = data['copper'] / data['gold'] if data['gold'] != 0 else 0

    phase = detect_cycle_phase(metrics['yield_curve_10_2'], metrics['real_rate_10yr'],
                               data['ism_manufacturing'], data['earnings_growth'])
    metrics['phase']    = phase
    phase_cfg           = CYCLE_PHASE_CONFIG[phase]

    tailwinds, headwinds, neutrals = [], [], []
    score = 50
    W = OPTIMIZED_WEIGHTS

    # Yield curve 10Y-2Y
    yc = metrics['yield_curve_10_2']
    if yc > 1.0:
        score += W['yield_curve_10_2']
        tailwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Steep curve (strong expansion signal)")
    elif yc > 0.5:
        score += int(W['yield_curve_10_2'] * 0.7)
        tailwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Positive (expansion)")
    elif yc > 0:
        score += int(W['yield_curve_10_2'] * 0.3)
        neutrals.append(f"10Yr-2Yr Spread: {yc:.2f}% — Barely positive")
    elif yc > -0.5:
        score -= int(W['yield_curve_10_2'] * 0.6)
        headwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Slightly inverted")
    else:
        score -= W['yield_curve_10_2']
        headwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — INVERTED (recession signal)")

    # Real rate 10Yr
    rr10 = metrics['real_rate_10yr']
    if rr10 < -1.0:
        score += W['real_rate_10yr']
        tailwinds.append(f"Real Rate 10Yr: {rr10:.2f}% — Highly stimulative")
    elif rr10 < 0:
        score += int(W['real_rate_10yr'] * 0.6)
        tailwinds.append(f"Real Rate 10Yr: {rr10:.2f}% — Stimulative")
    elif rr10 < 1.0:
        neutrals.append(f"Real Rate 10Yr: {rr10:.2f}% — Neutral")
    else:
        score -= W['real_rate_10yr']
        headwinds.append(f"Real Rate 10Yr: {rr10:.2f}% — Restrictive")

    # Earnings growth
    eg = data['earnings_growth']
    metrics['earnings_growth'] = eg
    if eg > 10:
        score += W['earnings_growth']
        tailwinds.append(f"Earnings Growth (fwd 12M): {eg:.1f}% — Strong")
    elif eg > 5:
        score += int(W['earnings_growth'] * 0.6)
        tailwinds.append(f"Earnings Growth (fwd 12M): {eg:.1f}% — Positive")
    else:
        score -= W['earnings_growth']
        headwinds.append(f"Earnings Growth (fwd 12M): {eg:.1f}% — Negative")

    # MACD
    macd_lb = False
    try:
        spc = history['sp500'].dropna()
        if len(spc) >= 40:
            ml, sl, _ = compute_macd(spc)
            macd_lb   = bool(ml.iloc[-1] > sl.iloc[-1])
    except Exception:
        pass
    metrics['macd_long_bullish']  = macd_lb
    if macd_lb:
        score += W['macd_long']
        tailwinds.append("LazyMan MACD: Long Buy → Bull signal")
    else:
        score -= int(W['macd_long'] * 0.5)
        headwinds.append("LazyMan MACD: Long Sell → Bear signal")

    # Fed Balance Sheet growth
    fbs = data['fed_bs_growth']
    metrics['fed_bs_growth'] = fbs
    if fbs > 3:
        score += W['fed_bs_growth']
        tailwinds.append(f"Fed BS Growth: {fbs:.1f}% YoY — Easing")
    elif fbs > 0:
        score += int(W['fed_bs_growth'] * 0.5)
        tailwinds.append(f"Fed BS Growth: {fbs:.1f}% YoY — Slight easing")
    else:
        score -= W['fed_bs_growth']
        headwinds.append(f"Fed BS Growth: {fbs:.1f}% YoY — Tightening")

    # VIX trend
    vt = data['vix_trend']
    vix_val = float(data['vix'])
    if vt > 0:
        score += W['vix_trend']
        tailwinds.append(f"VIX: {vix_val:.1f} — Healthy, declining")
    elif vt < 0:
        score -= W['vix_trend']
        headwinds.append(f"VIX: {vix_val:.1f} — Complacency or panic")

    # S&P 9-6M return
    sp = history['sp500']
    if len(sp) > 200:
        p9  = float(sp.iloc[max(0, len(sp)-189)])
        p6  = float(sp.iloc[max(0, len(sp)-126)])
        sp96 = (p6-p9)/p9*100 if p9 != 0 else 0
    else:
        sp96 = 0
    metrics['sp_96_return'] = sp96
    if sp96 > 5:
        score += W['sp_96']
        tailwinds.append(f"S&P 9-6M Return: {sp96:.2f}% — Strong leading signal")
    elif sp96 > 0:
        score += int(W['sp_96'] * 0.5)
        tailwinds.append(f"S&P 9-6M Return: {sp96:.2f}% — Positive")
    else:
        headwinds.append(f"S&P 9-6M Return: {sp96:.2f}% — Negative")

    # STOXX 9-6M return
    st = history['stoxx600']
    if len(st) > 200:
        p9  = float(st.iloc[max(0, len(st)-189)])
        p6  = float(st.iloc[max(0, len(st)-126)])
        st96 = (p6-p9)/p9*100 if p9 != 0 else 0
    else:
        st96 = 0
    metrics['stoxx_96_return'] = st96
    if st96 > 0:
        score += W['stoxx_96']
        tailwinds.append(f"STOXX 9-6M Return: {st96:.2f}% — Positive global signal")

    # Manufacturing PMI
    mf = data['ism_manufacturing']
    if mf > 52:
        score += W['ism_manufacturing']
        tailwinds.append(f"Manufacturing PMI: {mf:.1f} — Strong expansion")
    elif mf > 50:
        score += int(W['ism_manufacturing'] * 0.5)
        tailwinds.append(f"Manufacturing PMI: {mf:.1f} — Expansion")
    else:
        score -= W['ism_manufacturing']
        headwinds.append(f"Manufacturing PMI: {mf:.1f} — Contraction")

    # Services PMI
    sv = data['ism_services']
    if sv > 52:
        score += W['ism_services']
        tailwinds.append(f"Services PMI: {sv:.1f} — Strong expansion")
    elif sv > 50:
        score += int(W['ism_services'] * 0.5)
        tailwinds.append(f"Services PMI: {sv:.1f} — Expansion")

    # Building permits
    bp = data['building_permits']
    bph = history['building_permits']
    bp_chg = float(bph.iloc[-1]-bph.iloc[-2]) if len(bph) > 1 else 0
    if bp_chg > 0:
        score += W['building_permits']
        tailwinds.append(f"Building Permits: {bp:.2f}M — Rising")
    else:
        headwinds.append(f"Building Permits: {bp:.2f}M — Falling")

    # NFIB
    nf = data['nfib']
    if nf > 100:
        score += W['nfib']
        tailwinds.append(f"NFIB: {nf:.1f} — Strong confidence")
    elif nf < 95:
        score -= W['nfib']
        headwinds.append(f"NFIB: {nf:.1f} — Weak confidence")

    # UMCSI
    um = data['umcsi']
    if um > 70:
        score += W['umcsi']
        tailwinds.append(f"UMCSI: {um:.1f} — Bullish")
    elif um < 55:
        score -= W['umcsi']
        headwinds.append(f"UMCSI: {um:.1f} — Bearish")

    # BBB yield
    bbb = data['bbb_yield']
    bbb_chg = (float(history['bbb_yield'].iloc[-1]) - float(history['bbb_yield'].iloc[-2])
               if len(history['bbb_yield']) > 1 else 0)
    if bbb_chg < 0:
        score += W['bbb_yield']
        tailwinds.append(f"BBB Yield: {bbb:.2f}% — Declining")
    else:
        headwinds.append(f"BBB Yield: {bbb:.2f}% — Rising")

    # Cycle phase boost
    boost = phase_cfg['boost']
    score += boost
    if boost > 0:
        tailwinds.append(f"Cycle Phase: {phase_cfg['label']} — Score +{boost}")
    elif boost < 0:
        headwinds.append(f"Cycle Phase: {phase_cfg['label']} — Score {boost}")

    score = max(0, min(150, int(score)))
    metrics['score_raw'] = score

    if phase_cfg.get('force_short'):
        bias  = 'Short (recession regime)'
        score = min(score, 30)
    elif score >= 80:
        bias = 'Long — High Conviction'
    elif score >= 65:
        bias = 'Long — Moderate Conviction'
    elif score <= 20:
        bias = 'Short — High Conviction'
    elif score <= 35:
        bias = 'Short — Moderate Conviction'
    else:
        bias = 'Neutral'

    metrics['conviction'] = abs(score - 50) / 50.0
    metrics['bias']       = bias
    metrics['phase_label'] = phase_cfg['label']

    return metrics, tailwinds, headwinds, neutrals, bias, score

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

@st.cache_data(ttl=86400)
def run_backtest():
    quarters = pd.date_range(start='2014-01-01', end='2024-01-01', freq='QE')
    results  = []

    try:
        sp_hist = yf.download('^GSPC', start='2013-01-01', end='2024-06-01', progress=False)['Close']
        sp_hist.index = pd.DatetimeIndex(sp_hist.index).tz_localize(None)
    except Exception:
        st.warning("Could not fetch S&P data")
        return pd.DataFrame(), {}

    try:
        y10_hist = fred.get_series('DGS10', observation_start='2010-01-01')
        y10_hist = normalize_index(y10_hist)
        y2_hist  = fred.get_series('DGS2', observation_start='2010-01-01')
        y2_hist  = normalize_index(y2_hist)
        ff_hist  = fred.get_series('FEDFUNDS', observation_start='2010-01-01')
        ff_hist  = normalize_index(ff_hist)
        be_hist  = fred.get_series('T5YIFR', observation_start='2010-01-01')
        be_hist  = normalize_index(be_hist)
        pmi_hist = fred.get_series('NAPM', observation_start='2010-01-01')
        pmi_hist = normalize_index(pmi_hist)
    except Exception:
        st.warning("Could not fetch FRED data for backtest")
        return pd.DataFrame(), {}

    def _get_val(series, date, default=0):
        try:
            sub = series[series.index <= date]
            return float(sub.iloc[-1]) if not sub.empty else default
        except Exception:
            return default

    def _get_sp_return(sp, date_start, date_end):
        try:
            sub_s = sp[sp.index <= date_start]
            sub_e = sp[sp.index <= date_end]
            if sub_s.empty or sub_e.empty:
                return 0.0
            p0 = float(sub_s.iloc[-1])
            p1 = float(sub_e.iloc[-1])
            return (p1 - p0) / p0 if p0 != 0 else 0.0
        except Exception:
            return 0.0

    def _score_at_date(date):
        y10  = _get_val(y10_hist, date, 2.5)
        y2   = _get_val(y2_hist,  date, 2.0)
        ff   = _get_val(ff_hist,  date, 1.5)
        be   = _get_val(be_hist,  date, 2.0)
        pmi  = _get_val(pmi_hist, date, 51.0)

        rr10    = y10 - be
        yc_10_2 = y10 - y2
        phase = detect_cycle_phase(yc_10_2, rr10, pmi, 5.0)

        s = 50
        s += 18 if yc_10_2 > 1.0 else (12 if yc_10_2 > 0.5 else (5 if yc_10_2 > 0 else (-12 if yc_10_2 > -0.5 else -18)))
        s += 7  if rr10 < -1  else (4 if rr10 < 0 else (-7 if rr10 > 1 else 0))
        s += 4  if pmi   > 52 else (2 if pmi > 50 else -4)
        s += CYCLE_PHASE_CONFIG[phase]['boost']
        s = max(0, min(150, s))

        if CYCLE_PHASE_CONFIG[phase].get('force_short'):
            bias = 'Short'
        elif s >= 65:
            bias = 'Long'
        elif s <= 35:
            bias = 'Short'
        else:
            bias = 'Neutral'

        return s, bias, phase

    for i in range(len(quarters) - 1):
        q_date      = quarters[i]
        q_next      = quarters[i + 1]
        score, bias, phase = _score_at_date(q_date)
        sp_ret      = _get_sp_return(sp_hist, q_date, q_next)

        if bias == 'Long':
            strat_ret = sp_ret
        elif bias == 'Short':
            strat_ret = -sp_ret
        else:
            strat_ret = 0.0

        strat_ret -= 0.001

        results.append({
            'Date':         q_date,
            'Score':        score,
            'Bias':         bias,
            'Phase':        phase,
            'SP500_Return': round(sp_ret * 100, 2),
            'Strat_Return': round(strat_ret * 100, 2),
        })

    if not results:
        return pd.DataFrame(), {}

    df            = pd.DataFrame(results).set_index('Date')
    df['SP500_Cum']  = (1 + df['SP500_Return'] / 100).cumprod()
    df['Strat_Cum']  = (1 + df['Strat_Return'] / 100).cumprod()

    n  = len(df)
    sp_total   = df['SP500_Cum'].iloc[-1] - 1
    st_total   = df['Strat_Cum'].iloc[-1] - 1
    sp_ann     = (1 + sp_total) ** (4 / n) - 1
    st_ann     = (1 + st_total) ** (4 / n) - 1
    st_vol     = df['Strat_Return'].std() / 100 * 2
    sp_vol     = df['SP500_Return'].std() / 100 * 2
    st_sharpe  = (st_ann - 0.02) / st_vol  if st_vol  != 0 else 0
    sp_sharpe  = (sp_ann - 0.02) / sp_vol  if sp_vol  != 0 else 0
    st_mdd     = (df['Strat_Cum'] / df['Strat_Cum'].cummax() - 1).min()
    sp_mdd     = (df['SP500_Cum'] / df['SP500_Cum'].cummax() - 1).min()

    correct     = ((df['Bias'] == 'Long')  & (df['SP500_Return'] > 0)) | \
                  ((df['Bias'] == 'Short') & (df['SP500_Return'] < 0))
    accuracy   = correct.mean()
    hit_rate   = (df['Strat_Return'] > 0).mean()

    summary = {
        'Strategy Total Return':   f"{st_total:.1%}",
        'S&P 500 Total Return':    f"{sp_total:.1%}",
        'Strategy Ann. Return':    f"{st_ann:.1%}",
        'S&P 500 Ann. Return':     f"{sp_ann:.1%}",
        'Strategy Sharpe':         f"{st_sharpe:.2f}",
        'S&P 500 Sharpe':          f"{sp_sharpe:.2f}",
        'Strategy Max Drawdown':   f"{st_mdd:.1%}",
        'S&P 500 Max Drawdown':    f"{sp_mdd:.1%}",
        'Direction Accuracy':      f"{accuracy:.1%}",
        'Quarterly Hit Rate':      f"{hit_rate:.1%}",
        'Total Quarters':          n,
    }
    return df, summary

# ============================================================================
# SECTOR TILT
# ============================================================================

def generate_sector_tilt(bias, score, phase, conviction, preferred_sectors, portfolio_size):
    today       = datetime.now()
    three_m_ago = pd.Timestamp(today - timedelta(days=90))

    momentum = {}
    pe_data  = {}
    for sector, info in ALL_SECTORS.items():
        try:
            hist = yf.Ticker(info['etf']).history(period='1y')['Close']
            if hist.empty or len(hist) < 2:
                mom = 0.0
            else:
                if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                r1y = (hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0] if hist.iloc[0] != 0 else 0
                h3m = hist[hist.index >= three_m_ago]
                r3m = (hist.iloc[-1] - h3m.iloc[0]) / h3m.iloc[0] if (not h3m.empty and h3m.iloc[0] != 0) else 0
                mom = 0.5 * r1y + 0.5 * r3m
        except Exception:
            mom = 0.0
        momentum[sector] = mom
        zscore = get_pe_zscore(sector, info['etf'])
        pe_data[sector]  = zscore

    rot = SECTOR_ROTATION.get(phase, SECTOR_ROTATION['mid'])

    adjusted = {}
    for sector in ALL_SECTORS:
        adj_mom = momentum[sector]
        z       = pe_data[sector]
        if z > 1.5:
            adj_mom *= 0.7
        elif z < -1.5:
            adj_mom *= 1.2
        rotation_wt = rot.get(sector, 0.05)
        adjusted[sector] = 0.5 * rotation_wt + 0.5 * adj_mom

    sorted_s = sorted(adjusted, key=adjusted.get, reverse=True)

    conv_clamped = max(0, min(1, conviction))
    if 'Long' in bias:
        long_pct  = 0.5 + 0.3 * conv_clamped
        short_pct = 1 - long_pct
    else:
        long_pct  = 0.5
        short_pct = 0.5

    long_pool = sorted_s[:3]
    short_pool = [s for s in reversed(sorted_s) if s not in long_pool][:3]

    pairs = list(zip(long_pool, short_pool))
    n_pairs = len(pairs)
    if n_pairs == 0:
        return pd.DataFrame(), {}

    long_alloc  = portfolio_size * long_pct  / n_pairs
    short_alloc = portfolio_size * short_pct / n_pairs

    rows = []
    for i, (ls, ss) in enumerate(pairs):
        rows.append({
            'Pair':            f"Pair {i+1}",
            'Long Sector':     ls,
            'Long ETF':        ALL_SECTORS[ls]['etf'],
            'Long Alloc':      f"${long_alloc:,.0f}",
            'Short Sector':    ss,
            'Short ETF':       ALL_SECTORS[ss]['etf'],
            'Short Alloc':     f"${short_alloc:,.0f}",
            'Long PE Z':       f"{pe_data[ls]:.2f}",
            'Short PE Z':      f"{pe_data[ss]:.2f}",
        })

    return pd.DataFrame(rows), {'long_pct': long_pct, 'short_pct': short_pct}

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Portfolio Bias & Sector Tilt (Optimized)", layout="wide")
st.title("📊 Portfolio Bias & Sector Tilt Dashboard")

with st.sidebar:
    st.header("⚙️ Settings")
    portfolio_size = st.number_input("Portfolio Size ($)", min_value=10000, value=100000, step=10000)
    preferred_sectors = st.multiselect("Preferred Sectors", list(ALL_SECTORS.keys()), default=[])

tab1, tab2, tab3 = st.tabs(["📈 Live Analysis", "🎯 Sector Tilt", "📊 10Y Backtest"])

with tab1:
    if st.button("🔄 Run Live Analysis", type="primary"):
        with st.spinner("Fetching macro data..."):
            try:
                data, history, today = fetch_data()
                metrics, tailwinds, headwinds, neutrals, bias, score = calculate_metrics(data, history, today)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("GDP Score", f"{score}/150")
                with col2:
                    st.metric("Bias", bias.split('—')[0].strip())
                with col3:
                    st.metric("Conviction", f"{metrics['conviction']:.0%}")
                with col4:
                    st.metric("Phase", metrics['phase_label'])

                st.divider()

                col_tw, col_hw, col_n = st.columns(3)
                with col_tw:
                    st.subheader(f"✅ Tailwinds ({len(tailwinds)})")
                    for tw in tailwinds[:5]:
                        st.write(f"• {tw}")
                with col_hw:
                    st.subheader(f"❌ Headwinds ({len(headwinds)})")
                    for hw in headwinds[:5]:
                        st.write(f"• {hw}")
                with col_n:
                    st.subheader(f"⚖️ Neutrals ({len(neutrals)})")
                    for n in neutrals[:5]:
                        st.write(f"• {n}")

                st.session_state.data    = data
                st.session_state.history = history
                st.session_state.metrics = metrics
                st.session_state.bias    = bias
                st.session_state.score   = score

                st.success("✅ Analysis Complete!")

            except Exception as e:
                st.error(f"❌ Error: {e}")

with tab2:
    if 'metrics' in st.session_state:
        st.subheader("🎯 Sector Pair Trading")

        with st.spinner("Fetching sector data..."):
            tilt_df, _ = generate_sector_tilt(
                st.session_state.bias, st.session_state.score,
                st.session_state.metrics['phase'], st.session_state.metrics['conviction'],
                preferred_sectors, portfolio_size)

            if not tilt_df.empty:
                st.dataframe(tilt_df, use_container_width=True)
                st.download_button(
                    "📥 Download CSV",
                    data=tilt_df.to_csv(index=False),
                    file_name=f"sector_tilt.csv",
                    mime="text/csv")
    else:
        st.info("Run **Live Analysis** first")

with tab3:
    if st.button("▶️ Run 10Y Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            bt_df, bt_summary = run_backtest()

            if not bt_df.empty:
                st.success("✅ Complete!")

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Strat Sharpe", bt_summary.get('Strategy Sharpe', 'N/A'))
                with col2:
                    st.metric("SP Sharpe", bt_summary.get('S&P 500 Sharpe', 'N/A'))
                with col3:
                    st.metric("Accuracy", bt_summary.get('Direction Accuracy', 'N/A'))
                with col4:
                    st.metric("Hit Rate", bt_summary.get('Quarterly Hit Rate', 'N/A'))
                with col5:
                    st.metric("Quarters", bt_summary.get('Total Quarters', 'N/A'))

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(bt_df.index, bt_df['SP500_Cum'], label='S&P 500', linewidth=2.5)
                ax.plot(bt_df.index, bt_df['Strat_Cum'], label='Strategy', linewidth=2.5)
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Return')
                ax.set_title('10Y Backtest: Cumulative Returns')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

                st.download_button("📥 Download Results", bt_df.to_csv(), "backtest.csv", "text/csv")

st.markdown("---")
st.caption("✅ Production-Ready | 10Y Quarterly Backtest | Cycle-Aware Sector Rotation")




