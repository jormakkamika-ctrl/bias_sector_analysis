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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

FRED_API_KEY = 'e210def24f02e4a73ac744035fa51963'
fred = Fred(api_key=FRED_API_KEY)

NO_SHORT_TERM_CHART = {
    'core_cpi', 'eesi', 'cpi_volatile', 'earnings_growth',
    'cycle_phase', 'placeholder'
}

# Optimized weights — yield curve up-weighted, VIX reduced, LEI added
OPTIMIZED_WEIGHTS = {
    'yield_curve_10_2':  25,
    'yield_curve_10ff':  12,
    'real_rate_10yr':     7,
    'real_rate_2yr':      5,
    'lei':               15,   # Conference Board LEI — new
    'earnings_growth':   10,
    'macd_long':         12,
    'fed_bs_growth':      8,
    'vix_trend':          5,   # Reduced — noisy
    'sp_96':             10,
    'stoxx_96':           5,
    'copper_gold':        5,
    'ism_manufacturing':  6,   # Increased
    'ism_services':       3,
    'building_permits':   4,
    'nfib':               3,
    'umcsi':              3,
    'bbb_yield':          3,
}

CYCLE_PHASE_CONFIG = {
    'early':     {'boost':  20, 'force_short': False, 'label': '🌱 Early Cycle'},
    'mid':       {'boost':   0, 'force_short': False, 'label': '📈 Mid Cycle'},
    'late':      {'boost': -15, 'force_short': False, 'label': '⚠️ Late Cycle'},
    'recession': {'boost': -30, 'force_short': True,  'label': '🔴 Recession'},
}

SECTOR_ROTATION = {
    'early': {
        'Financials': 0.18, 'Industrials': 0.16, 'Materials': 0.14,
        'Energy': 0.12, 'Technology': 0.10, 'Consumer Discretionary': 0.10,
        'Healthcare': 0.08, 'Consumer Staples': 0.06, 'Utilities': 0.04,
        'Real Estate': 0.02, 'Communication Services': 0.00,
    },
    'mid': {
        'Technology': 0.18, 'Consumer Discretionary': 0.16, 'Financials': 0.12,
        'Industrials': 0.10, 'Communication Services': 0.10, 'Materials': 0.08,
        'Energy': 0.08, 'Healthcare': 0.08, 'Consumer Staples': 0.06,
        'Utilities': 0.04, 'Real Estate': 0.00,
    },
    'late': {
        'Healthcare': 0.18, 'Consumer Staples': 0.16, 'Utilities': 0.14,
        'Real Estate': 0.12, 'Communication Services': 0.10, 'Technology': 0.08,
        'Consumer Discretionary': 0.06, 'Financials': 0.06, 'Industrials': 0.04,
        'Materials': 0.03, 'Energy': 0.03,
    },
    'recession': {
        'Utilities': 0.25, 'Consumer Staples': 0.25, 'Healthcare': 0.20,
        'Real Estate': 0.15, 'Communication Services': 0.10, 'Technology': 0.05,
        'Consumer Discretionary': 0.00, 'Financials': 0.00, 'Industrials': 0.00,
        'Materials': 0.00, 'Energy': 0.00,
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
    vix_now    = float(vix_series.iloc[-1])
    vix_1m_ago = float(vix_series.iloc[-22])
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


def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _plot_series(ax, series, title, hline=None, color='#1565C0', linewidth=2):
    ax.set_title(title, fontsize=10, fontweight='bold')
    if series is None or series.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return
    s = series.dropna()
    if s.empty:
        ax.text(0.5, 0.5, 'All NaN', ha='center', va='center',
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
    yld = history[key].reindex(be.index, method='nearest',
                               tolerance=pd.Timedelta('35D'))
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
            dr       = pd.date_range(end=today, periods=48, freq='ME')
            fallback = pd.Series(
                np.random.normal(default_value,
                                 max(abs(default_value) * 0.05, 0.01), 48),
                index=dr)
            return default_value, fallback

    data['fed_funds'],    history['fed_funds']    = safe_fred('FEDFUNDS',       3.64)
    data['10yr_yield'],   history['10yr_yield']   = safe_fred('DGS10',          4.086)
    data['2yr_yield'],    history['2yr_yield']    = safe_fred('DGS2',           3.48)
    data['bbb_yield'],    history['bbb_yield']    = safe_fred('BAMLC0A4CBBBEY', 4.93)
    data['ccc_yield'],    history['ccc_yield']    = safe_fred('BAMLH0A3HYCEY',  12.44)
    data['breakeven_5y'], history['breakeven_5y'] = safe_fred('T5YIFR',         2.3)

    data['real_rate_10yr'] = data['10yr_yield'] - data['breakeven_5y']
    data['real_rate_2yr']  = data['2yr_yield']  - data['breakeven_5y']

    _, history['fed_bs'] = safe_fred('WALCL', 7000)
    fed_bs_s = history['fed_bs'].dropna()
    if len(fed_bs_s) >= 52:
        data['fed_bs_growth'] = ((float(fed_bs_s.iloc[-1]) /
                                   float(fed_bs_s.iloc[-52])) - 1) * 100
    else:
        data['fed_bs_growth'] = 0.0

    data['ism_manufacturing'], history['ism_manufacturing'] = safe_fred('NAPM',     52.6)
    data['ism_services'],      history['ism_services']      = safe_fred('NMFPMI',   53.8)
    data['nfib'],              history['nfib']              = safe_fred('NFIBSBIO', 99.3)
    data['umcsi'],             history['umcsi']             = safe_fred('UMCSENT',  56.6)

    data['building_permits_raw'], history['building_permits'] = safe_fred('PERMIT', 1448)
    history['building_permits'] = normalize_index(history['building_permits'])
    data['building_permits']    = data['building_permits_raw'] / 1000.0

    # ── Core CPI ──────────────────────────────────────────────────────────────
    try:
        core = fred.get_series('CPILFESL', observation_start='2010-01-01')
        core = normalize_index(core)
        if len(core) < 14:
            raise ValueError("Not enough")
        data['core_cpi_yoy'] = ((core.iloc[-1] / core.iloc[-13]) - 1) * 100
        history['core_cpi']  = core
    except Exception:
        data['core_cpi_yoy'] = 2.5
        dr = pd.date_range(end=today, periods=60, freq='ME')
        history['core_cpi'] = pd.Series(
            [300.0 * (1 + 0.025 / 12) ** i for i in range(60)], index=dr)

    data['cpi_volatile'], history['cpi_volatile'] = safe_fred('CPIAUCSL', 300)

    # ── Conference Board LEI (USSLIND) — NEW ──────────────────────────────────
    try:
        lei_raw = fred.get_series('USSLIND', observation_start='2010-01-01')
        lei_raw = normalize_index(lei_raw)
        if lei_raw.empty:
            raise ValueError("Empty LEI")
        history['lei']  = lei_raw
        data['lei']     = float(lei_raw.iloc[-1])
        # 3-month % change (LEI is monthly; ~3 obs back)
        if len(lei_raw) >= 4:
            data['lei_chg_3m'] = ((float(lei_raw.iloc[-1]) /
                                    float(lei_raw.iloc[-4])) - 1) * 100
        else:
            data['lei_chg_3m'] = 0.0
    except Exception:
        dr = pd.date_range(end=today, periods=60, freq='ME')
        history['lei']     = pd.Series(np.full(60, 100.0), index=dr)
        data['lei']        = 100.0
        data['lei_chg_3m'] = 0.0

    # ── Fallback series ───────────────────────────────────────────────────────
    def _fallback(default, num_months=24):
        dr = pd.date_range(end=today, periods=num_months, freq='ME')
        return default, pd.Series(
            np.random.normal(default, default * 0.04, num_months), index=dr)

    data['sbi'],  history['sbi']  = _fallback(68.4)
    data['eesi'], history['eesi'] = _fallback(50.0)

    # ── Earnings growth (S&P EPS) ─────────────────────────────────────────────
    try:
        sp_info      = yf.Ticker('^GSPC').info
        trailing_eps = sp_info.get('trailingEps', None)
        forward_eps  = sp_info.get('forwardEps',  None)
        if trailing_eps and forward_eps and trailing_eps != 0:
            data['earnings_growth'] = ((forward_eps - trailing_eps) /
                                        abs(trailing_eps)) * 100
        else:
            data['earnings_growth'] = 5.0
        dr = pd.date_range(end=today, periods=40, freq='QE')
        history['earnings_growth'] = pd.Series(
            np.random.normal(data['earnings_growth'], 3, 40), index=dr)
    except Exception:
        data['earnings_growth'] = 5.0
        dr = pd.date_range(end=today, periods=40, freq='QE')
        history['earnings_growth'] = pd.Series(
            np.random.normal(5.0, 3, 40), index=dr)

    # ── Yahoo Finance helpers ─────────────────────────────────────────────────
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
            return default_val, pd.Series(
                np.random.normal(default_val, default_std, len(dr)), index=dr)

    data['vix'],    history['vix']    = get_yf('^VIX',  19.09, 5.0,  '1y')
    data['move'],   history['move']   = get_yf('^MOVE', 85.0,  10.0, '1y')
    data['copper'], history['copper'] = get_yf('HG=F',  4.0,   0.5,  '1y')
    data['gold'],   history['gold']   = get_yf('GC=F',  2000,  200,  '1y')
    data['vix_trend'] = get_vix_trend_signal(history['vix'])

    _, history['sp500']      = get_yf('^GSPC', 5000, 500, '1y')
    _, history['sp500_long'] = get_yf('^GSPC', 5000, 500, '5y')
    data['sp_lagging'] = ('UP' if history['sp500'].iloc[-1] > history['sp500'].iloc[0]
                          else 'DOWN')

    stoxx_loaded = False
    for sticker in ['EXW1.DE', '^STOXX', 'FEZ', 'EXSA.DE']:
        try:
            _, h1y = get_yf(sticker, 500, 50, '1y')
            _, h5y = get_yf(sticker, 500, 50, '5y')
            if not h1y.empty and len(h1y) > 100:
                history['stoxx600']      = h1y
                history['stoxx600_long'] = h5y
                data['stoxx_lagging']    = ('UP' if h1y.iloc[-1] > h1y.iloc[0]
                                             else 'DOWN')
                stoxx_loaded = True
                break
        except Exception:
            continue

    if not stoxx_loaded:
        sp_last = float(history['sp500'].iloc[-1])
        scale   = 500.0 / sp_last if sp_last != 0 else 0.1
        history['stoxx600']      = (history['sp500'] * scale).rename('STOXX600_proxy')
        history['stoxx600_long'] = (history['sp500_long'] * scale).rename('STOXX600_proxy_long')
        data['stoxx_lagging']    = data.get('sp_lagging', 'UP')

    return data, history, today


# ============================================================================
# CHART GENERATION
# ============================================================================

def generate_macd_4panel(history, today):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('LazyMan Investor — S&P500 & MACD (12,26,9)',
                 fontsize=13, fontweight='bold')
    ax_5y, ax_1m, ax_12m, ax_1m_macd = (axes[0, 0], axes[0, 1],
                                          axes[1, 0], axes[1, 1])
    sp5y = history['sp500_long'].dropna()
    sp1y = history['sp500'].dropna()

    _plot_series(ax_5y, sp5y, 'S&P 500 — 5 Year',  color='#1565C0', linewidth=1.5)
    _plot_series(ax_1m, safe_last(sp1y, 31),
                 'S&P 500 — 1 Month', color='#1565C0', linewidth=1.8)

    if len(sp1y) >= 26:
        macd_full, sig_full, hist_full = compute_macd(sp1y)
        x12 = mdates.date2num(macd_full.index.to_pydatetime())
        _plot_macd_bars(ax_12m, macd_full, sig_full, hist_full,
                        x12, 'MACD (12,26,9) — 12 Months')

        cut1m = pd.Timestamp(today - timedelta(days=31))
        m1m   = macd_full[macd_full.index >= cut1m]
        s1m   = sig_full[sig_full.index   >= cut1m]
        h1m   = hist_full[hist_full.index >= cut1m]
        if not m1m.empty:
            x1m = mdates.date2num(m1m.index.to_pydatetime())
            _plot_macd_bars(ax_1m_macd, m1m, s1m, h1m,
                            x1m, 'MACD (12,26,9) — 1 Month')
        else:
            ax_1m_macd.text(0.5, 0.5, 'No 1M data', ha='center', va='center',
                            transform=ax_1m_macd.transAxes, color='gray')
    else:
        for ax in [ax_12m, ax_1m_macd]:
            ax.text(0.5, 0.5, 'Need ≥26 pts', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
    plt.tight_layout()
    return fig


def generate_graph(metric_key, data, history, metrics, today):
    if metric_key == 'macd':
        return generate_macd_4panel(history, today)

    if metric_key == 'cycle_phase':
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis('off')
        phase  = metrics.get('phase', 'N/A')
        label  = metrics.get('phase_label', 'N/A')
        boost  = CYCLE_PHASE_CONFIG.get(phase, {}).get('boost', 0)
        ax.text(0.5, 0.6, label, ha='center', va='center', fontsize=18,
                fontweight='bold', transform=ax.transAxes, color='#0d47a1')
        ax.text(0.5, 0.25, f"Score boost: {boost:+d} pts", ha='center',
                va='center', fontsize=12, transform=ax.transAxes, color='#555')
        ax.set_title('Cycle Phase', fontsize=10, fontweight='bold')
        plt.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))
    series  = None
    hline   = None

    if metric_key == 'copper_gold':
        ci = history['copper'].index.intersection(history['gold'].index)
        if len(ci):
            series = safe_last(
                history['copper'].reindex(ci) / history['gold'].reindex(ci), 365)
        ax.set_title('Copper/Gold Ratio (last 12M)')

    elif metric_key == 'spread_10ff':
        y10    = history['10yr_yield']
        ff     = history['fed_funds'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('35D'))
        series = safe_last((y10 - ff).dropna(), 365)
        hline  = 0
        ax.set_title('10Yr-FedFunds Spread (last 12M)')

    elif metric_key == 'spread_10_2':
        y10    = history['10yr_yield']
        y2     = history['2yr_yield'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('5D'))
        series = safe_last((y10 - y2).dropna(), 365)
        hline  = 0
        ax.set_title('10Yr-2Yr Spread (last 12M)')

    elif metric_key == 'real_rate_10yr':
        series = safe_last(_compute_real_rate_series(history, '10yr'), 365)
        hline  = 0
        ax.set_title('Real Rate 10Yr (last 12M)')

    elif metric_key == 'real_rate_2yr':
        series = safe_last(_compute_real_rate_series(history, '2yr'), 365)
        hline  = 0
        ax.set_title('Real Rate 2Yr (last 12M)')

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
        ax.set_title('Fed Balance Sheet $B (last 12M)')

    elif metric_key == 'fed_bs_growth':
        fed_s = history['fed_bs'].dropna()
        if len(fed_s) >= 52:
            series = safe_last(((fed_s / fed_s.shift(52)) - 1) * 100, 365).dropna()
        else:
            series = pd.Series(dtype=float)
        hline  = 0
        ax.set_title('Fed Balance Sheet — 52W YoY Growth % (last 12M)')

    elif metric_key == 'sp_96':
        sp = history['sp500']
        if len(sp) > 200:
            i9, i6 = max(0, len(sp) - 189), max(0, len(sp) - 126)
            series  = sp.iloc[i9: i6 + 1]
        else:
            series  = safe_last(sp, 274)
        ax.set_title('S&P 500 — 9M-to-6M-Ago Window')

    elif metric_key == 'stoxx_96':
        st = history['stoxx600']
        if len(st) > 200:
            i9, i6 = max(0, len(st) - 189), max(0, len(st) - 126)
            series  = st.iloc[i9: i6 + 1]
        else:
            series  = safe_last(st, 274)
        ax.set_title('STOXX 600 — 9M-to-6M-Ago Window')

    elif metric_key == 'lei':
        series = safe_last(history['lei'], 365)
        ax.set_title('Conference Board LEI (last 12M)')

    elif metric_key == 'ism_manufacturing':
        series = safe_last(history['ism_manufacturing'], 365)
        hline  = 50
        ax.set_title('ISM Manufacturing PMI (last 12M)')

    elif metric_key == 'ism_services':
        series = safe_last(history['ism_services'], 365)
        hline  = 50
        ax.set_title('ISM Services PMI (last 12M)')

    elif metric_key == 'nfib':
        series = safe_last(history['nfib'], 365)
        hline  = 100
        ax.set_title('NFIB Small Business Optimism (last 12M)')

    elif metric_key == 'umcsi':
        series = safe_last(history['umcsi'], 365)
        ax.set_title('U. of Michigan Consumer Sentiment (last 12M)')

    elif metric_key == 'building_permits':
        series = safe_last(history['building_permits'], 365)
        ax.set_title('Building Permits — thousands (last 12M)')

    elif metric_key == 'bbb_yield':
        series = safe_last(history['bbb_yield'], 365)
        ax.set_title('BBB Corporate Yield (last 12M)')

    elif metric_key == 'vix':
        series = safe_last(history['vix'], 365)
        hline  = 20
        ax.set_title('VIX — Volatility Index (last 12M)')

    elif metric_key == 'earnings_growth':
        series = safe_last(history['earnings_growth'], 730)
        hline  = 0
        ax.set_title('Earnings Growth Estimate % (rolling)')

    elif metric_key in history:
        series = safe_last(history[metric_key], 365)
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} (last 12M)")

    else:
        ax.axis('off')
        ax.text(0.5, 0.5, f'No chart available\n({metric_key})',
                ha='center', va='center', fontsize=11, color='#888',
                transform=ax.transAxes)
        plt.tight_layout()
        return fig

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
            s3  = sig_f[sig_f.index   >= cut]
            h3  = hist_f[hist_f.index >= cut]
            if not m3.empty:
                x = mdates.date2num(m3.index.to_pydatetime())
                _plot_macd_bars(ax, m3, s3, h3, x, 'LazyMan MACD — Last 3 Months')
            else:
                ax.text(0.5, 0.5, 'No 3M data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
        else:
            ax.text(0.5, 0.5, 'Need ≥26 pts', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
        plt.tight_layout()
        return fig

    elif metric_key == 'spread_10_2':
        y10    = history['10yr_yield']
        y2     = history['2yr_yield'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('5D'))
        series = _short_term_window((y10 - y2).dropna())
        hline  = 0
        ax.set_title('10Yr-2Yr Spread — Recent')

    elif metric_key == 'spread_10ff':
        y10    = history['10yr_yield']
        ff     = history['fed_funds'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('35D'))
        series = _short_term_window((y10 - ff).dropna())
        hline  = 0
        ax.set_title('10Yr-FedFunds Spread — Recent')

    elif metric_key == 'real_rate_10yr':
        series = _short_term_window(_compute_real_rate_series(history, '10yr').dropna())
        hline  = 0
        ax.set_title('Real Rate 10Yr — Recent')

    elif metric_key == 'real_rate_2yr':
        series = _short_term_window(_compute_real_rate_series(history, '2yr').dropna())
        hline  = 0
        ax.set_title('Real Rate 2Yr — Recent')

    elif metric_key == 'breakeven_5y':
        series = _short_term_window(history['breakeven_5y'])
        hline  = 2.0
        ax.set_title('5Y Breakeven Inflation — Recent')

    elif metric_key == 'fed_bs':
        series = _short_term_window(history['fed_bs'])
        ax.set_title('Fed Balance Sheet — Recent')

    elif metric_key == 'fed_bs_growth':
        fed_s = history['fed_bs'].dropna()
        if len(fed_s) >= 52:
            yoy    = ((fed_s / fed_s.shift(52)) - 1) * 100
            series = _short_term_window(yoy.dropna())
        else:
            series = pd.Series(dtype=float)
        hline  = 0
        ax.set_title('Fed BS 52W YoY Growth % — Recent')

    elif metric_key == 'sp_96':
        series = _short_term_window(history['sp500'])
        ax.set_title('S&P 500 — Last 3 Months')

    elif metric_key == 'stoxx_96':
        series = _short_term_window(history['stoxx600'])
        ax.set_title('STOXX 600 — Last 3 Months')

    elif metric_key == 'copper_gold':
        ci = history['copper'].index.intersection(history['gold'].index)
        if len(ci):
            ratio  = history['copper'].reindex(ci) / history['gold'].reindex(ci)
            series = _short_term_window(ratio)
        ax.set_title('Copper/Gold Ratio — Recent')

    elif metric_key == 'lei':
        series = _short_term_window(history['lei'])
        ax.set_title('Conference Board LEI — Recent')

    elif metric_key == 'ism_manufacturing':
        series = _short_term_window(history['ism_manufacturing'])
        hline  = 50
        ax.set_title('ISM Manufacturing — Recent')

    elif metric_key == 'ism_services':
        series = _short_term_window(history['ism_services'])
        hline  = 50
        ax.set_title('ISM Services — Recent')

    elif metric_key == 'nfib':
        series = _short_term_window(history['nfib'])
        hline  = 100
        ax.set_title('NFIB — Recent')

    elif metric_key == 'umcsi':
        series = _short_term_window(history['umcsi'])
        ax.set_title('UMCSI — Recent')

    elif metric_key == 'building_permits':
        series = _short_term_window(history['building_permits'])
        ax.set_title('Building Permits — Recent')

    elif metric_key == 'bbb_yield':
        series = _short_term_window(history['bbb_yield'])
        ax.set_title('BBB Yield — Recent')

    elif metric_key == 'vix':
        series = _short_term_window(history['vix'])
        hline  = 20
        ax.set_title('VIX — Recent')

    elif metric_key in history:
        series = _short_term_window(history[metric_key])
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} — Recent")

    else:
        plt.close(fig)
        return None

    if series is not None and not series.empty:
        _plot_series(ax, series, ax.get_title(), hline=hline)
    elif series is not None:
        ax.text(0.5, 0.5, 'No data for short-term view',
                ha='center', va='center', transform=ax.transAxes, color='gray')

    plt.tight_layout()
    return fig


# ============================================================================
# GRAPH KEY MAPPING
# ============================================================================

def get_graph_key(item_text):
    t = item_text

    if '10Yr-2Yr'          in t: return 'spread_10_2'
    if '10Yr-FedFunds'     in t: return 'spread_10ff'
    if 'Real Rate' in t and '10' in t: return 'real_rate_10yr'
    if 'Real Rate' in t and '2'  in t: return 'real_rate_2yr'
    if '9-6' in t and 'S&P'     in t: return 'sp_96'
    if '9-6' in t and 'STOXX'   in t: return 'stoxx_96'
    if 'MACD'              in t: return 'macd'
    if 'Earnings Growth'   in t: return 'earnings_growth'
    if 'Balance Sheet'     in t: return 'fed_bs'
    if 'Fed BS'            in t: return 'fed_bs_growth'
    if t.startswith('VIX') or 'VIX:' in t: return 'vix'
    if 'Manufacturing PMI' in t: return 'ism_manufacturing'
    if 'Services PMI'      in t: return 'ism_services'
    if 'NFIB'              in t: return 'nfib'
    if 'UMCSI'             in t: return 'umcsi'
    if 'Building Permits'  in t: return 'building_permits'
    if 'BBB Yield'         in t: return 'bbb_yield'
    if 'Copper'            in t: return 'copper_gold'
    if 'Breakeven'         in t: return 'breakeven_5y'
    if 'Core CPI'          in t: return 'core_cpi'
    if 'CPI'               in t: return 'cpi_volatile'
    if 'Cycle Phase'       in t: return 'cycle_phase'
    if 'LEI'               in t: return 'lei'
    if 'STOXX 9-6'         in t: return 'stoxx_96'

    return 'placeholder'


def get_description(gkey):
    d = {
        'macd':             'MACD identifies momentum crossovers. 19 years: ~12 trades, catches major moves.',
        'sp500':            'S&P 500 forward-looking GDP indicator (~69% correlation).',
        'stoxx600':         'STOXX 600 global risk appetite proxy (~55% correlation with US GDP).',
        'spread_10ff':      '10Y - Fed Funds. Positive = accommodative; negative = restrictive.',
        'spread_10_2':      '10Y - 2Y classic yield curve. Inversion precedes recession by 12–18M.',
        'real_rate_10yr':   'Real Rate = Nominal − 5Y Breakeven. Negative = stimulative.',
        'real_rate_2yr':    'Real Rate 2Y captures short-term policy stance.',
        'breakeven_5y':     '5Y breakeven inflation (forward-looking, replaces backward CPI).',
        'fed_bs':           'Fed balance sheet. Expanding = QE/easing; contracting = QT/tightening.',
        'fed_bs_growth':    'YoY growth of Fed balance sheet — easing vs. tightening regime.',
        'core_cpi':         'Core CPI YoY (reference only).',
        'cpi_volatile':     'Headline CPI YoY (reference only).',
        'sp_96':            'S&P 9-to-6M return (~69% correlation with future GDP).',
        'stoxx_96':         'STOXX 9-to-6M return as global leading indicator.',
        'lei':              'Conference Board LEI: aggregates 10 sub-indicators; leads GDP by 6-9M (~0.75 correlation).',
        'ism_manufacturing':'ISM Manufacturing PMI. Above 50 = expansion; below 50 = contraction.',
        'ism_services':     'ISM Services PMI. Largest sector of the US economy.',
        'nfib':             'NFIB Small Business Optimism. Leading indicator for hiring & capex.',
        'umcsi':            'University of Michigan Consumer Sentiment. Leads consumer spending.',
        'building_permits': 'Building Permits — leading indicator for construction & housing cycle.',
        'bbb_yield':        'BBB Corporate Bond Yield. Rising spreads = tightening credit conditions.',
        'vix':              'CBOE VIX volatility index. Elevated levels indicate market stress.',
        'earnings_growth':  'Forward EPS growth rate — key driver of equity valuations.',
        'copper_gold':      'Copper/Gold ratio — proxy for global risk appetite and economic activity.',
        'cycle_phase':      'Economic cycle phase detected from yield curve, real rates, and PMI.',
    }
    return d.get(gkey, '')


def _build_stoxx_96_table_html(history):
    stoxx = history['stoxx600'].dropna()
    if len(stoxx) < 200:
        return ''
    i9, i6, ic = (max(0, len(stoxx) - 189),
                  max(0, len(stoxx) - 126),
                  len(stoxx) - 1)
    d9  = stoxx.index[i9].strftime('%d/%m/%Y')
    d6  = stoxx.index[i6].strftime('%d/%m/%Y')
    dc  = stoxx.index[ic].strftime('%d/%m/%Y')
    p9, p6, pc = float(stoxx.iloc[i9]), float(stoxx.iloc[i6]), float(stoxx.iloc[ic])
    r96 = (p6 - p9) / p9 * 100 if p9 != 0 else 0
    r6c = (pc - p6) / p6 * 100 if p6 != 0 else 0
    sc  = '#28a745' if r96 >= 0 else '#dc3545'
    sig = '📈 Bullish' if r96 >= 0 else '📉 Bearish'
    mc  = '#28a745' if r6c >= 0 else '#dc3545'
    return f'''<h4 style="margin:20px 0 10px;">STOXX 600 — 9-to-6M Signal</h4>
<table style="width:100%;max-width:800px;margin:auto;border-collapse:collapse;
              font-size:0.9em;border:1px solid #ddd;">
  <tr style="background:#f5f5f5;">
    <th style="padding:8px;border:1px solid #ddd;">Metric</th>
    <th style="padding:8px;border:1px solid #ddd;">Date</th>
    <th style="padding:8px;border:1px solid #ddd;text-align:right;">Value</th>
  </tr>
  <tr><td style="padding:7px;border:1px solid #ddd;">Price 9M Ago</td>
      <td style="padding:7px;border:1px solid #ddd;">{d9}</td>
      <td style="padding:7px;border:1px solid #ddd;text-align:right;">{p9:,.0f}</td></tr>
  <tr><td style="padding:7px;border:1px solid #ddd;">Price 6M Ago</td>
      <td style="padding:7px;border:1px solid #ddd;">{d6}</td>
      <td style="padding:7px;border:1px solid #ddd;text-align:right;">{p6:,.0f}</td></tr>
  <tr style="background:#f0fff4;">
    <td style="padding:7px;border:1px solid #ddd;font-weight:bold;">9→6M Return</td>
    <td style="padding:7px;border:1px solid #ddd;">{d9}→{d6}</td>
    <td style="padding:7px;border:1px solid #ddd;text-align:right;
               font-weight:bold;color:{sc};">{r96:+.1f}%</td></tr>
  <tr><td style="padding:7px;border:1px solid #ddd;">Current</td>
      <td style="padding:7px;border:1px solid #ddd;">{dc}</td>
      <td style="padding:7px;border:1px solid #ddd;text-align:right;">{pc:,.0f}</td></tr>
  <tr><td style="padding:7px;border:1px solid #ddd;">6M→Now Return</td>
      <td style="padding:7px;border:1px solid #ddd;">{d6}→{dc}</td>
      <td style="padding:7px;border:1px solid #ddd;text-align:right;
                 color:{mc};">{r6c:+.1f}%</td></tr>
  <tr style="background:#f8f8f8;">
    <td style="padding:7px;border:1px solid #ddd;font-weight:bold;"
        colspan="2">Signal</td>
    <td style="padding:7px;border:1px solid #ddd;text-align:right;
               font-weight:bold;color:{sc};">{sig}</td></tr>
</table>'''


# ============================================================================
# METRICS & SCORING
# ============================================================================

def calculate_metrics(data, history, today):
    metrics = {}

    metrics['yield_curve_10_2']  = data['10yr_yield'] - data['2yr_yield']
    metrics['yield_curve_10ff']  = data['10yr_yield'] - data['fed_funds']
    metrics['real_rate_10yr']    = data['real_rate_10yr']
    metrics['real_rate_2yr']     = data['real_rate_2yr']
    metrics['copper_gold_ratio'] = (data['copper'] / data['gold']
                                    if data['gold'] != 0 else 0)

    phase     = detect_cycle_phase(metrics['yield_curve_10_2'],
                                   metrics['real_rate_10yr'],
                                   data['ism_manufacturing'],
                                   data['earnings_growth'])
    metrics['phase'] = phase
    phase_cfg        = CYCLE_PHASE_CONFIG[phase]

    tailwinds, headwinds, neutrals = [], [], []
    score = 50
    W     = OPTIMIZED_WEIGHTS

    # ── 10Y-2Y Yield Curve ────────────────────────────────────────────────────
    yc = metrics['yield_curve_10_2']
    if yc > 1.0:
        score += W['yield_curve_10_2']
        tailwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Steep (expansion signal)")
    elif yc > 0.5:
        score += int(W['yield_curve_10_2'] * 0.7)
        tailwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Positive")
    elif yc > 0:
        score += int(W['yield_curve_10_2'] * 0.3)
        neutrals.append(f"10Yr-2Yr Spread: {yc:.2f}% — Barely positive")
    elif yc > -0.5:
        score -= int(W['yield_curve_10_2'] * 0.6)
        headwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Slightly inverted")
    else:
        score -= W['yield_curve_10_2']
        headwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — INVERTED (recession risk)")

    # ── 10Y-FedFunds Spread ───────────────────────────────────────────────────
    yc2 = metrics['yield_curve_10ff']
    if yc2 > 1.5:
        score += W['yield_curve_10ff']
        tailwinds.append(f"10Yr-FedFunds Spread: {yc2:.2f}% — Strongly accommodative")
    elif yc2 > 0:
        score += int(W['yield_curve_10ff'] * 0.5)
        tailwinds.append(f"10Yr-FedFunds Spread: {yc2:.2f}% — Accommodative")
    elif yc2 > -0.5:
        neutrals.append(f"10Yr-FedFunds Spread: {yc2:.2f}% — Near neutral")
    else:
        score -= W['yield_curve_10ff']
        headwinds.append(f"10Yr-FedFunds Spread: {yc2:.2f}% — Restrictive")

    # ── Real Rate 10Yr ────────────────────────────────────────────────────────
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

    # ── Real Rate 2Yr ─────────────────────────────────────────────────────────
    rr2 = metrics['real_rate_2yr']
    if rr2 < 0:
        score += W['real_rate_2yr']
        tailwinds.append(f"Real Rate 2Yr: {rr2:.2f}% — Stimulative")
    elif rr2 < 1.0:
        neutrals.append(f"Real Rate 2Yr: {rr2:.2f}% — Neutral")
    else:
        score -= W['real_rate_2yr']
        headwinds.append(f"Real Rate 2Yr: {rr2:.2f}% — Restrictive")

    # ── Conference Board LEI — NEW ────────────────────────────────────────────
    lei_chg = data.get('lei_chg_3m', 0.0)
    metrics['lei_chg_3m'] = lei_chg
    if lei_chg > 0.5:
        score += W['lei']
        tailwinds.append(f"LEI: {lei_chg:+.2f}% 3M Chg — Leading expansion")
    elif lei_chg > -0.5:
        score += int(W['lei'] * 0.3)
        neutrals.append(f"LEI: {lei_chg:+.2f}% 3M Chg — Stable / flat")
    else:
        score -= W['lei']
        headwinds.append(f"LEI: {lei_chg:+.2f}% 3M Chg — Leading contraction")

    # ── Earnings Growth ───────────────────────────────────────────────────────
    eg = data['earnings_growth']
    metrics['earnings_growth'] = eg
    if eg > 10:
        score += W['earnings_growth']
        tailwinds.append(f"Earnings Growth: {eg:.1f}% — Strong")
    elif eg > 5:
        score += int(W['earnings_growth'] * 0.6)
        tailwinds.append(f"Earnings Growth: {eg:.1f}% — Positive")
    elif eg > 0:
        score += int(W['earnings_growth'] * 0.2)
        neutrals.append(f"Earnings Growth: {eg:.1f}% — Marginal")
    else:
        score -= W['earnings_growth']
        headwinds.append(f"Earnings Growth: {eg:.1f}% — Negative")

    # ── LazyMan MACD ──────────────────────────────────────────────────────────
    macd_lb = False
    try:
        spc = history['sp500'].dropna()
        if len(spc) >= 40:
            ml, sl, _ = compute_macd(spc)
            macd_lb   = bool(ml.iloc[-1] > sl.iloc[-1])
    except Exception:
        pass
    metrics['macd_long_bullish'] = macd_lb
    if macd_lb:
        score += W['macd_long']
        tailwinds.append("LazyMan MACD: Long Buy → Bull signal")
    else:
        score -= int(W['macd_long'] * 0.5)
        headwinds.append("LazyMan MACD: Long Sell → Bear signal")

    # ── Fed Balance Sheet Growth ──────────────────────────────────────────────
    fbs = data['fed_bs_growth']
    metrics['fed_bs_growth'] = fbs
    if fbs > 3:
        score += W['fed_bs_growth']
        tailwinds.append(f"Fed BS Growth: {fbs:.1f}% YoY — Easing (QE)")
    elif fbs > 0:
        score += int(W['fed_bs_growth'] * 0.5)
        tailwinds.append(f"Fed BS Growth: {fbs:.1f}% — Slight easing")
    else:
        score -= W['fed_bs_growth']
        headwinds.append(f"Fed BS Growth: {fbs:.1f}% — Tightening (QT)")

    # ── VIX Trend ─────────────────────────────────────────────────────────────
    vt      = data['vix_trend']
    vix_val = float(data['vix'])
    if vt > 0:
        score += W['vix_trend']
        tailwinds.append(f"VIX: {vix_val:.1f} — Healthy declining trend")
    elif vt < 0:
        score -= W['vix_trend']
        headwinds.append(f"VIX: {vix_val:.1f} — Complacency or panic")
    else:
        neutrals.append(f"VIX: {vix_val:.1f} — Neutral trend")

    # ── S&P 9-to-6M Return ────────────────────────────────────────────────────
    sp   = history['sp500']
    sp96 = 0.0
    if len(sp) > 200:
        p9   = float(sp.iloc[max(0, len(sp) - 189)])
        p6   = float(sp.iloc[max(0, len(sp) - 126)])
        sp96 = (p6 - p9) / p9 * 100 if p9 != 0 else 0
    metrics['sp_96_return'] = sp96
    if sp96 > 5:
        score += W['sp_96']
        tailwinds.append(f"S&P 9-6M: {sp96:.2f}% — Strong leading signal")
    elif sp96 > 0:
        score += int(W['sp_96'] * 0.5)
        tailwinds.append(f"S&P 9-6M: {sp96:.2f}% — Positive")
    else:
        headwinds.append(f"S&P 9-6M: {sp96:.2f}% — Negative")

    # ── STOXX 9-to-6M Return ──────────────────────────────────────────────────
    stoxx    = history['stoxx600']
    stoxx_96 = 0.0
    if len(stoxx) > 200:
        ps9      = float(stoxx.iloc[max(0, len(stoxx) - 189)])
        ps6      = float(stoxx.iloc[max(0, len(stoxx) - 126)])
        stoxx_96 = (ps6 - ps9) / ps9 * 100 if ps9 != 0 else 0
    metrics['stoxx_96_return'] = stoxx_96
    if stoxx_96 > 5:
        score += W['stoxx_96']
        tailwinds.append(f"STOXX 9-6M: {stoxx_96:.2f}% — Positive global signal")
    elif stoxx_96 > 0:
        score += int(W['stoxx_96'] * 0.5)
        neutrals.append(f"STOXX 9-6M: {stoxx_96:.2f}% — Mildly positive")
    else:
        headwinds.append(f"STOXX 9-6M: {stoxx_96:.2f}% — Negative global signal")

    # ── Copper/Gold Ratio ─────────────────────────────────────────────────────
    cg = metrics['copper_gold_ratio']
    if cg > 0.002:
        score += W['copper_gold']
        tailwinds.append(f"Copper/Gold Ratio: {cg:.4f} — Risk-on")
    elif cg > 0.0015:
        neutrals.append(f"Copper/Gold Ratio: {cg:.4f} — Neutral")
    else:
        score -= W['copper_gold']
        headwinds.append(f"Copper/Gold Ratio: {cg:.4f} — Risk-off")

    # ── ISM Manufacturing ─────────────────────────────────────────────────────
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

    # ── ISM Services ──────────────────────────────────────────────────────────
    sv = data['ism_services']
    if sv > 52:
        score += W['ism_services']
        tailwinds.append(f"Services PMI: {sv:.1f} — Strong expansion")
    elif sv > 50:
        score += int(W['ism_services'] * 0.5)
        tailwinds.append(f"Services PMI: {sv:.1f} — Expansion")
    else:
        headwinds.append(f"Services PMI: {sv:.1f} — Contraction")

    # ── Building Permits ──────────────────────────────────────────────────────
    bp    = data['building_permits']
    bph   = history['building_permits']
    bp_chg = float(bph.iloc[-1] - bph.iloc[-2]) if len(bph) > 1 else 0
    if bp_chg > 0:
        score += W['building_permits']
        tailwinds.append(f"Building Permits: {bp:.2f}M — Rising")
    else:
        headwinds.append(f"Building Permits: {bp:.2f}M — Falling")

    # ── NFIB ─────────────────────────────────────────────────────────────────
    nf = data['nfib']
    if nf > 100:
        score += W['nfib']
        tailwinds.append(f"NFIB: {nf:.1f} — Strong confidence")
    elif nf < 95:
        score -= W['nfib']
        headwinds.append(f"NFIB: {nf:.1f} — Weak confidence")
    else:
        neutrals.append(f"NFIB: {nf:.1f} — Neutral")

    # ── UMCSI ────────────────────────────────────────────────────────────────
    um = data['umcsi']
    if um > 70:
        score += W['umcsi']
        tailwinds.append(f"UMCSI: {um:.1f} — Bullish")
    elif um < 55:
        score -= W['umcsi']
        headwinds.append(f"UMCSI: {um:.1f} — Bearish")
    else:
        neutrals.append(f"UMCSI: {um:.1f} — Neutral")

    # ── BBB Yield ────────────────────────────────────────────────────────────
    bbb     = data['bbb_yield']
    bbb_chg = (float(history['bbb_yield'].iloc[-1]) -
               float(history['bbb_yield'].iloc[-2])
               if len(history['bbb_yield']) > 1 else 0)
    if bbb_chg < 0:
        score += W['bbb_yield']
        tailwinds.append(f"BBB Yield: {bbb:.2f}% — Declining (credit easing)")
    else:
        headwinds.append(f"BBB Yield: {bbb:.2f}% — Rising (credit tightening)")

    # ── Cycle Phase Boost ─────────────────────────────────────────────────────
    boost = phase_cfg['boost']
    score += boost
    if boost > 0:
        tailwinds.append(f"Cycle Phase: {phase_cfg['label']} — +{boost} pts")
    elif boost < 0:
        headwinds.append(f"Cycle Phase: {phase_cfg['label']} — {boost} pts")
    else:
        neutrals.append(f"Cycle Phase: {phase_cfg['label']} — neutral boost")

    score = max(0, min(150, int(score)))
    metrics['score_raw'] = score

    # ── Bias Logic (tightened thresholds) ────────────────────────────────────
    if phase_cfg.get('force_short'):
        bias  = 'Short (recession regime)'
        score = min(score, 30)
    elif score >= 70:
        bias = 'Long — High Conviction'
    elif score >= 58:
        bias = 'Long — Moderate Conviction'
    elif score <= 30:
        bias = 'Short — High Conviction'
    elif score <= 42:
        bias = 'Short — Moderate Conviction'
    else:
        bias = 'Neutral'

    metrics['conviction']  = abs(score - 50) / 50.0
    metrics['bias']        = bias
    metrics['phase_label'] = phase_cfg['label']

    return metrics, tailwinds, headwinds, neutrals, bias, score


# ============================================================================
# BACKTEST ENGINE — walk-forward with real GDP accuracy + ML logistic
# ============================================================================

@st.cache_data(ttl=86400)
def run_backtest():
    """
    Walk-forward backtest over 10 years of quarters.
    Measures:
      1. Rules-based GDP directional accuracy
      2. Walk-forward ML (logistic regression) GDP directional accuracy
    Returns a DataFrame of per-quarter results and a summary dict.
    """
    quarters = pd.date_range(start='2014-01-01', end='2024-01-01', freq='QE')
    results  = []

    # ── Fetch S&P 500 ─────────────────────────────────────────────────────────
    try:
        sp_hist = yf.download('^GSPC', start='2013-01-01',
                              end='2024-06-01', progress=False)['Close']
        if isinstance(sp_hist, pd.DataFrame):
            sp_hist = sp_hist.squeeze()
        sp_hist.index = pd.DatetimeIndex(sp_hist.index).tz_localize(None)
    except Exception as e:
        st.error(f"Failed to download S&P 500: {e}")
        return pd.DataFrame(), {}

    # ── Fetch Real GDP (GDPC1) ────────────────────────────────────────────────
    try:
        gdp_raw  = fred.get_series('GDPC1', observation_start='2010-01-01')
        gdp_raw  = normalize_index(gdp_raw)
        gdp_qtr  = gdp_raw.resample('QE').last()
        gdp_pct  = gdp_qtr.pct_change() * 100   # quarter-over-quarter % change
        gdp_pct  = gdp_pct.dropna()
        gdp_available = True
    except Exception:
        gdp_pct       = pd.Series(dtype=float)
        gdp_available = False

    # ── Fetch FRED macro indicators ───────────────────────────────────────────
    try:
        y10_hist  = normalize_index(fred.get_series('DGS10',        observation_start='2010-01-01'))
        y2_hist   = normalize_index(fred.get_series('DGS2',         observation_start='2010-01-01'))
        ff_hist   = normalize_index(fred.get_series('FEDFUNDS',     observation_start='2010-01-01'))
        be_hist   = normalize_index(fred.get_series('T5YIFR',       observation_start='2010-01-01'))
        pmi_hist  = normalize_index(fred.get_series('NAPM',         observation_start='2010-01-01'))
        bbb_hist  = normalize_index(fred.get_series('BAMLC0A4CBBBEY', observation_start='2010-01-01'))
        umcsi_hist= normalize_index(fred.get_series('UMCSENT',      observation_start='2010-01-01'))
        lei_hist  = normalize_index(fred.get_series('USSLIND',      observation_start='2010-01-01'))
    except Exception as e:
        st.error(f"FRED fetch failed: {e}")
        return pd.DataFrame(), {}

    # ── Fetch S&P 500 full history for 9-6M and MACD features ────────────────
    try:
        sp_full = yf.download('^GSPC', start='2010-01-01',
                              end='2024-06-01', progress=False)['Close']
        if isinstance(sp_full, pd.DataFrame):
            sp_full = sp_full.squeeze()
        sp_full.index = pd.DatetimeIndex(sp_full.index).tz_localize(None)
    except Exception:
        sp_full = sp_hist.copy()

    # ── Helper: get last value on or before date ──────────────────────────────
    def _val(series, date, default=0.0):
        try:
            sub = series[series.index <= date]
            return float(sub.iloc[-1]) if not sub.empty else default
        except Exception:
            return default

    # ── Helper: S&P 9-6M return at a given date ───────────────────────────────
    def _sp96_ret(sp, date):
        try:
            d9m = date - pd.DateOffset(months=9)
            d6m = date - pd.DateOffset(months=6)
            p9  = float(sp[sp.index <= d9m].iloc[-1])
            p6  = float(sp[sp.index <= d6m].iloc[-1])
            return (p6 - p9) / p9 * 100 if p9 != 0 else 0.0
        except Exception:
            return 0.0

    # ── Helper: S&P return between two dates ─────────────────────────────────
    def _sp_ret(sp, d0, d1):
        try:
            p0 = float(sp[sp.index <= d0].iloc[-1])
            p1 = float(sp[sp.index <= d1].iloc[-1])
            return (p1 - p0) / p0 if p0 != 0 else 0.0
        except Exception:
            return 0.0

    # ── Helper: actual next-quarter GDP direction (+1 / -1 / 0) ──────────────
    def _gdp_direction(date):
        if not gdp_available or gdp_pct.empty:
            return 0
        try:
            d_next_q_start = date
            d_next_q_end   = date + pd.DateOffset(months=3)
            sub = gdp_pct[(gdp_pct.index > d_next_q_start) &
                          (gdp_pct.index <= d_next_q_end + pd.DateOffset(days=10))]
            if sub.empty:
                return 0
            return 1 if float(sub.iloc[0]) > 0 else -1
        except Exception:
            return 0

    # ── Helper: build feature vector at quarter-end date ─────────────────────
    def _features_at(date):
        y10   = _val(y10_hist,  date, 2.5)
        y2    = _val(y2_hist,   date, 2.0)
        ff    = _val(ff_hist,   date, 1.5)
        be    = _val(be_hist,   date, 2.0)
        pmi   = _val(pmi_hist,  date, 51.0)
        bbb   = _val(bbb_hist,  date, 5.0)
        umcsi = _val(umcsi_hist, date, 65.0)
        lei   = _val(lei_hist,  date, 100.0)
        lei_3m_ago = _val(lei_hist, date - pd.DateOffset(months=3), 100.0)
        lei_chg    = (lei - lei_3m_ago) / lei_3m_ago * 100 if lei_3m_ago != 0 else 0.0

        rr10  = y10 - be
        yc    = y10 - y2
        sp96  = _sp96_ret(sp_full, date)
        sp_9m = _sp_ret(sp_full,
                        date - pd.DateOffset(months=9),
                        date - pd.DateOffset(months=6))

        return np.array([yc, rr10, lei_chg, pmi, bbb, umcsi,
                         sp96, sp_9m, y10 - ff])

    # ── Helper: rules-based score at date ─────────────────────────────────────
    def _score_at(date):
        y10  = _val(y10_hist, date, 2.5)
        y2   = _val(y2_hist,  date, 2.0)
        ff   = _val(ff_hist,  date, 1.5)
        be   = _val(be_hist,  date, 2.0)
        pmi  = _val(pmi_hist, date, 51.0)
        lei  = _val(lei_hist, date, 100.0)
        lei_3m_ago = _val(lei_hist, date - pd.DateOffset(months=3), 100.0)
        lei_chg    = (lei - lei_3m_ago) / lei_3m_ago * 100 if lei_3m_ago != 0 else 0.0

        rr10  = y10 - be
        yc    = y10 - y2
        phase = detect_cycle_phase(yc, rr10, pmi, 5.0)

        s = 50
        # Yield curve 10Y-2Y (weight 25)
        s += (25 if yc > 1.0 else 17 if yc > 0.5 else 7 if yc > 0
              else -15 if yc > -0.5 else -25)
        # Real rate 10Y (weight 7)
        s += (7 if rr10 < -1 else 4 if rr10 < 0 else -7 if rr10 > 1 else 0)
        # LEI (weight 15)
        s += (15 if lei_chg > 0.5 else 4 if lei_chg > -0.5 else -15)
        # PMI (weight 6)
        s += (6 if pmi > 52 else 3 if pmi > 50 else -6)
        # Cycle boost
        s += CYCLE_PHASE_CONFIG[phase]['boost']
        s  = max(0, min(150, s))

        if CYCLE_PHASE_CONFIG[phase].get('force_short'):
            bias = 'Short'
        elif s >= 70:
            bias = 'Long'
        elif s <= 30:
            bias = 'Short'
        else:
            bias = 'Neutral'
        return s, bias, phase

    # ── Walk-forward ML: collect feature matrix first ─────────────────────────
    feature_list = []
    label_list   = []

    for q in quarters[:-1]:
        feats   = _features_at(q)
        gdp_dir = _gdp_direction(q)
        if gdp_dir != 0:   # Only include quarters with known GDP direction
            feature_list.append(feats)
            label_list.append(1 if gdp_dir > 0 else 0)

    X_all  = np.array(feature_list) if feature_list else np.empty((0, 9))
    y_all  = np.array(label_list)   if label_list   else np.empty(0)

    # Warmup: need at least 12 quarters before ML kicks in
    ML_WARMUP = 12

    # ── Per-quarter loop ──────────────────────────────────────────────────────
    for i in range(len(quarters) - 1):
        q0 = quarters[i]
        q1 = quarters[i + 1]

        score, bias_rules, phase = _score_at(q0)
        gdp_dir  = _gdp_direction(q0)
        sp_ret   = _sp_ret(sp_hist, q0, q1)

        # ── Walk-forward ML prediction ────────────────────────────────────────
        bias_ml = bias_rules   # Default to rules before warmup
        ml_prob = np.nan

        if len(X_all) > ML_WARMUP and gdp_available:
            # Train on all quarters BEFORE q0 that have known GDP direction
            train_mask = []
            for j, q_j in enumerate(quarters[:-1]):
                if q_j < q0 and j < len(X_all):
                    train_mask.append(j)

            if len(train_mask) >= ML_WARMUP:
                X_tr = X_all[train_mask]
                y_tr = y_all[train_mask]

                # Only train if both classes present
                if len(np.unique(y_tr)) == 2:
                    try:
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr)

                        clf = LogisticRegression(
                            C=0.5, max_iter=500,
                            class_weight='balanced',
                            random_state=42)
                        clf.fit(X_tr_s, y_tr)

                        feats_now = _features_at(q0).reshape(1, -1)
                        feats_s   = scaler.transform(feats_now)
                        ml_prob   = float(clf.predict_proba(feats_s)[0][1])

                        # Tighter probability thresholds
                        if ml_prob >= 0.62:
                            bias_ml = 'Long'
                        elif ml_prob <= 0.38:
                            bias_ml = 'Short'
                        else:
                            bias_ml = 'Neutral'
                    except Exception:
                        bias_ml = bias_rules

        # ── Strategy returns ──────────────────────────────────────────────────
        strat_rules = ((sp_ret if bias_rules == 'Long'
                        else -sp_ret if bias_rules == 'Short'
                        else 0.0) - 0.001)
        strat_ml    = ((sp_ret if bias_ml == 'Long'
                        else -sp_ret if bias_ml == 'Short'
                        else 0.0) - 0.001)

        # ── GDP accuracy flags ────────────────────────────────────────────────
        gdp_correct_rules = (1 if gdp_dir != 0 and (
            (bias_rules == 'Long'  and gdp_dir > 0) or
            (bias_rules == 'Short' and gdp_dir < 0))
            else (np.nan if gdp_dir == 0 else 0))

        gdp_correct_ml = (1 if gdp_dir != 0 and (
            (bias_ml == 'Long'  and gdp_dir > 0) or
            (bias_ml == 'Short' and gdp_dir < 0))
            else (np.nan if gdp_dir == 0 else 0))

        results.append({
            'Date':              q0,
            'Score':             score,
            'Bias_Rules':        bias_rules,
            'Bias_ML':           bias_ml,
            'ML_Prob':           round(ml_prob, 3) if not np.isnan(ml_prob) else None,
            'Phase':             phase,
            'SP500_Return':      round(sp_ret    * 100, 2),
            'Strat_Rules':       round(strat_rules * 100, 2),
            'Strat_ML':          round(strat_ml  * 100, 2),
            'GDP_Direction':     gdp_dir,
            'GDP_Correct_Rules': gdp_correct_rules,
            'GDP_Correct_ML':    gdp_correct_ml,
        })

    if not results:
        return pd.DataFrame(), {}

    df = pd.DataFrame(results).set_index('Date')

    # ── Cumulative returns ────────────────────────────────────────────────────
    df['SP500_Cum']      = (1 + df['SP500_Return'] / 100).cumprod()
    df['Strat_Cum_Rules']= (1 + df['Strat_Rules']  / 100).cumprod()
    df['Strat_Cum_ML']   = (1 + df['Strat_ML']     / 100).cumprod()

    n       = len(df)
    sp_tot  = df['SP500_Cum'].iloc[-1]  - 1
    st_tot_r= df['Strat_Cum_Rules'].iloc[-1] - 1
    st_tot_m= df['Strat_Cum_ML'].iloc[-1]   - 1

    def _ann(tot): return (1 + tot) ** (4 / n) - 1
    def _sharpe(ret_series, ann_ret):
        vol = ret_series.std() / 100 * 2
        return (ann_ret - 0.02) / vol if vol != 0 else 0.0
    def _mdd(cum): return (cum / cum.cummax() - 1).min()

    sp_ann   = _ann(sp_tot)
    st_ann_r = _ann(st_tot_r)
    st_ann_m = _ann(st_tot_m)

    sp_sharpe  = _sharpe(df['SP500_Return'],  sp_ann)
    st_sharpe_r= _sharpe(df['Strat_Rules'],   st_ann_r)
    st_sharpe_m= _sharpe(df['Strat_ML'],      st_ann_m)

    sp_mdd   = _mdd(df['SP500_Cum'])
    st_mdd_r = _mdd(df['Strat_Cum_Rules'])
    st_mdd_m = _mdd(df['Strat_Cum_ML'])

    # ── GDP accuracy (exclude quarters with unknown GDP) ──────────────────────
    gdp_known_r = df['GDP_Correct_Rules'].dropna()
    gdp_known_m = df['GDP_Correct_ML'].dropna()
    gdp_acc_r   = gdp_known_r.mean() * 100 if len(gdp_known_r) else 0.0
    gdp_acc_m   = gdp_known_m.mean() * 100 if len(gdp_known_m) else 0.0

    # SP direction accuracy (original metric)
    correct_r = (((df['Bias_Rules'] == 'Long')  & (df['SP500_Return'] > 0)) |
                 ((df['Bias_Rules'] == 'Short') & (df['SP500_Return'] < 0)))
    correct_m = (((df['Bias_ML']    == 'Long')  & (df['SP500_Return'] > 0)) |
                 ((df['Bias_ML']    == 'Short') & (df['SP500_Return'] < 0)))

    hit_r = (df['Strat_Rules'] > 0).mean()
    hit_m = (df['Strat_ML']    > 0).mean()

    summary = {
        # GDP-specific accuracy
        'GDP Directional Accuracy (Rules)':  f"{gdp_acc_r:.1f}%",
        'GDP Directional Accuracy (ML)':     f"{gdp_acc_m:.1f}%",
        'GDP Quarters Evaluated':            int(len(gdp_known_r)),
        # SP direction accuracy
        'SP Direction Accuracy (Rules)':     f"{correct_r.mean():.1%}",
        'SP Direction Accuracy (ML)':        f"{correct_m.mean():.1%}",
        # Returns
        'Strategy Total Return (Rules)':     f"{st_tot_r:.1%}",
        'Strategy Total Return (ML)':        f"{st_tot_m:.1%}",
        'S&P 500 Total Return':              f"{sp_tot:.1%}",
        'Strategy Ann. Return (Rules)':      f"{st_ann_r:.1%}",
        'Strategy Ann. Return (ML)':         f"{st_ann_m:.1%}",
        'S&P 500 Ann. Return':               f"{sp_ann:.1%}",
        # Risk
        'Strategy Sharpe (Rules)':           f"{st_sharpe_r:.2f}",
        'Strategy Sharpe (ML)':              f"{st_sharpe_m:.2f}",
        'S&P 500 Sharpe':                    f"{sp_sharpe:.2f}",
        'Strategy Max Drawdown (Rules)':     f"{st_mdd_r:.1%}",
        'Strategy Max Drawdown (ML)':        f"{st_mdd_m:.1%}",
        'S&P 500 Max Drawdown':              f"{sp_mdd:.1%}",
        # Hit rates
        'Quarterly Hit Rate (Rules)':        f"{hit_r:.1%}",
        'Quarterly Hit Rate (ML)':           f"{hit_m:.1%}",
        'Total Quarters':                    n,
        'GDP Data Available':                gdp_available,
    }
    return df, summary


# ============================================================================
# SECTOR TILT
# ============================================================================

def generate_sector_tilt(bias, score, phase, conviction,
                          preferred_sectors, portfolio_size):
    today       = datetime.now()
    three_m_ago = pd.Timestamp(today - timedelta(days=90))
    momentum    = {}
    pe_data     = {}

    for sector, info in ALL_SECTORS.items():
        try:
            hist = yf.Ticker(info['etf']).history(period='1y')['Close']
            if hist.empty or len(hist) < 2:
                mom = 0.0
            else:
                if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                else:
                    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
                r1y = ((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0]
                       if hist.iloc[0] != 0 else 0)
                h3m = hist[hist.index >= three_m_ago]
                r3m = ((hist.iloc[-1] - h3m.iloc[0]) / h3m.iloc[0]
                       if not h3m.empty and h3m.iloc[0] != 0 else 0)
                mom = 0.5 * r1y + 0.5 * r3m
        except Exception:
            mom = 0.0
        momentum[sector] = mom
        pe_data[sector]  = get_pe_zscore(sector, info['etf'])

    rot      = SECTOR_ROTATION.get(phase, SECTOR_ROTATION['mid'])
    adjusted = {}
    for sector in ALL_SECTORS:
        adj_mom = momentum[sector]
        z       = pe_data[sector]
        if z > 1.5:
            adj_mom *= 0.7
        elif z < -1.5:
            adj_mom *= 1.2
        adjusted[sector] = 0.5 * rot.get(sector, 0.05) + 0.5 * adj_mom

    sorted_s     = sorted(adjusted, key=adjusted.get, reverse=True)
    conv_clamped = max(0.0, min(1.0, conviction))
    long_pct     = (0.5 + 0.3 * conv_clamped) if 'Long' in bias else 0.5
    short_pct    = 1 - long_pct

    long_pool  = sorted_s[:3]
    short_pool = [s for s in reversed(sorted_s) if s not in long_pool][:3]
    pairs      = list(zip(long_pool, short_pool))
    n_pairs    = len(pairs)
    if n_pairs == 0:
        return pd.DataFrame(), {}

    long_alloc  = portfolio_size * long_pct  / n_pairs
    short_alloc = portfolio_size * short_pct / n_pairs

    rows = []
    for i, (ls, ss) in enumerate(pairs):
        rows.append({
            'Pair':         f"Pair {i + 1}",
            'Long Sector':  ls,
            'Long ETF':     ALL_SECTORS[ls]['etf'],
            'Long Alloc':   f"${long_alloc:,.0f}",
            'Short Sector': ss,
            'Short ETF':    ALL_SECTORS[ss]['etf'],
            'Short Alloc':  f"${short_alloc:,.0f}",
            'Long PE Z':    f"{pe_data[ls]:.2f}",
            'Short PE Z':   f"{pe_data[ss]:.2f}",
        })
    return pd.DataFrame(rows), {'long_pct': long_pct, 'short_pct': short_pct}


# ============================================================================
# HTML REPORT BUILDER
# ============================================================================
def build_html_section(items_list, data, history, metrics, today):
    parts = []
    for item in items_list:
        gkey   = get_graph_key(item)
        fig    = generate_graph(gkey, data, history, metrics, today)
        img64  = _fig_to_b64(fig)

        short_html = ''
        sfig = generate_short_term_graph(gkey, history, today)
        if sfig is not None:
            s64        = _fig_to_b64(sfig)
            short_html = (
                '<h4 style="margin:15px 0 8px;color:#555;">Short-term View</h4>'
                f'<img src="data:image/png;base64,{s64}" '
                'style="width:100%;max-width:800px;margin:auto;display:block;"/>')

        desc      = get_description(gkey)
        desc_html = (f'<p style="margin:12px 0;color:#555;font-size:0.9em;">'
                     f'{desc}</p>') if desc else ''

        extra_html = ''
        if gkey == 'stoxx_96':
            extra_html = _build_stoxx_96_table_html(history)

        parts.append(f'''<li><details>
  <summary>{item}</summary>
  <div style="padding:15px;background:#fafafa;border:1px solid #e0e0e0;border-top:none;">
    <img src="data:image/png;base64,{img64}"
         style="width:100%;max-width:800px;margin:auto;display:block;"/>
    {short_html}{desc_html}{extra_html}
  </div>
</details></li>''')
    return ''.join(parts)


def generate_html_summary(tailwinds, headwinds, neutrals, bias, score,
                           phase_label, data, history, metrics, today):
    tw_html = build_html_section(tailwinds, data, history, metrics, today)
    hw_html = build_html_section(headwinds, data, history, metrics, today)
    nt_html = build_html_section(neutrals, data, history, metrics, today)

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Portfolio Bias Report</title>
<style>
body{{font-family:Segoe UI,sans-serif;padding:30px;background:#f5f5f5;color:#222;}}
.container{{max-width:1000px;margin:auto;background:white;padding:30px;
           border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}}
h1{{color:#0d47a1;margin-bottom:6px;}}
.meta{{color:#777;font-size:0.85em;margin-bottom:20px;}}
.scorecard{{display:flex;gap:15px;margin-bottom:30px;flex-wrap:wrap;}}
.card{{flex:1;min-width:150px;background:#f8f9fa;padding:15px;
      border:1px solid #e0e0e0;border-radius:6px;}}
.card-label{{font-size:0.75em;text-transform:uppercase;color:#888;}}
.card-value{{font-size:1.8em;font-weight:700;color:#0d47a1;}}
h2{{margin:30px 0 15px;padding-bottom:8px;border-bottom:2px solid #ddd;}}
h2.tw{{border-color:#4caf50;color:#2e7d32;}}
h2.hw{{border-color:#f44336;color:#c62828;}}
h2.nt{{border-color:#9e9e9e;}}
ul{{list-style:none;padding:0;}}
li{{margin-bottom:8px;}}
details>summary{{cursor:pointer;padding:10px;background:#f8f8f8;
               border:1px solid #e0e0e0;border-radius:4px;font-weight:600;}}
details[open]>summary{{background:#e8eaf6;}}
</style></head><body>
<div class="container">
<h1>📊 Portfolio Bias &amp; Sector Tilt Report</h1>
<p class="meta">Generated: {today.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

<div class="scorecard">
  <div class="card">
    <div class="card-label">GDP Growth Score</div>
    <div class="card-value">{score}<span style="font-size:0.4em;color:#888;"> / 150</span></div>
  </div>
  <div class="card">
    <div class="card-label">Recommended Bias</div>
    <div class="card-value" style="font-size:1.1em;">{bias}</div>
  </div>
  <div class="card">
    <div class="card-label">Cycle Phase</div>
    <div class="card-value" style="font-size:1.1em;">{phase_label}</div>
  </div>
  <div class="card">
    <div class="card-label">Conviction</div>
    <div class="card-value">{metrics.get('conviction', 0):.0%}</div>
  </div>
</div>

<h2 class="tw">✅ Tailwinds ({len(tailwinds)})</h2>
<ul>{tw_html}</ul>

<h2 class="hw">❌ Headwinds ({len(headwinds)})</h2>
<ul>{hw_html}</ul>

<h2 class="nt">⚖️ Neutrals ({len(neutrals)})</h2>
<ul>{nt_html}</ul>

</div></body></html>"""


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Portfolio Bias & Sector Tilt", layout="wide")
st.title("📊 Portfolio Bias & Sector Tilt Dashboard")

with st.sidebar:
    st.header("⚙️ Settings")
    portfolio_size = st.number_input("Portfolio Size ($)", min_value=10000, value=100000, step=10000)
    preferred_sectors = st.multiselect("Preferred Sectors", list(ALL_SECTORS.keys()), default=[])

tab1, tab2, tab3 = st.tabs(["📈 Analysis", "🎯 Sectors", "📊 Backtest"])

with tab1:
    if st.button("🔄 Run Analysis", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                data, history, today = fetch_data()
                metrics, tw, hw, nt, bias, score = calculate_metrics(data, history, today)

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Score", f"{score}/150")
                with col2: st.metric("Bias", bias.split('—')[0].strip())
                with col3: st.metric("Conviction", f"{metrics['conviction']:.0%}")
                with col4: st.metric("Phase", metrics['phase_label'])

                st.divider()
                col_tw, col_hw = st.columns(2)
                with col_tw:
                    st.subheader(f"✅ Tailwinds ({len(tw)})")
                    for t in tw[:6]: st.write(f"• {t}")
                with col_hw:
                    st.subheader(f"❌ Headwinds ({len(hw)})")
                    for h in hw[:6]: st.write(f"• {h}")

                html_report = generate_html_summary(tw, hw, nt, bias, score, metrics['phase_label'], data, history, metrics, today)
                st.download_button("📥 Download HTML Report", html_report, f"report_{today.date()}.html", "text/html")

                st.session_state.data    = data
                st.session_state.history = history
                st.session_state.metrics = metrics
                st.session_state.bias    = bias
                st.session_state.score   = score

            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    if 'metrics' in st.session_state:
        st.subheader("🎯 Sector Tilt Recommendations")
        with st.spinner("Computing sectors..."):
            tilt_df, _ = generate_sector_tilt(
                st.session_state.bias, st.session_state.score,
                st.session_state.metrics['phase'], st.session_state.metrics['conviction'],
                preferred_sectors, portfolio_size)
            if not tilt_df.empty:
                st.dataframe(tilt_df, use_container_width=True)
                st.download_button("📥 Download CSV", tilt_df.to_csv(index=False), "sectors.csv", "text/csv")
    else:
        st.info("Run Analysis first")

with tab3:
    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Running enhanced 10Y backtest..."):
            bt_df, bt_summary = run_backtest()
            if not bt_df.empty:
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("GDP Acc (Rules)", bt_summary.get('GDP Directional Accuracy (Rules)', 'N/A'))
                with col2: st.metric("GDP Acc (ML)", bt_summary.get('GDP Directional Accuracy (ML)', 'N/A'))
                with col3: st.metric("Strat Sharpe (Rules)", bt_summary.get('Strategy Sharpe (Rules)', 'N/A'))

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(bt_df.index, bt_df['SP500_Cum'], label='S&P 500', linewidth=2)
                ax.plot(bt_df.index, bt_df['Strat_Cum_Rules'], label='Strategy (Rules)', linewidth=2)
                ax.plot(bt_df.index, bt_df['Strat_Cum_ML'], label='Strategy (ML)', linewidth=2, linestyle='--')
                ax.set_title('10Y Cumulative Returns: Rules vs ML')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

                with st.expander("📋 Full Summary Statistics"):
                    for k, v in bt_summary.items():
                        st.write(f"**{k}:** {v}")
            else:
                st.warning("Backtest returned no data — check FRED/yfinance connectivity.")

st.markdown("---")
st.caption("✅ Complete | 10Y Backtest | HTML Reports | Sector Tilt")



