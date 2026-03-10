# ============================================================================
# DERIVED SERIES (for charts)
# ============================================================================

def _compute_real_rate_series(history, which='10yr'):
    """Real rate = nominal yield – 5Y breakeven inflation (monthly-aligned)."""
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


# ============================================================================
# CORE PLOTTING PRIMITIVES — LINE ONLY, NO SCATTER DOTS
# ============================================================================

def _plot_series(ax, series, title, hline=None, color='#1565C0', linewidth=2):
    """Clean line-only renderer with smart axis formatting."""
    ax.set_title(title, fontsize=10, fontweight='bold')
    if series is None or series.empty:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        return
    s = series.dropna()
    if s.empty:
        ax.text(0.5, 0.5, 'All values NaN',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        return
    if hline is not None:
        ax.axhline(hline, color='#E53935', linestyle='--', linewidth=1, alpha=0.7)
    s.plot(ax=ax, linewidth=linewidth, color=color)
    ax.grid(True, alpha=0.3)
    _apply_axis_format(ax, s)


def _plot_macd_bars(ax, macd_vals, sig_vals, hist_vals, x_dates, title):
    """Shared MACD bar+line renderer."""
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


def generate_macd_4panel(history, today):
    """4-panel MACD: 5Y price | 1M price | 12M MACD | 1M MACD."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('LazyMan Investor — S&P500 & MACD (12,26,9)', fontsize=13, fontweight='bold')
    ax_5y, ax_1m, ax_12m, ax_macd_1m = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    sp5y = history['sp500_long'].dropna()
    sp1y = history['sp500'].dropna()

    # Top-left: 5Y price
    _plot_series(ax_5y, sp5y, 'S&P 500 — 5 Year', color='#1565C0', linewidth=1.5)

    # Top-right: 1M price
    sp1m = safe_last(sp1y, 31)
    _plot_series(ax_1m, sp1m, 'S&P 500 — 1 Month', color='#1565C0', linewidth=1.8)

    # Bottom panels: MACD computed on full 1Y series
    if len(sp1y) >= 26:
        macd_full, sig_full, hist_full = compute_macd(sp1y)

        # 12M MACD
        x12 = mdates.date2num(macd_full.index.to_pydatetime())
        _plot_macd_bars(ax_12m, macd_full, sig_full, hist_full, x12, 'MACD (12,26,9) — 12 Months')

        # 1M MACD slice
        cut1m = pd.Timestamp(today - timedelta(days=31))
        m1m   = macd_full[macd_full.index >= cut1m]
        s1m   = sig_full[sig_full.index >= cut1m]
        h1m   = hist_full[hist_full.index >= cut1m]
        if not m1m.empty:
            x1m = mdates.date2num(m1m.index.to_pydatetime())
            _plot_macd_bars(ax_macd_1m, m1m, s1m, h1m, x1m, 'MACD (12,26,9) — 1 Month')
        else:
            ax_macd_1m.text(0.5, 0.5, 'No 1M MACD data',
                            ha='center', va='center', transform=ax_macd_1m.transAxes, color='gray')
    else:
        for ax in [ax_12m, ax_macd_1m]:
            ax.text(0.5, 0.5, f'Insufficient data ({len(sp1y)} pts, need ≥26)',
                    ha='center', va='center', transform=ax.transAxes, color='gray')

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN CHART (12M / long window)
# ============================================================================

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

    elif metric_key == 'yield_curve_compare':
        cut  = pd.Timestamp(today - timedelta(days=1095))
        ten  = history['10yr_yield'][history['10yr_yield'].index >= cut].dropna()
        two  = history['2yr_yield'].reindex(ten.index, method='nearest',
                                            tolerance=pd.Timedelta('5D')).dropna()
        series = (ten - two).dropna()
        hline  = 0
        ax.set_title('10Yr-2Yr Spread (last 3 years)')

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


# ============================================================================
# SHORT-TERM CHART (3M / ~6M for monthly series)
# ============================================================================

def generate_short_term_graph(metric_key, history, today):
    """Returns None for keys in NO_SHORT_TERM_CHART."""
    if metric_key in NO_SHORT_TERM_CHART:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    series  = None
    hline   = None

    if metric_key == 'copper_gold':
        ci = history['copper'].index.intersection(history['gold'].index)
        if len(ci):
            series = _short_term_window(history['copper'].reindex(ci) / history['gold'].reindex(ci))
        ax.set_title('Copper/Gold Ratio — Recent View')

    elif metric_key == 'spread_10ff':
        y10    = history['10yr_yield']
        ff     = history['fed_funds'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('35D'))
        series = _short_term_window((y10 - ff).dropna())
        hline  = 0
        ax.set_title('10Yr-FedFunds Spread — Recent View')

    elif metric_key == 'spread_10_2':
        y10    = history['10yr_yield']
        y2     = history['2yr_yield'].reindex(y10.index, method='nearest',
                                              tolerance=pd.Timedelta('5D'))
        series = _short_term_window((y10 - y2).dropna())
        hline  = 0
        ax.set_title('10Yr-2Yr Spread — Recent View')

    elif metric_key == 'yield_curve_compare':
        cut    = pd.Timestamp(today - timedelta(days=90))
        ten    = history['10yr_yield'][history['10yr_yield'].index >= cut].dropna()
        two    = history['2yr_yield'].reindex(ten.index, method='nearest',
                                              tolerance=pd.Timedelta('5D')).dropna()
        series = (ten - two).dropna()
        hline  = 0
        ax.set_title('10Yr-2Yr Spread — Last 3 Months')

    elif metric_key == 'real_rate_10yr':
        series = _short_term_window(_compute_real_rate_series(history, '10yr').dropna())
        hline  = 0
        ax.set_title('Real Rate 10Yr — Recent View')

    elif metric_key == 'real_rate_2yr':
        series = _short_term_window(_compute_real_rate_series(history, '2yr').dropna())
        hline  = 0
        ax.set_title('Real Rate 2Yr — Recent View')

    elif metric_key == 'breakeven_5y':
        series = _short_term_window(history['breakeven_5y'])
        hline  = 2.0
        ax.set_title('5Y Breakeven Inflation — Recent View')

    elif metric_key == 'fed_bs':
        series = _short_term_window(history['fed_bs'])
        ax.set_title('Fed Balance Sheet — Recent View')

    elif metric_key == 'macd':
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
                ax.text(0.5, 0.5, 'No 3M data',
                        ha='center', va='center', transform=ax.transAxes, color='gray')
        else:
            ax.text(0.5, 0.5, f'Not enough data ({len(sp_full)} pts)',
                    ha='center', va='center', transform=ax.transAxes, color='gray')
        plt.tight_layout()
        return fig

    elif metric_key == 'sp_96':
        series = _short_term_window(history['sp500'])
        ax.set_title('S&P 500 — Last 3 Months (current momentum)')

    elif metric_key == 'stoxx_96':
        series = _short_term_window(history['stoxx600'])
        ax.set_title('STOXX 600 — Last 3 Months (current momentum)')

    elif metric_key == 'building_permits':
        series = _short_term_window(history['building_permits'])
        ax.set_title('Building Permits — Recent View (~6M)')

    elif metric_key in ('m1', 'm2'):
        series = _short_term_window(history.get(metric_key, pd.Series(dtype=float)))
        ax.set_title(f"{metric_key.upper()} — Recent View (~6M)")

    elif metric_key in history:
        series = _short_term_window(history[metric_key])
        ax.set_title(f"{metric_key.replace('_', ' ').upper()} — Recent View")

    _plot_series(ax, series, ax.get_title(), hline=hline)
    plt.tight_layout()
    return fig


# ============================================================================
# GRAPH KEY RESOLVER
# ============================================================================

def get_graph_key(item_text):
    t = item_text
    if 'Copper/Gold'          in t: return 'copper_gold'
    if '10Yr-FedFunds'        in t: return 'spread_10ff'
    if '10Yr-2Yr'             in t: return 'spread_10_2'
    if 'Yield Curve'          in t: return 'yield_curve_compare'
    if 'Real Rate' in t and '10' in t: return 'real_rate_10yr'
    if 'Real Rate' in t and '2'  in t: return 'real_rate_2yr'
    if 'Breakeven'            in t: return 'breakeven_5y'
    if 'Balance Sheet'        in t: return 'fed_bs'
    if 'Fed Funds'            in t: return 'fed_funds'
    if '10-Yr'                in t: return '10yr_yield'
    if '2-Yr'                 in t: return '2yr_yield'
    if 'Core CPI'             in t: return 'core_cpi'
    if 'BBB'                  in t: return 'bbb_yield'
    if 'CCC'                  in t: return 'ccc_yield'
    if 'VIX'                  in t: return 'vix'
    if 'MOVE'                 in t: return 'move'
    if 'Manufacturing PMI'    in t: return 'ism_manufacturing'
    if 'Services PMI'         in t: return 'ism_services'
    if 'UMCSI'                in t: return 'umcsi'
    if 'Building'             in t: return 'building_permits'
    if 'NFIB'                 in t: return 'nfib'
    if 'CPI Volatile'         in t: return 'cpi_volatile'
    if 'SBI'                  in t: return 'sbi'
    if 'EESI'                 in t: return 'eesi'
    if 'Earnings Growth'      in t: return 'earnings_growth'
    if '9-6' in t and 'S&P'   in t: return 'sp_96'
    if '9-6' in t and 'STOXX' in t: return 'stoxx_96'
    if 'MACD'                 in t or 'LazyMan' in t: return 'macd'
    if 'S&P'                  in t: return 'sp500'
    if 'STOXX'                in t: return 'stoxx600'
    return 'placeholder'


def get_description(gkey):
    d = {
        'macd': (
            'You could stop now and just do this.<br>'
            'MACD identifies momentum: when the short-term EMA crosses above the long-term, '
            'it signals a potential uptrend; a cross below signals a downtrend.<br>'
            '<a href="https://moneyweek.com/518350/macd-histogram-stockmarket-trading-system-for-the-lazy-investor" '
            'target="_blank">The LazyMan MACD article (MoneyWeek)</a><br>'
            'In 19 years: 12 trades (6 Buy, 6 Sell). Catches major moves; misses exact highs/lows.'),
        'sp500': 'S&P 500 is a forward-looking GDP indicator. Correlation with US GDP: ~69%.',
        'stoxx600': 'STOXX 600 as global risk appetite proxy (~55% correlation with US GDP via trade/finance channels).',
        'spread_10ff': '10-Year minus Fed Funds rate. Positive = steep curve = accommodative conditions → expansionary.',
        'spread_10_2': '10-Year minus 2-Year spread (classic yield curve). Positive = expansion signal; inversion historically precedes recession by 12-18M.',
        'yield_curve_compare': '3-year view of 10Yr-2Yr spread showing the full cycle context.',
        'real_rate_10yr': 'Real rate 10Yr = Nominal yield − 5Y breakeven inflation. Forward-looking; negative = highly stimulative.',
        'real_rate_2yr': 'Real rate 2Yr = 2Y yield − 5Y breakeven inflation. Captures short-term real policy rate.',
        'breakeven_5y': '5Y breakeven inflation (FRED T5YIFR) = market-implied inflation expectation. Replaces backward-looking CPI for real rate calc.',
        'fed_bs': 'Fed balance sheet total assets. Expanding = QE / easing. Contracting = QT / tightening. Replaced M1/M2 as primary liquidity signal.',
        'vix': 'VIX (fear index). Signal uses trend + level: healthy de-escalation from 14-24 = bullish; <12 (complacency) or >28 (panic) = negative.',
        'bbb_yield': 'BBB corporate bond yields. Lower = cheaper borrowing = expansionary. Rising = credit stress.',
        'ccc_yield': 'CCC high-yield bond yields. Rising sharply = credit market stress = contractionary.',
        'sp_96': 'S&P 9-to-6-months-ago return: ~69% correlation with future GDP direction. Top chart = signal window (the key data). Bottom = current momentum.',
        'stoxx_96': 'STOXX 600 9-to-6M return as a global leading indicator. Top = signal window. Bottom = current momentum.',
        'core_cpi': 'Core CPI YoY (excl. food & energy). Shown for reference — real rates now use forward-looking 5Y breakeven.',
        'cpi_volatile': 'Headline CPI YoY (incl. food & energy). Reference only — exogenous oil/food shocks reduce signal quality.',
        'ism_manufacturing': 'ISM Manufacturing PMI. >50 = expansion; <50 = contraction. Cycle-leading indicator for industrial output.',
        'ism_services': 'ISM Services PMI. Services is 70%+ of US GDP — this is the dominant activity gauge.',
        'building_permits': 'Building permits lead housing starts by 1-2 months and are a reliable leading economic indicator.',
        'nfib': 'NFIB Small Business Optimism Index. >100 = strong confidence; <95 = weakness. Small businesses represent ~50% of US GDP.',
        'umcsi': 'University of Michigan Consumer Sentiment. >70 = bullish; <55 = bearish for consumer spending.',
        'earnings_growth': 'Forward 12M EPS growth estimate. Negative earnings growth is a structural headwind; >5% is a tailwind.',
        'copper_gold': 'Copper/Gold ratio: rising = growth expectations outpacing fear = positive leading indicator for risk assets.',
    }
    return d.get(gkey, '')


# ============================================================================
# STOXX 9-6M DYNAMIC TABLE
# ============================================================================

def _build_stoxx_96_table_html(history):
    stoxx = history['stoxx600'].dropna()
    if len(stoxx) < 200:
        return ''
    i9, i6, ic = max(0, len(stoxx)-189), max(0, len(stoxx)-126), len(stoxx)-1
    d9, d6, dc = (stoxx.index[i9].strftime('%d/%m/%Y'),
                  stoxx.index[i6].strftime('%d/%m/%Y'),
                  stoxx.index[ic].strftime('%d/%m/%Y'))
    p9, p6, pc = float(stoxx.iloc[i9]), float(stoxx.iloc[i6]), float(stoxx.iloc[ic])
    r96     = (p6 - p9) / p9 * 100 if p9 != 0 else 0
    r6c     = (pc - p6) / p6 * 100 if p6 != 0 else 0
    sc      = '#28a745' if r96 >= 0 else '#dc3545'
    sig     = '📈 Bullish Signal' if r96 >= 0 else '📉 Bearish Signal'
    mc      = '#28a745' if r6c >= 0 else '#dc3545'
    return f'''
    <h4 style="margin:25px 0 10px;color:#444;">STOXX 600 — 9-to-6 Month Leading Signal Window</h4>
    <table style="width:100%;max-width:820px;margin:12px auto;border-collapse:collapse;
                  font-size:0.93em;border:1px solid #ddd;">
      <thead><tr style="background:#f5f5f5;">
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Metric</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:left;">Date</th>
        <th style="padding:10px;border:1px solid #ddd;text-align:right;">Value</th>
      </tr></thead>
      <tbody>
        <tr>
          <td style="padding:9px;border:1px solid #ddd;">Price at 9M Ago (signal start)</td>
          <td style="padding:9px;border:1px solid #ddd;">{d9}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;">{p9:,.2f}</td>
        </tr>
        <tr>
          <td style="padding:9px;border:1px solid #ddd;">Price at 6M Ago (signal end)</td>
          <td style="padding:9px;border:1px solid #ddd;">{d6}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;">{p6:,.2f}</td>
        </tr>
        <tr style="background:#f0fff4;">
          <td style="padding:9px;border:1px solid #ddd;font-weight:bold;">9→6M Return (leading signal)</td>
          <td style="padding:9px;border:1px solid #ddd;">{d9} → {d6}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;font-weight:bold;
                     color:{sc};">{r96:+.2f}%</td>
        </tr>
        <tr>
          <td style="padding:9px;border:1px solid #ddd;">Current Price</td>
          <td style="padding:9px;border:1px solid #ddd;">{dc}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;">{pc:,.2f}</td>
        </tr>
        <tr>
          <td style="padding:9px;border:1px solid #ddd;">Return since 6M Ago (momentum)</td>
          <td style="padding:9px;border:1px solid #ddd;">{d6} → {dc}</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;color:{mc};">{r6c:+.2f}%</td>
        </tr>
        <tr style="background:#f8f8f8;">
          <td style="padding:9px;border:1px solid #ddd;font-weight:bold;" colspan="2">Signal Interpretation</td>
          <td style="padding:9px;border:1px solid #ddd;text-align:right;font-weight:bold;
                     color:{sc};">{sig}</td>
        </tr>
      </tbody>
    </table>'''


# ============================================================================
# METRICS / SCORING (optimised from backtest)
# ============================================================================

def calculate_metrics(data, history, today):
    metrics = {}

    # Derived metrics
    metrics['yield_curve_10_2']      = data['10yr_yield'] - data['2yr_yield']
    metrics['yield_curve_10ff']      = data['10yr_yield'] - data['fed_funds']
    metrics['real_rate_10yr']        = data['real_rate_10yr']
    metrics['real_rate_2yr']         = data['real_rate_2yr']
    metrics['copper_gold_ratio']     = data['copper'] / data['gold'] if data['gold'] != 0 else 0
    metrics['copper_gold_chg']       = (
        (float(history['copper'].iloc[-1]) / float(history['gold'].iloc[-1]))
      - (float(history['copper'].iloc[0])  / float(history['gold'].iloc[0])))

    # Cycle phase
    phase               = detect_cycle_phase(
        metrics['yield_curve_10_2'],
        metrics['real_rate_10yr'],
        data['ism_manufacturing'],
        data['earnings_growth'])
    metrics['phase']    = phase
    phase_cfg           = CYCLE_PHASE_CONFIG[phase]

    tailwinds, headwinds, neutrals = [], [], []

    # --- Scoring engine (optimised weights) ---
    score = 50   # Start at neutral centre

    W = OPTIMIZED_WEIGHTS   # Shorthand

    # 1. Yield curve 10Y-2Y (strongest leading signal)
    yc = metrics['yield_curve_10_2']
    if yc > 1.0:
        score += W['yield_curve_10_2']
        tailwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Steep curve (strong expansion signal)")
    elif yc > 0.5:
        score += int(W['yield_curve_10_2'] * 0.7)
        tailwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Positive (expansion)")
    elif yc > 0:
        score += int(W['yield_curve_10_2'] * 0.3)
        neutrals.append(f"10Yr-2Yr Spread: {yc:.2f}% — Barely positive (caution)")
    elif yc > -0.5:
        score -= int(W['yield_curve_10_2'] * 0.6)
        headwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — Slightly inverted (warning)")
    else:
        score -= W['yield_curve_10_2']
        headwinds.append(f"10Yr-2Yr Spread: {yc:.2f}% — INVERTED (recession signal)")

    # 2. Yield curve 10Y-FedFunds
    ff_sp = metrics['yield_curve_10ff']
    if ff_sp > 0:
        score += W['yield_curve_10ff']
        tailwinds.append(f"10Yr-FedFunds Spread: {ff_sp:.2f}% — Positive (accommodative)")
    else:
        score -= W['yield_curve_10ff']
        headwinds.append(f"10Yr-FedFunds Spread: {ff_sp:.2f}% — Negative (restrictive)")

    # 3. Real rate 10Yr (forward-looking — breakeven inflation)
    rr10 = metrics['real_rate_10yr']
    if rr10 < -1.0:
        score += W['real_rate_10yr']
        tailwinds.append(f"Real Rate 10Yr: {rr10:.2f}% — Highly stimulative")
    elif rr10 < 0:
        score += int(W['real_rate_10yr'] * 0.6)
        tailwinds.append(f"Real Rate 10Yr: {rr10:.2f}% — Stimulative")
    elif rr10 < 1.0:
        neutrals.append(f"Real Rate 10Yr: {rr10:.2f}% — Neutral territory")
    else:
        score -= W['real_rate_10yr']
        headwinds.append(f"Real Rate 10Yr: {rr10:.2f}% — Restrictive")

    # 4. Real rate 2Yr
    rr2 = metrics['real_rate_2yr']
    if rr2 < 0:
        score += W['real_rate_2yr']
        tailwinds.append(f"Real Rate 2Yr: {rr2:.2f}% — Stimulative")
    elif rr2 < 1.0:
        neutrals.append(f"Real Rate 2Yr: {rr2:.2f}% — Neutral")
    else:
        score -= W['real_rate_2yr']
        headwinds.append(f"Real Rate 2Yr: {rr2:.2f}% — Restrictive")

    # 5. Earnings growth
    eg = data['earnings_growth']
    metrics['earnings_growth'] = eg
    if eg > 10:
        score += W['earnings_growth']
        tailwinds.append(f"Earnings Growth (fwd 12M): {eg:.1f}% — Strong")
    elif eg > 5:
        score += int(W['earnings_growth'] * 0.6)
        tailwinds.append(f"Earnings Growth (fwd 12M): {eg:.1f}% — Positive")
    elif eg > 0:
        neutrals.append(f"Earnings Growth (fwd 12M): {eg:.1f}% — Weak positive")
    else:
        score -= W['earnings_growth']
        headwinds.append(f"Earnings Growth (fwd 12M): {eg:.1f}% — Negative (structural headwind)")

    # 6. MACD
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
    if macd_lb:
        score += W['macd_long']
        tailwinds.append(
            f"LazyMan MACD: Short {'Buy' if macd_sb else 'Sell'} | Long Buy → Bull signal — Simply Buy")
    else:
        score -= int(W['macd_long'] * 0.5)
        headwinds.append(
            f"LazyMan MACD: Short {'Buy' if macd_sb else 'Sell'} | Long Sell → Bear signal")

    # 7. Fed Balance Sheet growth (replaces M1/M2)
    fbs = data['fed_bs_growth']
    metrics['fed_bs_growth'] = fbs
    if fbs > 3:
        score += W['fed_bs_growth']
        tailwinds.append(f"Fed Balance Sheet Growth: {fbs:.1f}% YoY — Easing (QE / liquidity injection)")
    elif fbs > 0:
        score += int(W['fed_bs_growth'] * 0.5)
        tailwinds.append(f"Fed Balance Sheet Growth: {fbs:.1f}% YoY — Slight easing")
    elif fbs > -3:
        neutrals.append(f"Fed Balance Sheet Growth: {fbs:.1f}% YoY — Slight tightening")
    else:
        score -= W['fed_bs_growth']
        headwinds.append(f"Fed Balance Sheet Growth: {fbs:.1f}% YoY — Tightening (QT / liquidity drain)")

    # 8. VIX trend signal
    vt = data['vix_trend']
    metrics['vix_trend'] = vt
    vix_val = float(data['vix'])
    if vt > 0:
        score += W['vix_trend']
        tailwinds.append(f"VIX: {vix_val:.1f} — Healthy, declining trend (positive for risk assets)")
    elif vt < 0:
        score -= W['vix_trend']
        headwinds.append(f"VIX: {vix_val:.1f} — Complacency (<12) or Panic (>28) — caution")
    else:
        neutrals.append(f"VIX: {vix_val:.1f} — Neutral (elevated but not trending)")

    # 9. S&P 9-6M return
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
        tailwinds.append(f"S&P 9-6M Return: {sp96:.2f}% — Positive leading signal")
    elif sp96 > -5:
        score -= int(W['sp_96'] * 0.5)
        headwinds.append(f"S&P 9-6M Return: {sp96:.2f}% — Slight negative leading signal")
    else:
        score -= W['sp_96']
        headwinds.append(f"S&P 9-6M Return: {sp96:.2f}% — Negative leading signal")

    # 10. STOXX 9-6M return
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
        tailwinds.append(f"STOXX 600 9-6M Return: {st96:.2f}% — Positive global signal")
    else:
        score -= int(W['stoxx_96'] * 0.5)
        headwinds.append(f"STOXX 600 9-6M Return: {st96:.2f}% — Negative global signal")

    # 11. Copper/Gold
    cgc = metrics['copper_gold_chg']
    if cgc > 0:
        score += W['copper_gold']
        tailwinds.append(f"Copper/Gold Ratio: Rising ({cgc:+.4f}) — growth > fear (positive)")
    else:
        score -= int(W['copper_gold'] * 0.5)
        headwinds.append(f"Copper/Gold Ratio: Falling ({cgc:+.4f}) — fear > growth (negative)")

    # 12. Manufacturing PMI
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

    # 13. Services PMI
    sv = data['ism_services']
    if sv > 52:
        score += W['ism_services']
        tailwinds.append(f"Services PMI: {sv:.1f} — Strong expansion")
    elif sv > 50:
        score += int(W['ism_services'] * 0.5)
        tailwinds.append(f"Services PMI: {sv:.1f} — Expansion")
    else:
        score -= W['ism_services']
        headwinds.append(f"Services PMI: {sv:.1f} — Contraction")

    # 14. Building permits
    bp = data['building_permits']
    bph = history['building_permits']
    bp_chg = float(bph.iloc[-1]-bph.iloc[-2]) if len(bph) > 1 else 0
    if bp_chg > 0:
        score += W['building_permits']
        tailwinds.append(f"Building Permits: {bp:.2f}M — Rising (positive leading indicator)")
    else:
        score -= int(W['building_permits'] * 0.5)
        headwinds.append(f"Building Permits: {bp:.2f}M — Falling (negative leading indicator)")

    # 15. NFIB
    nf = data['nfib']
    if nf > 100:
        score += W['nfib']
        tailwinds.append(f"NFIB Small Business Optimism: {nf:.1f} — Strong confidence (>100)")
    elif nf < 95:
        score -= W['nfib']
        headwinds.append(f"NFIB Small Business Optimism: {nf:.1f} — Weak confidence (<95)")
    else:
        neutrals.append(f"NFIB Small Business Optimism: {nf:.1f} — Neutral (95–100)")

    # 16. UMCSI
    um = data['umcsi']
    if um > 70:
        score += W['umcsi']
        tailwinds.append(f"UMCSI Consumer Sentiment: {um:.1f} — Bullish (>70)")
    elif um < 55:
        score -= W['umcsi']
        headwinds.append(f"UMCSI Consumer Sentiment: {um:.1f} — Bearish (<55)")
    else:
        neutrals.append(f"UMCSI Consumer Sentiment: {um:.1f} — Neutral")

    # 17. BBB corporate yield
    bbb = data['bbb_yield']
    bbb_chg = (float(history['bbb_yield'].iloc[-1]) - float(history['bbb_yield'].iloc[-2])
               if len(history['bbb_yield']) > 1 else 0)
    if bbb_chg < 0:
        score += W['bbb_yield']
        tailwinds.append(f"BBB Yield: {bbb:.2f}% — Declining (cheaper borrowing, positive)")
    else:
        score -= int(W['bbb_yield'] * 0.5)
        headwinds.append(f"BBB Yield: {bbb:.2f}% — Rising (tightening credit, negative)")

    # Cycle phase boost / penalty
    boost = phase_cfg['boost']
    score += boost
    if boost > 0:
        tailwinds.append(
            f"Cycle Phase: {phase_cfg['label']} — Score boosted by +{boost} pts")
    elif boost < 0:
        headwinds.append(
            f"Cycle Phase: {phase_cfg['label']} — Score penalised by {boost} pts")
    else:
        neutrals.append(f"Cycle Phase: {phase_cfg['label']} — No adjustment")

    # Clamp and determine bias
    score = max(0, min(150, int(score)))
    metrics['score_raw'] = score

    if phase_cfg.get('force_short'):
        bias  = 'Short (recession regime — force override)'
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

    # Conviction level (0–1)
    metrics['conviction'] = abs(score - 50) / 50.0
    metrics['bias']       = bias
    metrics['phase_label'] = phase_cfg['label']

    # S&P and STOXX lagging (for HTML report)
    data['sp_lagging']    = 'UP' if history['sp500'].iloc[-1] > history['sp500'].iloc[0] else 'DOWN'

    # Bear/Bull tables
    for bkey, bh, bav in [('sp_bear',    'sp500_long',    997),
                           ('stoxx_bear', 'stoxx600_long', 857)]:
        try:
            sl  = history[bh].dropna()
            if len(sl) >= 10:
                cur   = float(sl.iloc[-1])
                ath   = float(sl.max())
                nh    = sl[sl == ath].index[-1]
                rec   = sl[sl.index >= today - timedelta(days=1825)]
                pl    = float(rec.min())  if not rec.empty else 0.0
                pld   = rec.idxmin()      if not rec.empty else sl.index[0]
                db    = (today.date() - pld.date()).days
                metrics[bkey] = dict(
                    current_date=sl.index[-1].strftime('%d/%m/%Y'), current=cur,
                    last_high_date=nh.strftime('%d/%m/%Y'), last_high=ath,
                    new_bear_threshold=ath * 0.8,
                    prev_bear_date=pld.strftime('%d/%m/%Y'), prev_bear=pl,
                    days_bull=db, avg_days_bull=bav)
            else:
                metrics[bkey] = {}
        except Exception:
            metrics[bkey] = {}

    # 10Yr terminal
    metrics['terminal_10yr'] = float(history['10yr_yield'].max())

    return metrics, tailwinds, headwinds, neutrals, bias, score


# ============================================================================
# BACKTEST ENGINE (10Y quarterly)
# ============================================================================

@st.cache_data(ttl=86400)
def run_backtest():
    """
    10-year quarterly backtest (Q1 2014 – Q4 2023).
    At each quarter-end date we reconstruct macro indicators from
    historical FRED / yfinance data and compute the bias score.
    Strategy return = next-quarter S&P 500 return × directional position.
    """
    quarters = pd.date_range(start='2014-01-01', end='2024-01-01', freq='QE')
    results  = []

    # Fetch full historical series once
    try:
        sp_hist    = yf.download('^GSPC',  start='2013-01-01', end='2024-06-01', progress=False)['Close']
        sp_hist.index = pd.DatetimeIndex(sp_hist.index).tz_localize(None)
    except Exception:
        sp_hist = pd.Series(dtype=float)

    try:
        y10_hist = fred.get_series('DGS10', observation_start='2010-01-01')
        y10_hist = normalize_index(y10_hist)
        y2_hist  = fred.get_series('DGS2',  observation_start='2010-01-01')
        y2_hist  = normalize_index(y2_hist)
        ff_hist  = fred.get_series('FEDFUNDS', observation_start='2010-01-01')
        ff_hist  = normalize_index(ff_hist)
        be_hist  = fred.get_series('T5YIFR', observation_start='2010-01-01')
        be_hist  = normalize_index(be_hist)
        pmi_hist = fred.get_series('NAPM', observation_start='2010-01-01')
        pmi_hist = normalize_index(pmi_hist)
        bp_hist  = fred.get_series('PERMIT', observation_start='2010-01-01')
        bp_hist  = normalize_index(bp_hist)
        umi_hist = fred.get_series('UMCSENT', observation_start='2010-01-01')
        umi_hist = normalize_index(umi_hist)
    except Exception:
        st.warning("⚠️ Could not fetch full FRED history for backtest. Results will be approximate.")
        return pd.DataFrame(), {}

    def _get_val(series, date, default=0):
        """Get most recent value of series at or before date."""
        try:
            sub = series[series.index <= date]
            return float(sub.iloc[-1]) if not sub.empty else default
        except Exception:
            return default

    def _get_sp_return(sp, date_start, date_end):
        """Compute S&P 500 total return between two dates."""
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

    def _sp_96_return(sp, date):
        """S&P 9-to-6 month ago return as of date."""
        try:
            sub = sp[sp.index <= date]
            if len(sub) < 200:
                return 0.0
            p9 = float(sub.iloc[max(0, len(sub)-189)])
            p6 = float(sub.iloc[max(0, len(sub)-126)])
            return (p6-p9)/p9*100 if p9 != 0 else 0.0
        except Exception:
            return 0.0

    def _score_at_date(date):
        """Reconstruct macro bias score at a historical date."""
        y10  = _get_val(y10_hist, date, 2.5)
        y2   = _get_val(y2_hist,  date, 2.0)
        ff   = _get_val(ff_hist,  date, 1.5)
        be   = _get_val(be_hist,  date, 2.0)
        pmi  = _get_val(pmi_hist, date, 51.0)
        umi  = _get_val(umi_hist, date, 65.0)
        bp   = _get_val(bp_hist,  date, 1400) / 1000.0

        rr10    = y10 - be
        yc_10_2 = y10 - y2
        yc_10ff = y10 - ff
        sp96    = _sp_96_return(sp_hist, date)
        eg_proxy = 5.0  # Constant 5% proxy (earnings growth data sparse historically)

        phase = detect_cycle_phase(yc_10_2, rr10, pmi, eg_proxy)

        # Simplified scoring (same weights, subset of signals available historically)
        s = 50
        s += 18 if yc_10_2 > 1.0 else (12 if yc_10_2 > 0.5 else (5 if yc_10_2 > 0 else (-12 if yc_10_2 > -0.5 else -18)))
        s += 12 if yc_10ff > 0 else -12
        s += 7  if rr10 < -1  else (4 if rr10 < 0 else (-7 if rr10 > 1 else 0))
        s += 8  if sp96  > 5  else (4 if sp96 > 0 else (-4 if sp96 > -5 else -8))
        s += 4  if pmi   > 52 else (2 if pmi > 50 else -4)
        s += 3  if umi   > 70 else (0 if umi > 55 else -3)
        s += 4  if bp    > 1.4 else -2
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

    # Run quarter by quarter
    for i in range(len(quarters) - 1):
        q_date      = quarters[i]
        q_next      = quarters[i + 1]
        score, bias, phase = _score_at_date(q_date)
        sp_ret      = _get_sp_return(sp_hist, q_date, q_next)

        # Strategy position
        if bias == 'Long':
            strat_ret = sp_ret
        elif bias == 'Short':
            strat_ret = -sp_ret
        else:
            strat_ret = 0.0

        strat_ret -= 0.001  # Transaction cost

        results.append({
            'Date':         q_date,
            'Quarter':      q_date.strftime('%Y-Q%q') if hasattr(q_date, 'quarter') else str(q_date)[:7],
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
    ann_factor = 4  # Quarterly → annual

    sp_total   = df['SP500_Cum'].iloc[-1] - 1
    st_total   = df['Strat_Cum'].iloc[-1] - 1
    sp_ann     = (1 + sp_total) ** (ann_factor / n) - 1
    st_ann     = (1 + st_total) ** (ann_factor / n) - 1
    st_vol     = df['Strat_Return'].std() / 100 * np.sqrt(ann_factor)
    sp_vol     = df['SP500_Return'].std() / 100 * np.sqrt(ann_factor)
    st_sharpe  = (st_ann - 0.02) / st_vol  if st_vol  != 0 else 0
    sp_sharpe  = (sp_ann - 0.02) / sp_vol  if sp_vol  != 0 else 0
    st_mdd     = (df['Strat_Cum'] / df['Strat_Cum'].cummax() - 1).min()
    sp_mdd     = (df['SP500_Cum'] / df['SP500_Cum'].cummax() - 1).min()

    correct     = ((df['Bias'] == 'Long')  & (df['SP500_Return'] > 0)) | \
                  ((df['Bias'] == 'Short') & (df['SP500_Return'] < 0)) | \
                  ((df['Bias'] == 'Neutral') & (abs(df['SP500_Return']) < 2))
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
# SECTOR TILT (cycle-aware + valuation-filtered + pair trading)
# ============================================================================

def generate_sector_tilt(bias, score, phase, conviction, preferred_sectors, portfolio_size):
    """
    Builds sector recommendations using:
    1. Cycle-phase rotation weights (from SECTOR_ROTATION)
    2. Momentum filter (12M + 3M blend)
    3. Valuation filter (P/E z-score)
    4. Pair trading (long top N / short bottom N)
    5. Conviction-scaled allocation
    """
    today       = datetime.now()
    three_m_ago = pd.Timestamp(today - timedelta(days=90))

    # Fetch momentum
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

        # Valuation filter
        zscore = get_pe_zscore(sector, info['etf'])
        pe_data[sector]  = zscore

    # Rotation weights for this phase
    rot = SECTOR_ROTATION.get(phase, SECTOR_ROTATION['mid'])

    # Adjusted score: 50% rotation + 50% val-adjusted momentum
    adjusted = {}
    for sector in ALL_SECTORS:
        adj_mom = momentum[sector]
        z       = pe_data[sector]
        if z > 1.5:
            adj_mom *= 0.7   # Overvalued → discount
        elif z < -1.5:
            adj_mom *= 1.2   # Undervalued → boost
        rotation_wt = rot.get(sector, 0.05)
        adjusted[sector] = 0.5 * rotation_wt + 0.5 * adj_mom

    sorted_s = sorted(adjusted, key=adjusted.get, reverse=True)

    # Conviction-scaled allocation
    conv_clamped = max(0, min(1, conviction))
    if 'Long' in bias:
        long_pct  = 0.5 + 0.3 * conv_clamped   # 50–80%
        short_pct = 1 - long_pct
        n_long, n_short = 3, 3
    elif 'Short' in bias:
        short_pct = 0.5 + 0.3 * conv_clamped
        long_pct  = 1 - short_pct
        n_long, n_short = 3, 3
    else:
        long_pct  = 0.5
        short_pct = 0.5
        n_long, n_short = 3, 3

    # If user specified preferred sectors, prioritise those among longs
    if preferred_sectors:
        pref_ranked = [s for s in sorted_s if s in preferred_sectors]
        rest_ranked = [s for s in sorted_s if s not in preferred_sectors]
        long_pool   = (pref_ranked + rest_ranked)[:n_long]
    else:
        long_pool = sorted_s[:n_long]

    short_pool = [s for s in reversed(sorted_s) if s not in long_pool][:n_short]

    # Pair trading: pair top vs bottom
    pairs = list(zip(long_pool, short_pool))
    n_pairs = len(pairs)
    if n_pairs == 0:
        return pd.DataFrame(), {}

    long_alloc  = portfolio_size * long_pct  / n_pairs
    short_alloc = portfolio_size * short_pct / n_pairs

    rows = []
    for i, (ls, ss) in enumerate(pairs):
        rows.append({
            'Pair':              f"Pair {i+1}",
            'Long Sector':       ls,
            'Long ETF':          ALL_SECTORS[ls]['etf'],
            'Long Alloc ($)':    f"${long_alloc:,.0f}",
            'Short Sector':      ss,
            'Short ETF':         ALL_SECTORS[ss]['etf'],
            'Short Alloc ($)':   f"${short_alloc:,.0f}",
            'Long Rotation Wt':  f"{rot.get(ls, 0):.1%}",
            'Short Rotation Wt': f"{rot.get(ss, 0):.1%}",
            'Long PE Zscore':    f"{pe_data[ls]:.2f}",
            'Short PE Zscore':   f"{pe_data[ss]:.2f}",
        })

    return pd.DataFrame(rows), {'long_pct': long_pct, 'short_pct': short_pct}


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def build_html_section(items_list, data, history, metrics, today):
    """Build collapsible HTML sections for each indicator."""
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
            extra_html += _build_stoxx_96_table_html(history)

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


def generate_html_summary(tailwinds, headwinds, neutrals, bias, score, phase_label, 
                         data, history, metrics, today):
    """Generate full HTML report with all charts."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Portfolio Bias & Sector Tilt Report</title>
<style>
body{{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;padding:40px 20px;
     background:#fafafa;color:#222;max-width:1100px;margin:auto;}}
h1{{color:#0d47a1;font-size:28px;margin-bottom:8px;}}
.meta{{font-size:0.95em;color:#666;margin-bottom:24px;}}
.bias{{font-size:1.4em;font-weight:bold;color:#0d47a1;margin-bottom:20px;
       border-bottom:3px solid #0d47a1;padding-bottom:12px;}}
.score{{display:inline-block;font-size:2em;font-weight:bold;color:#fff;
        background:#0d47a1;padding:8px 16px;border-radius:6px;margin-right:20px;}}
.score-label{{display:inline-block;font-size:0.9em;color:#666;vertical-align:middle;}}
.phase{{display:inline-block;font-size:1.1em;color:#0d47a1;font-weight:600;
        background:#e3f2fd;padding:6px 12px;border-radius:4px;}}
h2{{font-size:20px;border-bottom:3px solid #ddd;padding-bottom:10px;margin-top:40px;}}
h2.tailwinds{{border-color:#4caf50;}}
h2.headwinds{{border-color:#f44336;}}
h2.neutrals{{border-color:#9e9e9e;}}
ul{{list-style:none;padding:0;margin:0;}} 
li{{margin-bottom:8px;}}
details>summary{{font-weight:600;cursor:pointer;padding:11px 15px;
                 background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;
                 list-style:none;}}
details>summary:hover{{background:#efefef;}}
details[open]>summary{{border-bottom:none;border-radius:6px 6px 0 0;}}
</style></head><body>
<h1>📊 Portfolio Bias &amp; Sector Tilt Report</h1>
<p class="meta">Generated: {today.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
<div class="bias">
  <div style="margin-bottom:12px;">
    <span class="score">{score}</span>
    <span class="score-label">GDP Growth Score (0–150)</span>
  </div>
  <div>Recommended Bias: <strong>{bias}</strong></div>
  <div style="margin-top:8px;">Cycle Phase: <span class="phase">{phase_label}</span></div>
</div>

<h2 class="tailwinds">✅ Tailwinds ({len(tailwinds)} indicators)</h2>
<ul>{build_html_section(tailwinds, data, history, metrics, today)}</ul>

<h2 class="headwinds">❌ Headwinds ({len(headwinds)} indicators)</h2>
<ul>{build_html_section(headwinds, data, history, metrics, today)}</ul>

<h2 class="neutrals">⚖️ Neutrals ({len(neutrals)} indicators)</h2>
<ul>{build_html_section(neutrals, data, history, metrics, today)}</ul>

</body></html>"""


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Portfolio Bias & Sector Tilt (Optimized)", layout="wide")
st.title("📊 Portfolio Bias & Sector Tilt Dashboard (Optimized 10Y Backtest)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    portfolio_size = st.number_input("Portfolio Size ($)", min_value=10000, value=100000, step=10000)
    preferred_sectors = st.multiselect(
        "Preferred Sectors (optional)",
        list(ALL_SECTORS.keys()),
        default=[])

# Main tabs
tab1, tab2, tab3 = st.tabs(["📈 Live Analysis", "🎯 Sector Tilt", "📊 10Y Backtest"])

with tab1:
    if st.button("🔄 Run Live Analysis", type="primary", key="analysis_btn"):
        with st.spinner("Fetching latest macro data (this may take 1-2 min)..."):
            try:
                data, history, today = fetch_data()
                metrics, tailwinds, headwinds, neutrals, bias, score = calculate_metrics(
                    data, history, today)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("GDP Growth Score", f"{score}/150")
                with col2:
                    st.metric("Recommended Bias", bias.split('—')[0].strip())
                with col3:
                    st.metric("Conviction", f"{metrics['conviction']:.0%}")
                with col4:
                    st.metric("Cycle Phase", metrics['phase_label'].split()[1] if len(metrics['phase_label'].split()) > 1 else metrics['phase_label'])

                st.divider()

                # Key metrics cards
                st.subheader("🔑 Key Forward-Looking Indicators")
                k1, k2, k3, k4, k5 = st.columns(5)
                with k1:
                    yc = metrics['yield_curve_10_2']
                    color = "🟢" if yc > 0 else "🔴"
                    st.metric(f"{color} 10Y-2Y Spread", f"{yc:.2f}%")
                with k2:
                    rr = metrics['real_rate_10yr']
                    color = "🟢" if rr < 0 else "🔴"
                    st.metric(f"{color} Real Rate 10Y", f"{rr:.2f}%")
                with k3:
                    eg = data['earnings_growth']
                    color = "🟢" if eg > 0 else "🔴"
                    st.metric(f"{color} Earnings Growth", f"{eg:.1f}%")
                with k4:
                    fbs = data['fed_bs_growth']
                    color = "🟢" if fbs > 0 else "🔴"
                    st.metric(f"{color} Fed BS Growth", f"{fbs:.1f}%")
                with k5:
                    sp96 = metrics['sp_96_return']
                    color = "🟢" if sp96 > 0 else "🔴"
                    st.metric(f"{color} S&P 9-6M", f"{sp96:.2f}%")

                st.divider()

                # Indicators by category
                st.subheader("✅ Tailwinds")
                for tw in tailwinds[:7]:
                    st.write(f"• {tw}")
                if len(tailwinds) > 7:
                    with st.expander(f"... and {len(tailwinds)-7} more"):
                        for tw in tailwinds[7:]:
                            st.write(f"• {tw}")

                st.subheader("❌ Headwinds")
                for hw in headwinds[:7]:
                    st.write(f"• {hw}")
                if len(headwinds) > 7:
                    with st.expander(f"... and {len(headwinds)-7} more"):
                        for hw in headwinds[7:]:
                            st.write(f"• {hw}")

                st.subheader("⚖️ Neutrals")
                for n in neutrals[:5]:
                    st.write(f"• {n}")

                st.divider()

                # Download HTML report
                with st.spinner("Generating detailed HTML report with all charts..."):
                    html_report = generate_html_summary(
                        tailwinds, headwinds, neutrals, bias, score, 
                        metrics['phase_label'], data, history, metrics, today)
                    st.download_button(
                        label="📥 Download Full HTML Report",
                        data=html_report,
                        file_name=f"macro_bias_report_{today.date()}.html",
                        mime="text/html",
                        key="download_html")

                # Store in session state for other tabs
                st.session_state.data     = data
                st.session_state.history  = history
                st.session_state.today    = today
                st.session_state.metrics  = metrics
                st.session_state.bias     = bias
                st.session_state.score    = score

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.exception(e)

with tab2:
    if 'metrics' in st.session_state:
        metrics = st.session_state.metrics
        bias    = st.session_state.bias
        score   = st.session_state.score

        st.subheader("🎯 Sector Pair Trading Recommendations")
        st.write(f"**Portfolio Size:** ${portfolio_size:,} | **Bias:** {bias} | **Conviction:** {metrics['conviction']:.0%}")

        with st.spinner("Fetching sector data..."):
            tilt_df, alloc_info = generate_sector_tilt(
                bias, score, st.session_state.metrics['phase'], 
                st.session_state.metrics['conviction'], preferred_sectors, portfolio_size)

            if not tilt_df.empty:
                st.dataframe(tilt_df, use_container_width=True)
                st.download_button(
                    label="📥 Download Sector Tilt CSV",
                    data=tilt_df.to_csv(index=False),
                    file_name=f"sector_tilt_{st.session_state.today.date()}.csv",
                    mime="text/csv",
                    key="download_csv")
            else:
                st.warning("⚠️ Unable to generate sector tilt recommendations.")

    else:
        st.info("👈 Run the **Live Analysis** tab first to generate sector tilt recommendations.")

with tab3:
    if st.button("▶️ Run 10-Year Quarterly Backtest", type="primary", key="backtest_btn"):
        with st.spinner("Running 10Y quarterly backtest (Q1 2014 – Q4 2023)..."):
            bt_df, bt_summary = run_backtest()

            if not bt_df.empty:
                st.success("✅ Backtest Complete!")

                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Strategy Sharpe", bt_summary.get('Strategy Sharpe', 'N/A'))
                with col2:
                    st.metric("S&P 500 Sharpe", bt_summary.get('S&P 500 Sharpe', 'N/A'))
                with col3:
                    st.metric("Direction Accuracy", bt_summary.get('Direction Accuracy', 'N/A'))
                with col4:
                    st.metric("Hit Rate", bt_summary.get('Quarterly Hit Rate', 'N/A'))
                with col5:
                    st.metric("Total Quarters", bt_summary.get('Total Quarters', 'N/A'))

                st.divider()

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Strategy Performance")
                    for key in ['Strategy Total Return', 'Strategy Ann. Return', 'Strategy Max Drawdown']:
                        st.write(f"**{key}:** {bt_summary.get(key, 'N/A')}")
                with col_b:
                    st.subheader("S&P 500 Benchmark")
                    for key in ['S&P 500 Total Return', 'S&P 500 Ann. Return', 'S&P 500 Max Drawdown']:
                        st.write(f"**{key}:** {bt_summary.get(key, 'N/A')}")

                st.divider()

                # Equity curves
                st.subheader("Cumulative Returns: Strategy vs. S&P 500")
                fig_curves = plt.figure(figsize=(12, 6))
                plt.plot(bt_df.index, bt_df['SP500_Cum'], label='S&P 500 (Buy & Hold)', 
                        linewidth=2.5, color='#1565C0')
                plt.plot(bt_df.index, bt_df['Strat_Cum'], label='Macro Bias Strategy',  
                        linewidth=2.5, color='#F57C00')
                plt.xlabel('Date', fontsize=11)
                plt.ylabel('Cumulative Return (x)', fontsize=11)
                plt.title('10-Year Quarterly Backtest: Cumulative Returns', fontsize=13, fontweight='bold')
                plt.legend(fontsize=10, loc='upper left')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_curves, use_container_width=True)
                plt.close(fig_curves)

                # Quarterly returns distribution
                st.subheader("Quarterly Returns Distribution")
                fig_hist = plt.figure(figsize=(12, 5))
                plt.hist(bt_df['SP500_Return'], bins=15, label='S&P 500', alpha=0.6, color='#1565C0')
                plt.hist(bt_df['Strat_Return'], bins=15, label='Strategy', alpha=0.6, color='#F57C00')
                plt.xlabel('Quarterly Return (%)', fontsize=11)
                plt.ylabel('Frequency', fontsize=11)
                plt.title('Distribution of Quarterly Returns', fontsize=13, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig_hist, use_container_width=True)
                plt.close(fig_hist)

                # Results table
                st.subheader("Detailed Quarterly Results")
                st.dataframe(
                    bt_df.reset_index().rename(columns={'Date': 'Quarter'})[
                        ['Quarter', 'Score', 'Bias', 'Phase', 'SP500_Return', 'Strat_Return']],
                    use_container_width=True, height=400)

                # Download backtest results
                st.download_button(
                    label="📥 Download Backtest Results CSV",
                    data=bt_df.reset_index().to_csv(index=False),
                    file_name=f"backtest_results_{datetime.now().date()}.csv",
                    mime="text/csv",
                    key="download_backtest")

            else:
                st.error("❌ Backtest failed. Check FRED/yfinance connectivity.")

st.markdown("---")
st.caption(
    "**Optimized Macro Bias & Sector Tilt Dashboard** | "
    "10Y Quarterly Backtest (Q1 2014–Q4 2023) | "
    "Weights optimized via Sharpe ratio maximization | "
    "Forward-looking signals (5Y breakeven inflation, Fed balance sheet, earnings growth) | "
    "Cycle-aware sector rotation with valuation filter | "
    "Pair trading for beta-neutral exposure"
)




