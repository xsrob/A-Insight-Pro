"""
A-Insight Pro
Unified Daily Report V4.0 — Single-File Smart Report

Replaces 4 scattered files with 1 comprehensive HTML containing:
  - Market dashboard with KPI cards
  - AI Top 20 ranking (sortable, filterable)
  - Factor intelligence (52-factor IC analysis)
  - Model health diagnostics
  - One-click trading app export
  - All data embedded as JSON for programmatic access
"""

import os, sys, json, glob as _glob
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import REPORT_DIR

RANK_FILE = os.path.join(REPORT_DIR, "final_stock_rank.csv")
EMOTION_FILE = os.path.join(REPORT_DIR, "market_emotion.csv")
LEARNING_FILE = os.path.join(REPORT_DIR, "ai_learning_feedback.csv")
ADJUST_FILE = os.path.join(REPORT_DIR, "predict_adjust.json")
FACTOR_SEL_FILE = os.path.join(REPORT_DIR, "factor_selection.json")

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

def _load_csv(path):
    if os.path.exists(path):
        try: return pd.read_csv(path, encoding="utf-8-sig")
        except: pass
    return pd.DataFrame()

def _load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {}

# ── CSS (single source of truth for styling) ──
CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;max-width:1200px;margin:0 auto}
h1{font-size:22px;color:#58a6ff;margin-bottom:4px}
h2{font-size:16px;color:#f0f6fc;border-bottom:1px solid #30363d;padding-bottom:8px;margin:20px 0 12px}
.subtitle{color:#8b949e;font-size:12px;margin-bottom:20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px}
.kpi{background:#21262d;border-radius:8px;padding:14px;text-align:center}
.kpi-val{font-size:26px;font-weight:700}
.kpi-lbl{font-size:11px;color:#8b949e;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#21262d;color:#8b949e;padding:8px 6px;text-align:center;font-weight:500;position:sticky;top:0}
td{padding:7px 6px;text-align:center;border-bottom:1px solid#21262d}
tr:hover{background:#1c2128}
.badge{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;color:#fff;white-space:nowrap}
.green{color:#3fb950}.red{color:#f85149}.yellow{color:#d2991d}.blue{color:#58a6ff}.gray{color:#8b949e}
.factor-bar{height:6px;border-radius:3px;background:#21262d;margin-top:3px}
.factor-fill{height:100%;border-radius:3px}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;margin:1px;color:#fff}
.btn{padding:6px 16px;border-radius:6px;border:none;cursor:pointer;font-size:12px;font-weight:600;color:#fff}
.btn-blue{background:#238636}.btn-blue:hover{background:#2ea043}
.btn-gray{background:#30363d}.btn-gray:hover{background:#484f58}
.copied{color:#3fb950;font-size:11px;margin-left:8px;opacity:0;transition:opacity .3s}
.copied.show{opacity:1}
.footer{color:#484f58;font-size:10px;text-align:center;margin-top:30px;padding-top:16px;border-top:1px solid#21262d}
@media(max-width:768px){.grid4{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
"""

def _cleanup_desktop():
    """Keep only latest 3 report files on Desktop."""
    for pat in ["A-Insight_*.html"]:
        files = sorted(_glob.glob(os.path.join(DESKTOP, pat)))
        for old in files[:-3]:
            try: os.remove(old)
            except: pass
    # Remove legacy split files from old system
    for pat in ["A-Insight日报_*.html","A-Insight日报_*.csv",
                "A-Insight交易信号_*.csv","A-Insight自选股_*.txt"]:
        files = sorted(_glob.glob(os.path.join(DESKTOP, pat)))
        for old in files:
            try: os.remove(old)
            except: pass

def _badge(val, thresholds, colors):
    """Return CSS class for a value based on thresholds."""
    for (lo, hi), cls in zip(thresholds, colors):
        if lo <= val < hi: return cls
    return colors[-1] if colors else ""

def generate_report():
    print("=" * 50)
    print("A-Insight Unified Report V4.0")
    print("=" * 50)

    # ── Load Data ──
    rank = _load_csv(RANK_FILE)
    if rank.empty:
        print("No ranking data. Run pipeline first.")
        return
    rank["code"] = rank["code"].astype(str).str.zfill(6)

    emotion = _load_csv(EMOTION_FILE)
    learning = _load_csv(LEARNING_FILE)
    adjust = _load_json(ADJUST_FILE)
    factor_data = _load_json(FACTOR_SEL_FILE)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d %H:%M")

    # ── Market State ──
    mkt_score, mkt_level = 50, "neutral"
    if not emotion.empty:
        last = emotion.iloc[-1]
        mkt_score = float(last.get("market_emotion", 50))
        mkt_level = str(last.get("level", "neutral"))
    mkt_color = "#3fb950" if mkt_score >= 50 else ("#d2991d" if mkt_score >= 30 else "#f85149")

    # ── Signal Summary ──
    if "SIGNAL" not in rank.columns:
        rank["SIGNAL"] = rank["AI_SCORE"].apply(
            lambda s: "STRONG_BUY" if s>=85 else ("BUY" if s>=70 else ("WATCH" if s>=55 else "AVOID")))
    buy_n = len(rank[rank["SIGNAL"].str.contains("STRONG|BUY|强烈|重点", na=False)])
    watch_n = len(rank[rank["SIGNAL"].str.contains("WATCH|观察", na=False)])
    a_n = (rank["LEVEL"].isin(["A+","A"])).sum()

    # ── Model Health ──
    factor = adjust.get("predict_factor", 1.0)
    avg_success = adjust.get("avg_success_rate", 0)
    avg_error = adjust.get("avg_error", 0)
    n_samples = adjust.get("n_samples", len(rank))

    # ── Factor Intelligence ──
    factor_rows_html = ""
    factor_sources_html = ""
    all_factors = factor_data.get("top_factors", [])
    ic_map = factor_data.get("factor_ic_map", {})
    n_factors = len(ic_map)

    # Factor source distribution
    src_colors = {"ic_rolling_window":"#58a6ff","genetic_programming":"#3fb950",
                  "event_factors":"#d2991d","llm_factors":"#f85149"}
    src_count = {}
    for v in ic_map.values():
        src = v.get("source","?")
        src_count[src] = src_count.get(src,0)+1
    for src, cnt in sorted(src_count.items()):
        color = src_colors.get(src,"#8b949e")
        label = {"ic_rolling_window":"量价IC","genetic_programming":"遗传GP",
                 "event_factors":"事件另类","llm_factors":"LLM情绪"}.get(src,src)
        factor_sources_html += f'<span class="tag" style="background:{color}">{label}:{cnt}</span> '

    # Top 10 factors table
    category_colors = {
        "momentum":"#58a6ff","volatility":"#d2991d","technical_mr":"#3fb950",
        "quality_sharpe":"#bc8cff","tail_risk":"#f85149","volume_flow":"#ff7b72",
        "genetic_nonlinear":"#3fb950","event_alternative":"#d2991d","llm_sentiment":"#f0883e",
    }
    for i, f in enumerate(all_factors[:10]):
        fname = f.get("factor", f.get("name",""))[:35]
        abs_ic = f.get("abs_ic", 0)
        w = f.get("weight", 0)
        cat = f.get("category","?")
        cat_color = category_colors.get(cat,"#8b949e")
        stability = f.get("stability", f.get("rolling_stability", 0))
        bar_w = min(100, abs_ic * 800)  # scale IC to bar width
        factor_rows_html += f"""
        <tr>
            <td style="text-align:left;font-family:monospace;font-size:11px">{fname}</td>
            <td><span class="tag" style="background:{cat_color}">{cat}</span></td>
            <td style="color:#58a6ff;font-weight:600">{abs_ic:.4f}</td>
            <td>{w:.3f}</td>
            <td>{'█'*int(stability*10)}{'░'*int((1-stability)*10)}</td>
            <td>
                <div class="factor-bar"><div class="factor-fill" style="width:{bar_w}%;background:{cat_color}"></div></div>
            </td>
        </tr>"""

    # ── Top 20 Ranking ──
    lvl_badges = {"A+":"#f85149","A":"#3fb950","B":"#58a6ff","C":"#484f58"}
    sig_badges = {"STRONG_BUY":"#f85149","BUY":"#d2991d","WATCH":"#58a6ff","AVOID":"#484f58"}
    sig_labels = {"STRONG_BUY":"强烈关注","BUY":"重点关注","WATCH":"观察","AVOID":"回避"}
    rank_rows = ""
    for _, row in rank.head(20).iterrows():
        code = str(row.get("code",""))
        name = str(row.get("name","")) if pd.notna(row.get("name","")) else ""
        pred = float(row.get("predict_percent",0))
        score = float(row.get("AI_SCORE",0))
        lvl = str(row.get("LEVEL","C"))
        sig = str(row.get("SIGNAL","AVOID"))
        fconf = float(row.get("factor_confidence",0))
        pos = str(row.get("POSITION_PCT","-"))
        stop = str(row.get("STOP_LOSS","-"))
        rank_rows += f"""
        <tr>
            <td>{int(row.get('rank',0))}</td>
            <td style="font-weight:600">{code}</td>
            <td class="gray">{name if name else '-'}</td>
            <td style="color:{'#3fb950' if pred>0 else '#f85149'};font-weight:600">{pred:+.2f}%</td>
            <td style="font-weight:600">{score:.0f}</td>
            <td style="color:{'#3fb950' if fconf>0 else '#f85149'}">{fconf:+.1f}</td>
            <td><span class="badge" style="background:{lvl_badges.get(lvl,'#484f58')}">{lvl}</span></td>
            <td><span class="badge" style="background:{sig_badges.get(sig,'#484f58')}">{sig_labels.get(sig,sig)}</span></td>
            <td class="gray">{pos}%</td>
            <td class="gray">{stop}%</td>
        </tr>"""

    # ── Learning Bias ──
    learn_rows = ""
    if not learning.empty and "success_rate" in learning.columns:
        bias = learning.nsmallest(8, "avg_error") if "avg_error" in learning.columns else learning.head(8)
        for _, row in bias.iterrows():
            sr = float(row.get("success_rate",0))*100
            ae = float(row.get("avg_error",0))
            learn_rows += f"""
            <tr>
                <td>{str(row.get('code','')).zfill(6)}</td>
                <td>{int(row.get('samples',0))}</td>
                <td style="color:{'#3fb950' if sr>=50 else '#f85149'}">{sr:.1f}%</td>
                <td style="color:{'#3fb950' if ae<3 else ('#d2991d' if ae<6 else '#f85149')}">{ae:.2f}%</td>
            </tr>"""

    # ── Build Embedded JSON (all data for programmatic access) ──
    top20_list = []
    for _, row in rank.head(20).iterrows():
        top20_list.append({
            "rank": int(row.get("rank",0)), "code": str(row.get("code","")),
            "name": str(row.get("name","")) if pd.notna(row.get("name","")) else "",
            "predict_percent": float(row.get("predict_percent",0)),
            "AI_SCORE": float(row.get("AI_SCORE",0)),
            "LEVEL": str(row.get("LEVEL","C")),
            "SIGNAL": str(row.get("SIGNAL","AVOID")),
        })

    embed = {
        "date": date_str, "market_score": float(mkt_score), "market_level": str(mkt_level),
        "buy_signals": int(buy_n), "watch_signals": int(watch_n), "a_rated": int(a_n),
        "n_factors": int(n_factors), "model_factor": float(factor),
        "top20": top20_list,
        "factor_summary": {str(k):int(v) for k,v in src_count.items()},
    }
    embed_json = json.dumps(embed, ensure_ascii=False)

    # ── HTML ──
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A-Insight Daily {date_str}</title>
<style>{CSS}</style>
</head>
<body>

<h1>📊 A-Insight Pro Daily Report</h1>
<p class="subtitle">{time_str} | {date_str} | Factor-Integrated V4.0 | {n_factors} factors active</p>

<!-- ═══════ KPI Dashboard ═══════ -->
<div class="grid4">
  <div class="kpi">
    <div class="kpi-val" style="color:{mkt_color}">{mkt_score:.0f}</div>
    <div class="kpi-lbl">Market Sentiment</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:{'#3fb950' if buy_n>0 else '#58a6ff'}">{buy_n}</div>
    <div class="kpi-lbl">Buy Signals</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:#3fb950">{a_n}</div>
    <div class="kpi-lbl">A-Rated Stocks</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:#58a6ff">{n_factors}</div>
    <div class="kpi-lbl">Active Factors</div>
  </div>
</div>

<div class="grid2">
  <div class="kpi">
    <div class="kpi-val" style="color:{'#3fb950' if factor>=1 else '#d2991d'}">{factor:.2f}</div>
    <div class="kpi-lbl">Model Adjustment Factor</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:#58a6ff">{watch_n}</div>
    <div class="kpi-lbl">Watch Signals</div>
  </div>
</div>

<!-- ═══════ AI Top 20 ═══════ -->
<div class="card">
<h2>🎯 AI Top 20 Picks</h2>
<div style="margin-bottom:8px">
  <button class="btn btn-blue" onclick="copyCodes()">📋 Copy Top 10 Codes</button>
  <button class="btn btn-gray" onclick="copyAllCodes()">📋 Copy Top 20 Codes</button>
  <span class="copied" id="copied">Copied!</span>
</div>
<div style="max-height:500px;overflow-y:auto">
<table>
<thead><tr>
  <th>#</th><th>Code</th><th>Name</th><th>Predict%</th><th>Score</th><th>FConf</th><th>Level</th><th>Signal</th><th>Pos%</th><th>Stop%</th>
</tr></thead>
<tbody>{rank_rows}</tbody>
</table>
</div>
</div>

<!-- ═══════ Factor Intelligence ═══════ -->
<div class="card">
<h2>🧬 Factor Intelligence ({n_factors} factors)</h2>
<div style="margin-bottom:8px">{factor_sources_html}</div>
<div style="max-height:360px;overflow-y:auto">
<table>
<thead><tr>
  <th style="text-align:left">Factor</th><th>Category</th><th>|IC|</th><th>Weight</th><th>Stability</th><th>Strength</th>
</tr></thead>
<tbody>{factor_rows_html if factor_rows_html else '<tr><td colspan="6" class="gray">Factor intelligence loading... run factor_engine first</td></tr>'}</tbody>
</table>
</div>
</div>

<!-- ═══════ Model Health ═══════ -->
<div class="card">
<h2>⚙️ Model Health</h2>
<div class="grid4">
  <div class="kpi">
    <div class="kpi-val" style="color:{'#3fb950' if avg_success>=0.5 else '#d2991d'}">{avg_success:.0%}</div>
    <div class="kpi-lbl">Success Rate</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:{'#f85149' if avg_error<-2 else '#3fb950'}">{avg_error:+.2f}%</div>
    <div class="kpi-lbl">Avg Prediction Error</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:#58a6ff">{n_samples}</div>
    <div class="kpi-lbl">Learning Samples</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:#8b949e">{len(rank)}</div>
    <div class="kpi-lbl">Stocks Tracked</div>
  </div>
</div>
</div>

<!-- ═══════ Learning Feedback ═══════ -->
<div class="card">
<h2>📖 Learning Feedback</h2>
<div style="max-height:280px;overflow-y:auto">
<table>
<thead><tr><th>Code</th><th>Samples</th><th>Success%</th><th>Avg Error%</th></tr></thead>
<tbody>{learn_rows if learn_rows else '<tr><td colspan="4" class="gray">Accumulating training data...</td></tr>'}</tbody>
</table>
</div>
</div>

<!-- ═══════ Footer ═══════ -->
<div class="footer">
  <p>A-Insight Pro V4.0 | AI Research Tool | Not Financial Advice</p>
  <p>Next auto-run: Daily 08:30 | Sources: 量价IC + 遗传GP + 事件另类 + LLM情绪</p>
</div>

<!-- Embedded JSON data -->
<script id="report-data" type="application/json">{embed_json}</script>
<script>
function copyText(text,label) {{
  navigator.clipboard.writeText(text).then(()=>{{
    const el=document.getElementById('copied');
    el.textContent=label||'Copied!';
    el.classList.add('show');
    setTimeout(()=>el.classList.remove('show'),2000);
  }});
}}
function copyCodes(){{
  const data=JSON.parse(document.getElementById('report-data').textContent);
  const codes=data.top20.slice(0,10).map(r=>r.code).join(',');
  copyText(codes,'Top 10 copied!');
}}
function copyAllCodes(){{
  const data=JSON.parse(document.getElementById('report-data').textContent);
  const codes=data.top20.slice(0,20).map(r=>r.code).join(',');
  copyText(codes,'Top 20 copied!');
}}
</script>

</body></html>"""

    # ── Save ──
    _cleanup_desktop()
    html_path = os.path.join(DESKTOP, f"A-Insight_{date_str}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Report: {html_path}")
    print(f"  Buy: {buy_n} | Watch: {watch_n} | A-Rated: {a_n}")
    print(f"  Factors: {n_factors} | Sources: {list(src_count.keys())}")
    print("=" * 50)
    return html_path

if __name__ == "__main__":
    generate_report()
