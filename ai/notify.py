"""
A-Insight Pro
Daily Push Notification V1.1

Channels:
- Server酱 (WeChat push) — default, no email needed
- SMTP email — fallback for QQ/163/Gmail
"""

import os, json, requests
from datetime import datetime
import pandas as pd

CONFIG_FILE = "config/notify_config.json"
RANK_FILE = "reports/final_stock_rank.csv"
EMOTION_FILE = "reports/market_emotion.csv"
SUMMARY_FILE = "reports/historical_review_summary.csv"


def load_config():
    cfg = {
        "server_chan_key": os.environ.get("SERVER_CHAN_KEY", ""),
        "smtp_host": os.environ.get("SMTP_HOST", ""),
        "smtp_port": int(os.environ.get("SMTP_PORT", "465")),
        "smtp_user": os.environ.get("SMTP_USER", ""),
        "smtp_pass": os.environ.get("SMTP_PASS", ""),
        "to_email": os.environ.get("TO_EMAIL", ""),
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def build_content():
    """Build text content for push notification."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    # Market emotion
    mkt_score, mkt_level = 50, "neutral"
    smart_signal, smart_activity = "", ""
    position_pct = 20
    if os.path.exists(EMOTION_FILE):
        try:
            em = pd.read_csv(EMOTION_FILE, encoding="utf-8-sig")
            if not em.empty:
                last = em.iloc[-1]
                mkt_score = float(last.get("market_emotion", 50))
                mkt_level = str(last.get("level", "neutral"))
                smart_signal = str(last.get("smart_money_signal", ""))
                smart_activity = str(last.get("smart_money_activity", ""))
                position_pct = int(last.get("suggested_position_pct", 20))
                heat = float(last.get("heat_score", 50))
                smart = float(last.get("smart_money_score", 50))
        except Exception:
            heat, smart = 50, 50
    else:
        heat, smart = 50, 50

    # Model health
    bias_str, acc_str = "", ""
    if os.path.exists(SUMMARY_FILE):
        try:
            s = pd.read_csv(SUMMARY_FILE)
            if not s.empty:
                bias = float(s["bias_pct"].iloc[0])
                acc = float(s["directional_accuracy_pct"].iloc[0])
                bias_str = f"{bias:+.2f}%"
                acc_str = f"{acc:.1f}%"
        except Exception:
            pass

    # Top picks
    buy_n, watch_n, a_rated = 0, 0, 0
    top_lines = []
    if os.path.exists(RANK_FILE):
        try:
            rank = pd.read_csv(RANK_FILE, encoding="utf-8-sig")
            top = rank.head(10)
            buy_n = rank["SIGNAL"].str.contains("STRONG|BUY|强烈|重点", na=False).sum()
            watch_n = rank["SIGNAL"].str.contains("WATCH|观察", na=False).sum()
            a_rated = rank["LEVEL"].isin(["A+", "A"]).sum()

            for _, r in top.iterrows():
                code = str(r.get("code", ""))
                name = str(r.get("name", "")) if pd.notna(r.get("name", "")) else ""
                pred = float(r.get("predict_percent", 0))
                score = float(r.get("AI_SCORE", 0))
                sig = str(r.get("SIGNAL", ""))
                lvl = str(r.get("LEVEL", ""))
                top_lines.append(
                    f"  #{int(r.get('rank',0))} {code} {name}  {pred:+.1f}%  {score:.0f}分  {sig}[{lvl}]"
                )
        except Exception:
            pass

    # Build message
    emoji_map = {
        "主力吸筹": "🔴", "主力出货": "🟢", "主力观望": "🟡",
        "主力休息": "⚪", "分歧": "🟠",
    }
    s_emoji = emoji_map.get(smart_signal, "")

    text = f"""A-Insight Pro 日报 {date_str}

📊 市场情绪: {mkt_score:.0f} ({mkt_level})
💰 主力动向: {s_emoji} {smart_signal} [{smart_activity}]
📈 建议仓位: {position_pct}%
  热度: {heat:.0f} | 主力活跃度: {smart:.0f}

🎯 信号统计: {buy_n}买入 | {watch_n}观察 | {a_rated}A级

🏆 Top 10:
{chr(10).join(top_lines)}

📐 模型偏差: {bias_str} | 方向准确率: {acc_str}
"""
    return text


def push_server_chan(text, key):
    """Push via Server酱 (WeChat)."""
    url = f"https://sctapi.ftqq.com/{key}.send"
    title = f"A-Insight Daily {datetime.now().strftime('%Y-%m-%d')}"
    try:
        resp = requests.post(url, data={"title": title, "desp": text}, timeout=15)
        result = resp.json()
        if result.get("code") == 0:
            print(f"WeChat push OK (Server酱)")
            return True
        else:
            print(f"Server酱 error: {result.get('message', resp.text)}")
            return False
    except Exception as e:
        print(f"Server酱 failed: {e}")
        return False


def push():
    """Main entry — tries WeChat push first, falls back to email."""
    print("=" * 40)
    print("A-Insight Daily Push")
    print("=" * 40)

    cfg = load_config()
    text = build_content()

    print(f"Config: server_chan_key={'***' if cfg.get('server_chan_key') else 'MISSING'}, smtp_host={cfg.get('smtp_host') or 'MISSING'}, to_email={cfg.get('to_email') or 'MISSING'}")

    print("Sending push...")
    key = cfg.get("server_chan_key", "")
    if key:
        ok = push_server_chan(text, key)
        if ok:
            print("=" * 40)
            return

    # Channel 2: SMTP email (fallback)
    if cfg.get("smtp_host"):
        print("Trying email fallback...")
        _send_email(text, cfg)
    else:
        print("No push channel configured!")

    print("=" * 40)


def _send_email(text, cfg):
    """SMTP email fallback."""
    import smtplib, ssl
    from email.mime.text import MIMEText

    msg = MIMEText(text, "plain", "utf-8")
    msg["Subject"] = f"A-Insight Daily {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = cfg["smtp_user"]
    msg["To"] = cfg["to_email"]

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], context=ctx, timeout=10) as s:
            s.login(cfg["smtp_user"], cfg["smtp_pass"])
            s.sendmail(cfg["smtp_user"], [cfg["to_email"]], msg.as_string())
        print(f"Email sent to {cfg['to_email']}")
    except Exception as e:
        print(f"Email failed: {e}")


if __name__ == "__main__":
    push()
