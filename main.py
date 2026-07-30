"""
A-Insight Pro V3.0 — Unified Entry Point
Usage:
    python main.py --pipeline   Run full daily pipeline
    python main.py --feature    Run feature engineering
    python main.py --factor     Run factor mining IC analysis
    python main.py --simulate   Run simulate review
    python main.py --weight     Run weight learning
    python main.py --dashboard  Launch Streamlit dashboard
    python main.py --backtest   Run backtest simulation (walk-forward OOS)
    python main.py --train-rf   Train RF model (walk-forward CV)
    python main.py --train-lstm Train LSTM model
    python main.py --predict    Run ensemble prediction (RF+LSTM)
    python main.py --score      Run calibrated scoring
    python main.py --report     Generate daily report
    python main.py --emotion    Calculate market emotion + smart money
    python main.py --regime     Detect market regime
    python main.py --learning   Run self-learning (per-stock)
    python main.py --review     Run historical accuracy review
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import settings


def cmd_train_rf():
    from ai.train import train; train()

def cmd_train_lstm():
    try:
        from models.train_lstm import train_lstm; train_lstm()
    except ImportError:
        print("LSTM training requires PyTorch. Install with: pip install torch")

def cmd_predict():
    from ai.predict import predict; predict()

def cmd_score():
    from ai.scoring import scoring; scoring()

def cmd_backtest():
    from ai.backtest import backtest; backtest()

def cmd_report():
    from ai.daily_report import generate_report; generate_report()

def cmd_emotion():
    from ai.emotion import compute_emotion; compute_emotion()

def cmd_regime():
    from ai.regime import print_regime; print_regime()

def cmd_learning():
    from ai.self_learning import learning; learning()

def cmd_feature():
    from data_center.feature_engine import run; run()

def cmd_factor():
    from ai.factor_engine import run; run()

def cmd_simulate():
    from ai.simulate_review import run; run()

def cmd_weight():
    from ai.weight_learning import learn_weight; learn_weight()

def cmd_review():
    from ai.historical_review import run; run()

def cmd_factor_engine():
    from ai.factor_engine import run; run()

def cmd_event_factors():
    from ai.event_factors import list_registered_factors, auto_discover_new_factors
    print("=" * 50)
    print("Event & Alternative Factor Discovery")
    print("=" * 50)
    list_registered_factors()
    codes = ["000001", "600519", "000858", "002594", "300750",
             "000670", "001211", "000815", "000050", "000762"]
    print(f"\nTesting auto-discovery on {len(codes)} stocks...")
    auto_discover_new_factors(codes)

def cmd_dashboard():
    os.system('streamlit run dashboard/app.py')

def cmd_pipeline():
    print(f'\n{"="*50}')
    print(f'{settings.PROJECT_NAME} V{settings.VERSION} Daily Pipeline')
    print(f'{"="*50}\n')
    steps = [
        ('Update Data',      lambda: exec(open('ai/update_data.py', encoding='utf-8').read())),
        ('Feature Engine',   cmd_feature),
        ('Market Emotion',   cmd_emotion),
        ('Market Regime',    cmd_regime),
        ('Factor Engine',    cmd_factor_engine),
        ('Event Factors',    cmd_event_factors),
        ('Historical Review', cmd_review),
        ('Self-Learning',    cmd_learning),
        ('Weight Learning',  cmd_weight),
        ('AI Predict',       cmd_predict),
        ('AI Scoring',       cmd_score),
        ('Daily Report',     cmd_report),
    ]
    for name, fn in steps:
        print(f'[{name}]...')
        try: fn()
        except Exception as e: print(f'  WARNING: {e}')
    print(f'\n{"="*50}\nPipeline complete!\n{"="*50}')

def main():
    p = argparse.ArgumentParser(description='A-Insight Pro V3.0')
    p.add_argument('--train-rf', action='store_true', help='Train RandomForest (walk-forward CV)')
    p.add_argument('--train-lstm', action='store_true', help='Train LSTM model')
    p.add_argument('--train', action='store_true', help='Train both RF + LSTM')
    p.add_argument('--predict', action='store_true', help='Run ensemble prediction')
    p.add_argument('--score', action='store_true', help='Run calibrated scoring')
    p.add_argument('--backtest', action='store_true', help='Run walk-forward OOS backtest')
    p.add_argument('--report', action='store_true', help='Generate daily report')
    p.add_argument('--emotion', action='store_true', help='Market emotion + smart money')
    p.add_argument('--regime', action='store_true', help='Detect market regime')
    p.add_argument('--learning', action='store_true', help='Self-learning feedback')
    p.add_argument('--feature', action='store_true', help='Feature engineering')
    p.add_argument('--factor', action='store_true', help='Factor engine (IC analysis)')
    p.add_argument('--simulate', action='store_true', help='Simulate review')
    p.add_argument('--weight', action='store_true', help='Weight learning')
    p.add_argument('--review', action='store_true', help='Historical accuracy review')
    p.add_argument('--factor-engine', action='store_true', help='Advanced factor mining engine')
    p.add_argument('--event-factors', action='store_true', help='Event & alternative factor discovery')
    p.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    p.add_argument('--pipeline', action='store_true', help='Run full daily pipeline')
    args = p.parse_args()

    if not any(vars(args).values()):
        p.print_help()
        print(f'\n{settings.PROJECT_NAME} V{settings.VERSION} | {settings.DATA_SOURCE}')
        return

    if args.train_rf: cmd_train_rf()
    if args.train_lstm: cmd_train_lstm()
    if args.train: cmd_train_rf(); cmd_train_lstm()
    if args.predict: cmd_predict()
    if args.score: cmd_score()
    if args.backtest: cmd_backtest()
    if args.report: cmd_report()
    if args.emotion: cmd_emotion()
    if args.regime: cmd_regime()
    if args.learning: cmd_learning()
    if args.feature: cmd_feature()
    if args.factor: cmd_factor()
    if args.simulate: cmd_simulate()
    if args.weight: cmd_weight()
    if args.review: cmd_review()
    if args.factor_engine: cmd_factor_engine()
    if args.event_factors: cmd_event_factors()
    if args.dashboard: cmd_dashboard()
    if args.pipeline: cmd_pipeline()

if __name__ == '__main__':
    main()
