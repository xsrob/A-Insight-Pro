"""
A-Insight Pro
Sector/Industry Feature Builder V1.0

Builds sector-level features for better cross-stock learning:
- Sector membership (one-hot encoded for top sectors)
- Sector momentum (average return of sector peers)
- Sector relative strength (stock vs sector)
- Sector breadth (% of sector stocks above MA20)
- Sector volume flow

These features help the model understand:
- Is this stock moving with or against its sector?
- Is the sector itself trending?
- Is there sector rotation happening?
"""

import os
import numpy as np
import pandas as pd

FEATURE_DIR = "features"
DATA_DIR = "data"

# A-share sector classification (by industry prefix / sector codes)
# Simplified mapping based on CSRC industry classification
SECTOR_MAPPING = {
    # Financials
    "银行": ["001", "002", "600000", "600015", "600016", "600036", "601009",
             "601128", "601166", "601169", "601229", "601288", "601318",
             "601328", "601390", "601398", "601628", "601668", "601688",
             "601818", "601857", "601939", "601988", "601998"],
    "券商": ["600030", "600061", "600109", "600369", "600837", "600958",
             "600999", "601066", "601099", "601108", "601162", "601198",
             "601211", "601236", "601375", "601377", "601456", "601555",
             "601696", "601878", "601881", "601901"],
    "保险": ["601318", "601319", "601336", "601601", "601628"],

    # Technology
    "半导体": ["002049", "002129", "002156", "002185", "002371", "002409",
               "300223", "300327", "300474", "300604", "300623", "600460",
               "600584", "600703", "603005", "603160", "603501", "603893",
               "688008", "688012", "688036", "688256", "688396", "688981"],
    "软件": ["002065", "002153", "002195", "002230", "002268", "002405",
             "002410", "002439", "300033", "300036", "300253", "300339",
             "300352", "300454", "300624", "600536", "600570", "600588",
             "600718", "600756", "603927", "688111", "688561"],
    "硬件": ["000066", "000725", "000977", "002180", "002236", "002241",
             "002415", "002475", "300124", "300408", "300433", "300502",
             "600271", "601138", "603019", "603236"],

    # Consumer
    "白酒": ["000568", "000596", "000799", "000858", "000860", "002304",
             "600519", "600559", "600702", "600779", "600809", "603198",
             "603369", "603589"],
    "食品": ["000639", "000716", "000848", "000895", "002216", "002481",
             "002507", "002557", "002570", "002582", "002695", "002714",
             "300146", "600298", "600305", "600872", "600882", "600887",
             "603027", "603043", "603288", "603317", "603345", "603866"],
    "家电": ["000333", "000418", "000651", "002032", "002050", "002242",
             "002508", "002677", "600690", "603486", "605555"],

    # Healthcare
    "医药": ["000423", "000513", "000538", "000623", "000650", "000661",
             "000739", "000963", "002001", "002007", "002019", "002020",
             "002022", "002030", "002038", "002099", "002107", "002118",
             "002252", "002262", "002294", "002317", "002332", "002399",
             "002411", "002422", "002424", "002433", "002550", "002603",
             "002644", "002653", "002675", "002727", "002737", "002773",
             "300003", "300009", "300015", "300016", "300026", "300122",
             "300142", "300147", "300199", "300294", "300347", "300357",
             "300363", "300406", "300463", "300482", "300529", "300558",
             "300595", "300601", "300630", "300633", "300676", "300677",
             "300702", "300705", "300720", "300725", "300759", "300760",
             "600056", "600062", "600079", "600085", "600129", "600161",
             "600196", "600276", "600329", "600332", "600380", "600436",
             "600479", "600511", "600518", "600521", "600535", "600566",
             "600572", "600587", "600763", "600771", "600976", "603259",
             "603456", "603658", "603882", "688029"],
    "医疗器械": ["002223", "002382", "002432", "002551", "002690", "002901",
                 "300003", "300030", "300049", "300109", "300146", "300171",
                 "300206", "300238", "300244", "300289", "300298", "300314",
                 "300326", "300396", "300401", "300404", "300406", "300412",
                 "300439", "300453", "300463", "300482", "300529", "300562",
                 "300595", "300633", "300639", "300642", "300653", "300676",
                 "300677", "300685", "300702", "300705", "300720", "300725",
                 "300753", "300760", "300832", "688029", "688050", "688068",
                 "688105", "688114", "688139", "688161", "688185", "688198",
                 "688202", "688212", "688236", "688276", "688298", "688301",
                 "688310", "688314", "688317", "688321"],

    # Energy & Materials
    "新能源": ["000009", "000012", "000027", "000040", "000049", "000591",
               "002056", "002074", "002129", "002202", "002218", "002245",
               "002249", "002256", "002309", "002340", "002460", "002466",
               "002497", "002594", "002709", "002812", "300014", "300037",
               "300068", "300073", "300088", "300118", "300124", "300207",
               "300274", "300316", "300376", "300390", "300438", "300450",
               "300457", "300496", "300568", "300618", "300750", "300763",
               "600110", "600152", "600438", "600478", "600580", "600703",
               "600732", "600884", "600885", "601012", "601615", "601865",
               "603026", "603185", "603259", "603396", "603659", "603799",
               "688005", "688012", "688036", "688116", "688388", "688390",
               "688599"],
    "有色": ["000060", "000426", "000603", "000612", "000630", "000688",
             "000758", "000762", "000807", "000831", "000878", "000933",
             "000960", "000962", "000969", "000970", "000975", "002056",
             "002149", "002155", "002167", "002182", "002203", "002237",
             "002378", "002460", "002466", "002540", "002738", "002756",
             "300034", "300224", "300337", "300395", "300428", "300618",
             "600111", "600219", "600259", "600338", "600362", "600392",
             "600456", "600459", "600489", "600497", "600516", "600531",
             "600547", "600549", "600673", "600711", "600888", "600988",
             "601020", "601069", "601168", "601212", "601600", "601677",
             "601899", "601958", "603260", "603399", "603663", "603799",
             "603876", "603937", "603993", "605376"],

    # Real Estate & Infrastructure
    "地产": ["000002", "000006", "000011", "000014", "000031", "000036",
             "000042", "000046", "000069", "000402", "000517", "000540",
             "000560", "000620", "000631", "000656", "000667", "000671",
             "000718", "000732", "000736", "000797", "000838", "000863",
             "000886", "000897", "000918", "000926", "000961", "000965",
             "001979", "002146", "002208", "002244", "002285", "002305",
             "600048", "600064", "600067", "600094", "600153", "600162",
             "600173", "600185", "600208", "600223", "600239", "600246",
             "600266", "600325", "600340", "600376", "600383", "600393",
             "600466", "600503", "600510", "600533", "600565", "600604",
             "600606", "600622", "600638", "600641", "600649", "600657",
             "600658", "600663", "600665", "600675", "600683", "600684",
             "600692", "600708", "600716", "600730", "600736", "600743",
             "600748", "600773", "600791", "600807", "600823", "600848",
             "600895", "601155", "601588"],

    # Transportation & Logistics
    "交运": ["000088", "000089", "000099", "000429", "000507", "000520",
             "000548", "000582", "000755", "000828", "000885", "000900",
             "002120", "002183", "002210", "002245", "002320", "002352",
             "002468", "002492", "002627", "002682", "002800", "002928",
             "300240", "300350", "600004", "600009", "600012", "600017",
             "600018", "600020", "600021", "600026", "600029", "600033",
             "600035", "600050", "600106", "600115", "600119", "600125",
             "600153", "600180", "600190", "600233", "600269", "600270",
             "600279", "600317", "600350", "600368", "600377", "600428",
             "600548", "600561", "600575", "600611", "600650", "600662",
             "600676", "600717", "600751", "600787", "600794", "600798",
             "600834", "600897", "601000", "601006", "601008", "601018",
             "601021", "601107", "601111", "601333", "601518", "601866",
             "601872", "601880", "601919", "603056", "603128", "603167",
             "603329", "603535", "603569", "603648", "603713", "603871",
             "603885"],

    # Military / Defense
    "军工": ["000519", "000547", "000561", "000576", "000625", "000638",
             "000697", "000733", "000738", "000768", "000801", "000901",
             "002013", "002023", "002025", "002046", "002049", "002111",
             "002151", "002179", "002190", "002214", "002231", "002246",
             "002254", "002265", "002297", "002298", "002300", "002302",
             "002338", "002361", "002389", "002413", "002414", "002423",
             "002465", "002519", "002544", "002651", "300008", "300024",
             "300034", "300045", "300065", "300101", "300114", "300123",
             "300159", "300177", "300185", "300200", "300252", "300324",
             "300342", "300354", "300395", "300397", "300424", "300447",
             "300474", "300489", "300527", "300581", "300589", "300593",
             "300629", "300696", "300719", "300722", "300726", "300733",
             "300762", "300775", "300777", "300855", "600038", "600072",
             "600118", "600150", "600151", "600184", "600316", "600343",
             "600372", "600391", "600435", "600480", "600482", "600523",
             "600562", "600590", "600592", "600685", "600705", "600760",
             "600764", "600765", "600855", "600862", "600879", "600893",
             "600967", "600990", "601606", "601698", "601890", "601989",
             "603129", "603261", "603267", "603678", "605123"],
}

# Build reverse lookup: code -> sector
CODE_TO_SECTOR = {}
for sector, codes in SECTOR_MAPPING.items():
    for code in codes:
        if code not in CODE_TO_SECTOR:
            CODE_TO_SECTOR[code] = sector


def get_sector(code):
    """Get sector name for a stock code."""
    return CODE_TO_SECTOR.get(str(code).zfill(6), "其他")


def compute_sector_features():
    """
    Compute sector-level features for all stocks in feature/ directory.

    For each stock, adds:
    - sector:             sector name
    - sector_momentum_5d: median 5d return of sector peers
    - sector_momentum_20d: median 20d return of sector peers
    - sector_breadth:     % of sector stocks above MA20
    - sector_relative:    stock's return vs sector median (relative strength)
    - sector_vol_flow:    sector aggregate volume vs 20d average
    """
    if not os.path.exists(FEATURE_DIR):
        print("Feature directory not found")
        return

    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    print(f"Computing sector features for {len(files)} stocks...")

    # First pass: collect per-stock metrics
    stock_metrics = {}  # {code: {return_5d, return_20d, above_ma20, vol_ratio}}
    for fname in files:
        code = fname.replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, fname), encoding="utf-8-sig")
            if len(df) < 60:
                continue

            latest = df.iloc[-1]
            metrics = {}

            if "close" in df.columns:
                if len(df) >= 5:
                    metrics["return_5d"] = df["close"].iloc[-1] / df["close"].iloc[-5] - 1
                if len(df) >= 20:
                    metrics["return_20d"] = df["close"].iloc[-1] / df["close"].iloc[-20] - 1

            if "ma20" in df.columns and "close" in df.columns:
                metrics["above_ma20"] = float(latest.get("close", 0)) > float(latest.get("ma20", 0))

            if "volume" in df.columns and len(df) >= 20:
                avg_vol = df["volume"].tail(20).mean()
                if avg_vol > 0:
                    metrics["vol_ratio"] = float(latest.get("volume", 0)) / avg_vol

            if metrics:
                stock_metrics[code] = metrics

        except Exception:
            continue

    # Second pass: compute sector aggregates
    sector_agg = {}  # {sector: {median_return_5d, median_return_20d, breadth, avg_vol_ratio}}
    for sector in set(CODE_TO_SECTOR.values()):
        sector_codes = [c for c, s in CODE_TO_SECTOR.items() if s == sector and c in stock_metrics]
        if len(sector_codes) < 3:
            continue

        returns_5d = [stock_metrics[c].get("return_5d", 0) for c in sector_codes if "return_5d" in stock_metrics[c]]
        returns_20d = [stock_metrics[c].get("return_20d", 0) for c in sector_codes if "return_20d" in stock_metrics[c]]
        breadths = [stock_metrics[c].get("above_ma20", False) for c in sector_codes if "above_ma20" in stock_metrics[c]]
        vol_ratios = [stock_metrics[c].get("vol_ratio", 1.0) for c in sector_codes if "vol_ratio" in stock_metrics[c]]

        sector_agg[sector] = {
            "n_stocks": len(sector_codes),
            "median_return_5d": np.median(returns_5d) if returns_5d else 0,
            "median_return_20d": np.median(returns_20d) if returns_20d else 0,
            "breadth": np.mean(breadths) if breadths else 0.5,
            "avg_vol_ratio": np.median(vol_ratios) if vol_ratios else 1.0,
        }

    # Third pass: write sector features to each stock's feature file
    updated = 0
    for fname in files:
        code = fname.replace(".csv", "").zfill(6)
        sector = get_sector(code)
        agg = sector_agg.get(sector)

        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, fname), encoding="utf-8-sig")

            # Sector membership (one-hot encoded as single string)
            df["sector"] = sector

            if agg:
                df["sector_momentum_5d"] = agg["median_return_5d"]
                df["sector_momentum_20d"] = agg["median_return_20d"]
                df["sector_breadth"] = agg["breadth"]
                df["sector_vol_flow"] = agg["avg_vol_ratio"]

                # Relative strength: stock vs sector
                if code in stock_metrics:
                    stock_ret_20d = stock_metrics[code].get("return_20d", 0)
                    sector_ret_20d = agg["median_return_20d"]
                    df["sector_relative_strength"] = stock_ret_20d - sector_ret_20d
            else:
                df["sector_momentum_5d"] = 0
                df["sector_momentum_20d"] = 0
                df["sector_breadth"] = 0.5
                df["sector_vol_flow"] = 1.0
                df["sector_relative_strength"] = 0

            df.to_csv(os.path.join(FEATURE_DIR, fname), index=False, encoding="utf-8-sig")
            updated += 1

        except Exception:
            continue

    print(f"  Updated {updated} stocks with sector features")
    print(f"  Sectors found: {len(sector_agg)}")

    # Print sector summary
    print(f"\n  Sector Summary:")
    sorted_sectors = sorted(sector_agg.items(),
                            key=lambda x: x[1]["median_return_5d"], reverse=True)
    for sector, agg in sorted_sectors:
        direction = "↑" if agg["median_return_5d"] > 0 else "↓"
        print(f"    {sector:<8s} {direction} "
              f"5d:{agg['median_return_5d']:+.2%}  "
              f"20d:{agg['median_return_20d']:+.2%}  "
              f"breadth:{agg['breadth']:.0%}  "
              f"vol:{agg['avg_vol_ratio']:.2f}x  "
              f"({agg['n_stocks']} stocks)")

    return sector_agg


if __name__ == "__main__":
    compute_sector_features()
