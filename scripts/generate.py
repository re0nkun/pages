"""
日本株マルチファクター・スクリーナー
TradingView Query API からデータを取得し、フィルタリング結果を
output/index.html として書き出す。GitHub Actions から日次実行される想定。
"""

import sys
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from tradingview_screener import Query, col

OUTPUT_PATH = "output/index.html"
JST = timezone(timedelta(hours=9))

SECTOR_FCF_YIELD = {s: t for t, ss in [
    (0.03, ['Electronic Technology', 'Technology Services', 'Health Technology']),
    (0.04, ['Health Services', 'Consumer Durables', 'Consumer Non-Durables', 'Consumer Services',
            'Retail Trade', 'Commercial Services', 'Communications', 'Distribution Services',
            'Industrial Services', 'Producer Manufacturing', 'Miscellaneous']),
    (0.05, ['Transportation', 'Process Industries', 'Non-Energy Minerals', 'Utilities']),
    (0.06, ['Energy Minerals']),
] for s in ss}

EXCLUDED_INDUSTRIES = {
    'Multi-Line Insurance', 'Specialty Insurance', 'Insurance Brokers/Services',
    'Major Banks', 'Regional Banks', 'Investment Banks/Brokers', 'Investment Managers',
    'Financial Conglomerates', 'Finance/Rental/Leasing', 'Financial Publishing/Services',
    'Real Estate Investment Trusts', 'Real Estate Development',
}

COLUMN_LABELS_JA = {
    'name': '銘柄名',
    'sector': 'セクター',
    'industry': '業種',
    'market_cap_basic': '時価総額',
    'fcf_yield': 'FCFイールド',
    'fcf_yield_threshold': 'FCF閾値',
    'price_book_ratio': 'PBR',
    'price_in_range_pct': '52週レンジ位置(%)',
    'ebitda_yoy_growth_fy': 'EBITDA成長率(前年比)',
    'total_revenue_yoy_growth_fy': '売上高成長率(前年比)',
    'ebitda_vs_assets_growth': 'EBITDA-資産成長差',
    'asset_shrinking': '資産縮小フラグ',
    'return_on_equity': 'ROE',
    'current_ratio': '流動比率',
    'debt_to_equity': 'D/E倍率',
    'average_volume_10d_calc': '10日平均出来高',
}


def fetch_data() -> pd.DataFrame:
    df = (Query()
        .select('name', 'sector', 'industry', 'market_cap_basic', 'free_cash_flow_fy',
                'total_assets_yoy_growth_fy', 'ebitda_yoy_growth_fy',
                'close', 'price_52_week_high', 'price_52_week_low',
                'debt_to_equity',
                'return_on_equity',
                'net_income_fy',
                'current_ratio',
                'average_volume_10d_calc',
                'cash_f_operating_activities_fy',
                'total_revenue_yoy_growth_fy',
                'price_book_ratio',
        )
        .where(
            col('is_primary') == True,
            col('typespecs').has('common'),
            col('type') == 'stock',
            col('exchange') == 'TSE',
            col('market_cap_basic').between(1e9, 2e10),
            col('free_cash_flow_fy') > 0,
            col('cash_f_operating_activities_fy') > 0,
            col('ebitda_yoy_growth_fy') >= 5,
            col('total_revenue_yoy_growth_fy') >= 3,
            col('total_assets_yoy_growth_fy') >= -5,
            col('total_assets_yoy_growth_fy') <= col('ebitda_yoy_growth_fy'),
            col('net_income_fy') > 0,
            col('return_on_equity') >= 8,
            col('debt_to_equity').between(0, 2.0),
            col('current_ratio') >= 1.2,
            col('average_volume_10d_calc') >= 50000,
        )
        .limit(3000)
        .set_markets('japan')
        .get_scanner_data())[1]
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df['industry'].isin(EXCLUDED_INDUSTRIES)].copy()

    # FCFイールド
    df['fcf_yield'] = df['free_cash_flow_fy'] / df['market_cap_basic']
    df['fcf_yield_threshold'] = df['sector'].map(SECTOR_FCF_YIELD).fillna(0.04)

    # 52週レンジ位置
    week52_range = df['price_52_week_high'] - df['price_52_week_low']
    df['price_in_range_pct'] = (
        (df['close'] - df['price_52_week_low']) / week52_range.replace(0, np.nan) * 100
    ).clip(0, 100).round(1)

    # 成長の質
    asset_growth = df['total_assets_yoy_growth_fy']
    df['ebitda_vs_assets_growth'] = (df['ebitda_yoy_growth_fy'] - asset_growth).round(2)
    df['asset_shrinking'] = asset_growth < 0

    # フィルタリング
    df = df[df['fcf_yield'] >= df['fcf_yield_threshold'] * 1.2]

    return df.sort_values('fcf_yield', ascending=False)


def make_tv_link(name):
    if pd.isna(name):
        return ""
    return f'<a href="https://jp.tradingview.com/symbols/TSE-{name}/" target="_blank">{name}</a>'


def build_result_table(df: pd.DataFrame) -> pd.DataFrame:
    result = df[['name']].copy()

    # FCFイールドと閾値を1カラムに統合
    result['FCFイールド'] = df.apply(
        lambda r: f"{r['fcf_yield']:.2%}(閾値{r['fcf_yield_threshold']:.2%})", axis=1
    )

    # EBITDA-資産成長差と資産縮小フラグを1カラムに統合
    result['EBITDA-資産成長差'] = df.apply(
        lambda r: f"{r['ebitda_vs_assets_growth']:.1f}" + (" ⚠️(資産縮小)" if r['asset_shrinking'] else ""),
        axis=1
    )

    result['ROE(基準8.0以上)'] = df['return_on_equity']
    result['D/E倍率(基準0〜2.00)'] = df['debt_to_equity']
    result['52週レンジ位置'] = df['price_in_range_pct'].map(
        lambda v: f"{v:.1f}%" if pd.notna(v) else ""
    )

    return result.rename(columns={'name': COLUMN_LABELS_JA['name']})


def render_html(result: pd.DataFrame) -> str:
    format_dict = {
        '銘柄名': make_tv_link,
        'ROE(基準8.0以上)': '{:.1f}',
        'D/E倍率(基準0〜2.00)': '{:.2f}',
    }

    table_html = (
        result.style
        .format(format_dict)
        .hide(axis="index")
        .set_table_attributes('class="screener-table"')
        .to_html(escape=False)
    )

    updated_at = datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>日本株マルチファクター・スクリーナー</title>
<style>
  body {{
    font-family: "Hiragino Sans", "Yu Gothic", sans-serif;
    background: #0f1115;
    color: #e6e6e6;
    margin: 0;
    padding: 24px;
  }}
  h1 {{
    font-size: 1.4rem;
    margin-bottom: 4px;
  }}
  .meta {{
    color: #999;
    font-size: 0.85rem;
    margin-bottom: 20px;
  }}
  table.screener-table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.85rem;
  }}
  table.screener-table th, table.screener-table td {{
    border: 1px solid #333;
    padding: 6px 10px;
    text-align: right;
    white-space: nowrap;
  }}
  table.screener-table th:first-child, table.screener-table td:first-child {{
    text-align: left;
  }}
  table.screener-table th {{
    background: #1a1d24;
    position: sticky;
    top: 0;
  }}
  table.screener-table tr:nth-child(even) {{
    background: #171a21;
  }}
  table.screener-table a {{
    color: #6db3f2;
    text-decoration: none;
  }}
  table.screener-table a:hover {{
    text-decoration: underline;
  }}
  .count {{
    margin-bottom: 12px;
    font-size: 0.9rem;
    color: #bbb;
  }}
</style>
</head>
<body>
  <h1>日本株マルチファクター・スクリーナー</h1>
  <div class="meta">最終更新: {updated_at} ／ 該当銘柄数: {len(result)}</div>
  {table_html}
</body>
</html>
"""


def render_error_html(message: str) -> str:
    updated_at = datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>スクリーナー: エラー</title>
<style>
  body {{ font-family: sans-serif; background: #0f1115; color: #e6e6e6; padding: 24px; }}
  pre {{ background: #1a1d24; padding: 16px; overflow-x: auto; color: #f28b82; }}
</style>
</head>
<body>
  <h1>データ取得または処理でエラーが発生しました</h1>
  <div>最終試行: {updated_at}</div>
  <pre>{message}</pre>
</body>
</html>
"""


def main():
    try:
        raw = fetch_data()
        df = process(raw)
        result = build_result_table(df)
        html = render_html(result)
    except Exception:
        # 失敗してもページは必ず生成し、原因が分かるようにする
        html = render_error_html(traceback.format_exc())
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(html)
        print("generate.py failed; error page written.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {len(result)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
