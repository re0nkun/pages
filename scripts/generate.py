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

LOGO_BASE_URL = "https://s3-symbol-logo.tradingview.com/{logoid}.svg"

SECTOR_LABELS_JA = {
    'Commercial Services': '商業サービス',
    'Communications': '通信',
    'Consumer Durables': '耐久消費財',
    'Consumer Non-Durables': '非耐久消費財',
    'Consumer Services': '消費者サービス',
    'Distribution Services': '流通サービス',
    'Electronic Technology': '電子技術',
    'Energy Minerals': 'エネルギー資源',
    'Finance': '金融',
    'Health Services': 'ヘルスケアサービス',
    'Health Technology': 'ヘルスケア技術',
    'Industrial Services': '産業サービス',
    'Miscellaneous': 'その他',
    'Non-Energy Minerals': '非エネルギー資源',
    'Process Industries': 'プロセス産業',
    'Producer Manufacturing': '生産財製造業',
    'Retail Trade': '小売業',
    'Technology Services': '技術サービス',
    'Transportation': '運輸',
    'Utilities': '公益事業',
    'Government': '行政',
}


def fetch_data() -> pd.DataFrame:
    df = (Query()
        .select('name', 'description', 'logoid', 'sector', 'industry', 'market_cap_basic',
                'free_cash_flow_fy',
                'total_assets_yoy_growth_fy', 'ebitda_yoy_growth_fy',
                'close', 'price_52_week_high', 'price_52_week_low',
                'Perf.W', 'Perf.1M', 'Perf.3M', 'Perf.6M', 'Perf.Y',
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


def make_range_bar(pct):
    """52週レンジ内の現在値の位置を横バーで可視化する簡易チャート。"""
    if pd.isna(pct):
        return ""
    pct = max(0.0, min(100.0, float(pct)))
    # 低位=赤寄り、高位=緑寄りのマーカー色
    if pct >= 70:
        color = "#5fbf6a"
    elif pct <= 30:
        color = "#e0605a"
    else:
        color = "#d8c25a"
    return (
        f'<span class="range-bar" title="{pct:.1f}%">'
        f'<span class="range-bar-track">'
        f'<span class="range-bar-marker" style="left:{pct:.1f}%; background:{color};"></span>'
        f'</span>'
        f'<span class="range-bar-label">{pct:.0f}%</span>'
        f'</span>'
    )


def make_logo_tag(logoid):
    """logoidからロゴ<img>タグを生成。logoidが無い場合は空文字（余白確保用のスペーサーは付けない）。"""
    if pd.isna(logoid) or not logoid:
        return ""
    url = LOGO_BASE_URL.format(logoid=logoid)
    return f'<img src="{url}" class="logo-icon" alt="" loading="lazy">'


def make_sparkline_svg(row, width=64, height=22):
    """
    Perf.Y / Perf.6M / Perf.3M / Perf.1M / Perf.W (パフォーマンス%) から
    現在値を基準に過去の相対株価を逆算し、簡易スパークラインを描画する。
    ※ 日次データではなく6点(1年前・6ヶ月前・3ヶ月前・1ヶ月前・1週間前・現在)の近似トレンド。
    """
    close = row.get('close')
    perf_keys = ['Perf.Y', 'Perf.6M', 'Perf.3M', 'Perf.1M', 'Perf.W']

    if pd.isna(close):
        return ""

    points = []
    for k in perf_keys:
        perf = row.get(k)
        if pd.isna(perf):
            points.append(None)
        else:
            points.append(close / (1 + perf / 100))
    points.append(close)

    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return ""

    # 欠損値は前後の値で単純補間（先頭欠損は最初の有効値で埋める）
    filled = []
    last_valid = None
    for p in points:
        if p is not None:
            last_valid = p
        filled.append(last_valid if last_valid is not None else next(v for v in points if v is not None))

    lo, hi = min(filled), max(filled)
    span = hi - lo if hi != lo else 1
    n = len(filled)
    step = width / (n - 1)

    coords = [
        (round(i * step, 1), round(height - 2 - (v - lo) / span * (height - 4), 1))
        for i, v in enumerate(filled)
    ]
    path = " ".join(f"{x},{y}" for x, y in coords)

    trend_up = filled[-1] >= filled[0]
    color = "#4caf7d" if trend_up else "#e5534b"

    return (
        f'<svg class="sparkline" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none">'
        f'<polyline points="{path}" fill="none" stroke="{color}" '
        f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def make_chart_cell(row):
    """スパークラインと52週レンジ位置(%)を1セルにまとめる。"""
    spark = make_sparkline_svg(row)
    pct = row.get('price_in_range_pct')
    pct_label = f'<span class="range-pct">({pct:.1f}%)</span>' if pd.notna(pct) else ""
    return f'<div class="chart-cell">{spark}{pct_label}</div>'


def make_name_cell(row):
    """ロゴ + 会社名(ティッカーへのリンク)を1セルにまとめる。"""
    name = row['name']
    if pd.isna(name):
        return ""
    label = row['description'] if pd.notna(row.get('description')) and row['description'] else name
    logo_tag = make_logo_tag(row.get('logoid'))
    link = (
        f'<a href="https://jp.tradingview.com/symbols/TSE-{name}/" '
        f'target="_blank" title="{label}">{label}</a>'
    )
    return f'<span class="name-cell">{logo_tag}{link}</span>'


def build_result_table(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    result[COLUMN_LABELS_JA['name']] = df.apply(make_name_cell, axis=1)
    result[COLUMN_LABELS_JA['sector']] = df['sector'].map(SECTOR_LABELS_JA).fillna(df['sector'])
    result['株価推移(12ヶ月) / 52週レンジ位置'] = df.apply(make_chart_cell, axis=1)

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

    return result


def render_html(result: pd.DataFrame) -> str:
    format_dict = {
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
    width: 140px;
    max-width: 140px;
    overflow: hidden;
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
  .name-cell {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 100%;
  }}
  .name-cell a {{
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }}
  .logo-icon {{
    width: 18px;
    height: 18px;
    object-fit: contain;
    border-radius: 3px;
    flex-shrink: 0;
    background: #fff;
  }}
  .sparkline {{
    display: block;
    width: 64px;
    height: 22px;
    margin: 0 auto;
  }}
  .chart-cell {{
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    gap: 6px;
  }}
  table.screener-table td:has(.chart-cell) {{
    text-align: center;
  }}
  .chart-cell .range-bar {{
    width: auto;
  }}
  .range-pct {{
    color: #999;
    font-size: 0.75rem;
  }}
  .range-bar {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    width: 100%;
  }}
  .range-bar-track {{
    position: relative;
    display: inline-block;
    width: 70px;
    height: 6px;
    background: #2a2e38;
    border-radius: 3px;
    flex-shrink: 0;
  }}
  .range-bar-marker {{
    position: absolute;
    top: 50%;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    transform: translate(-50%, -50%);
  }}
  .range-bar-label {{
    color: #999;
    font-size: 0.8rem;
    min-width: 2.5em;
    text-align: right;
  }}
  .count {{
    margin-bottom: 12px;
    font-size: 0.9rem;
    color: #bbb;
  }}
</style>
</head>
<body>
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
