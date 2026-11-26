"""Render the locked screening and confirmation results for the EDA report."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {
    'account_balance_usd': 'số dư', 'auth_method': 'xác thực', 'billing_shipping_mismatch': 'lệch địa chỉ',
    'card_age_months': 'tuổi thẻ', 'card_type': 'loại thẻ', 'channel': 'kênh', 'customer_age': 'tuổi khách',
    'cvv_retry_count': 'thử CVV', 'day_of_week': 'ngày trong tuần', 'device_type': 'thiết bị',
    'hours_since_last_txn': 'giờ từ GD trước', 'ip_country_mismatch': 'lệch quốc gia IP',
    'is_ai_generated_scam_attempt': 'cờ scam AI', 'is_foreign_transaction': 'GD nước ngoài',
    'is_new_merchant': 'merchant mới', 'merchant_category': 'ngành merchant',
    'merchant_risk_score': 'rủi ro merchant', 'prior_disputes': 'tranh chấp',
    'txn_count_last_24h': 'số GD 24h', 'used_vpn': 'VPN', 'velocity_score': 'điểm velocity',
}
STAGES = ('a_to_b', 'b_to_a', 'confirm_c', 'frozen_ensemble_holdout')


def render():
    doc = json.loads((HERE / 'triple_eda_results.json').read_text(encoding='utf-8'))
    data = doc['triple_results']
    values = np.array([[row.get(key, {}).get('gain_logloss_x1m', np.nan) for key in STAGES] for row in data])
    for i, row in enumerate(data):
        if not row['pure_rank_a']:
            values[i, 0] = np.nan
        if not row['pure_rank_b']:
            values[i, 1] = np.nan
    limit = max(1., float(np.nanmax(np.abs(values))))
    cmap = matplotlib.colormaps['RdBu'].copy()
    cmap.set_bad('#edf0f4')
    fig, ax = plt.subplots(figsize=(11.6, 9.3), layout='constrained')
    colors = ax.imshow(values, aspect='auto', cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit))
    ax.set_xticks(range(len(STAGES)), ['Học A → chấm B', 'Học B → chấm A', 'Xác nhận C', 'Holdout cũ'])
    ax.set_yticks(range(len(data)), [' × '.join(LABELS[name] for name in row['fields']) for row in data])
    ax.tick_params(length=0, pad=9)
    ax.xaxis.tick_top()
    for i in range(len(data)):
        for j in range(len(STAGES)):
            x = values[i, j]
            cell = '—' if np.isnan(x) else f'{x:+.2f}'.replace('.', ',')
            ax.text(j, i, cell, ha='center', va='center',
                    color='white' if np.isfinite(x) and abs(x) >= .65 * limit else '#132033', fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Phần tương tác ba biến còn lại sau khi trừ cả ba cặp', fontsize=14, fontweight='bold', pad=40)
    colorbar = fig.colorbar(colors, ax=ax, orientation='horizontal', pad=.055, shrink=.68)
    colorbar.set_label('Mức giảm log loss × 1 triệu · Dương là tốt hơn · Ô xám thiếu hỗ trợ hoặc chưa qua sàng lọc')
    fig.savefig(HERE / 'triple_eda.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    render()
