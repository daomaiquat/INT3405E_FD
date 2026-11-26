"""Verify saved GPU experiments and write the Vietnamese imbalance report."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PLAN = HERE.parent / 'plan_runs'
REFERENCE = PLAN / 'results' / 'full'
OUTPUT = HERE / 'ket_qua_thu_nghiem_lech_nhan.md'
BASELINE = 'XGB-D03'
TREE_CASES = (BASELINE, 'XGB-W2', 'XGB-W4', 'XGB-RANDOM', 'XGB-HARD')
MLP_CASES = ('MLP-BCE', 'MLP-FOCAL-G1', 'MLP-FOCAL-G2', 'MLP-PAIRWISE')


@dataclass
class Result:
    name: str
    dir: Path
    auc: float
    ap: float
    loss: float
    fold_auc: list[float]
    scores: np.ndarray
    precision: float
    recall: float
    time: float


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_close(calculated, stored, label):
    require(np.isclose(calculated, stored, rtol=0, atol=1e-9), f'{label}: {calculated} != {stored}')


def data_checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_experiment(name, path, manifest, ids, y, fold, test_ids):
    metrics = read_json(path / 'metrics.json')
    require(metrics['name'] == name and metrics['status'] == 'completed', f'Incomplete experiment {name}')
    for key in ('rows', 'test_rows', 'fold_hash', 'early_stop_hash', 'inputs', 'models_use_llm'):
        require(metrics[key] == manifest[key], f'{name}: mismatch for {key}')
    require(metrics['smoke'] is False, f'{name} is a smoke run')
    require(len(metrics['folds']) == 5 and all('RTX 4060' in f['device'] for f in metrics['folds']),
            f'{name}: five RTX 4060 fold records required')
    with np.load(path / 'predictions.npz', allow_pickle=False) as record:
        require(np.array_equal(record['ids'], ids) and np.array_equal(record['fold'], fold),
                f'{name}: OOF ID or fold mismatch')
        require(np.array_equal(record['test_ids'], test_ids), f'{name}: test ID mismatch')
        scores, test_scores = record['oof'], record['test']
        test_by_fold = record['test_by_fold']
        require(test_by_fold.shape == (len(test_ids), 5) and np.isfinite(test_by_fold).all()
                and np.allclose(test_by_fold.mean(axis=1), test_scores, rtol=0, atol=1e-12),
                f'{name}: test fold scores do not reproduce the mean')
    for label, values, size in (('OOF', scores, len(ids)), ('test', test_scores, len(test_ids))):
        require(values.shape == (size,) and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
                f'{name}: invalid {label} scores')
    auc, ap, loss = roc_auc_score(y, scores), average_precision_score(y, scores), log_loss(y, scores)
    check_close(auc, metrics['oof_auc'], f'{name} AUC')
    check_close(ap, metrics['average_precision'], f'{name} AP')
    check_close(loss, metrics['log_loss'], f'{name} Log-loss')
    per_fold = []
    for k, summary in enumerate(metrics['folds']):
        mask = fold == k
        score = roc_auc_score(y[mask], scores[mask])
        require(summary['fold'] == k + 1, f'{name}: fold order differs')
        check_close(score, summary['auc'], f'{name} fold {k + 1} AUC')
        per_fold.append(score)
    budget = int(np.ceil(0.01 * len(y)))
    head = np.lexsort((ids, -scores))[:budget]
    found = int(y[head].sum())
    return Result(name, path, auc, ap, loss, per_fold, scores, found / budget, found / int(y.sum()),
                  float(metrics['seconds']))


def difference(item, baseline):
    return item.auc - baseline.auc, sum(x > y for x, y in zip(item.fold_auc, baseline.fold_auc, strict=True))


def render_table(results):
    base = results[0]
    lines = ['| Cấu hình | OOF ROC AUC | Δ AUC | Fold tăng | AP | Precision@1% | Recall@1% | Giây |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in results:
        delta, wins = difference(item, base)
        change, count = ('—', '—') if item is base else (f'{delta:+.6f}', f'{wins}/5')
        lines.append(f'| `{item.name}` | {item.auc:.6f} | {change} | {count} | {item.ap:.6f} | '
                     f'{item.precision:.2%} | {item.recall:.2%} | {item.time:.1f} |')
    return '\n'.join(lines)


def fold_table(results):
    lines = ['| Cấu hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |', '|---|---:|---:|---:|---:|---:|']
    for item in results:
        values = ' | '.join(f'{x:.6f}' for x in item.fold_auc)
        lines.append(f'| `{item.name}` | {values} |')
    return '\n'.join(lines)


def compare_line(candidate, baseline):
    delta, wins = difference(candidate, baseline)
    return (f'`{candidate.name}` đạt **{candidate.auc:.6f}**, Δ AUC **{delta:+.6f}** với '
            f'**{wins}/5** fold tăng. Δ Average Precision là **{candidate.ap - baseline.ap:+.6f}**.')


def verify_training_contract(tree, mlp):
    baseline, *weighted = [read_json(item.dir / 'metrics.json') for item in tree]
    rounds = [record['rounds'] for record in baseline['folds']]
    for item in weighted:
        require(item['config']['params'] == baseline['config']['params'], f"{item['name']}: XGB settings differ")
        require([record['rounds'] for record in item['folds']] == rounds, f"{item['name']}: round budget differs")
    random, hard = weighted[-2:]
    for comparison in zip(random['folds'], hard['folds'], strict=True):
        require(comparison[0]['negatives_upweighted'] == comparison[1]['negatives_upweighted'] > 0,
                'Hard and random negative counts must match')
    networks = [read_json(item.dir / 'metrics.json') for item in mlp]
    for fold in zip(*(item['folds'] for item in networks), strict=True):
        shared = {(record['initialization_sha256'], record['batch_order_sha256'], record['epochs'],
                   record['feature_count']) for record in fold}
        require(len(shared) == 1, 'The MLP initialization, batch order, epoch or feature count differs')
    return sum(record['skipped_batches'] for record in networks[-1]['folds'])


def uncertainty(candidate, baseline, y):
    sys.path.insert(0, str(PLAN))
    from ensemble import bootstrap_delta

    print(f'Bootstrap 500 mẫu cho {candidate.name} so với {baseline.name}', flush=True)
    interval = bootstrap_delta(y, candidate.scores, baseline.scores, repeats=500, seed=42)
    low, high = interval['ci_low'], interval['ci_high']
    return (f'Khoảng 95% bootstrap ghép cặp cho Δ AUC của `{candidate.name}` so với '
            f'`{baseline.name}` là **[{low:+.6f}, {high:+.6f}]**. Khoảng này tính trên '
            'dự đoán OOF cố định, với 500 mẫu lấy lại theo từng lớp.')


def model_findings(results, y):
    base, *alternatives = results
    best = max(alternatives, key=lambda item: item.auc)
    lines = [compare_line(best, base)]
    changes = [(item, *difference(item, base)) for item in alternatives]
    qualifying = [item for item, delta, wins in changes if delta >= 0.0001 and wins >= 4]
    if qualifying:
        names = ', '.join(f'`{item.name}`' for item in qualifying)
        lines.append(f'{names} đạt ngưỡng sàng lọc của dự án là tăng ít nhất 0.000100 AUC '
                     'và cải thiện ít nhất 4/5 fold.')
        lines.append(uncertainty(max(qualifying, key=lambda item: item.auc), base, y))
    else:
        lines.append('Nhóm này chưa có cấu hình đồng thời tăng ít nhất 0.000100 AUC '
                     'và cải thiện ít nhất 4/5 fold so với đối chứng.')
    return '\n\n'.join(lines)


def main():
    manifest = read_json(REFERENCE / 'manifest.json')
    require(manifest['smoke'] is False and manifest['rows'] == 699960 and manifest['test_rows'] == 299984,
            'Official full-run manifest required')
    for name, expected_hash in manifest['inputs'].items():
        require(data_checksum(PROJECT.parent / "data" / name) == expected_hash, f'{name} SHA-256 differs')
    train = pd.read_csv(PROJECT.parent / 'data' / 'train.csv', usecols=['id', 'is_fraud'])
    ids, y = train.id.to_numpy(), train.is_fraud.to_numpy(dtype=np.int8)
    test_ids = pd.read_csv(PROJECT.parent / 'data' / 'test.csv', usecols=['id']).id.to_numpy()
    with np.load(REFERENCE / BASELINE / 'predictions.npz', allow_pickle=False) as earlier:
        fold = earlier['fold']
    assignment = pd.DataFrame({'id': ids, 'is_fraud': y, 'fold': fold})
    fold_bytes = assignment.to_csv(index=False, lineterminator='\n').encode()
    require(hashlib.sha256(fold_bytes).hexdigest() == manifest['fold_hash'], 'Fold assignments differ')

    tree = []
    for name in TREE_CASES:
        path = (REFERENCE if name == BASELINE else HERE / 'results' / 'xgb') / name
        tree.append(verify_experiment(name, path, manifest, ids, y, fold, test_ids))
    mlp = [verify_experiment(name, HERE / 'results' / 'mlp' / name, manifest, ids, y, fold, test_ids)
           for name in MLP_CASES]
    skipped = verify_training_contract(tree, mlp)
    tree_text = model_findings(tree, y)
    mlp_text = model_findings(mlp, y)
    hard, random = tree[-1], tree[-2]
    hard_text = compare_line(hard, random)

    best_tree = max(tree, key=lambda item: item.auc)
    best_mlp = max(mlp, key=lambda item: item.auc)
    stable_tree = [item for item in tree[1:] if difference(item, tree[0])[0] >= 0.0001
                   and difference(item, tree[0])[1] >= 4]
    if stable_tree:
        candidate = max(stable_tree, key=lambda item: item.auc)
        decision = (f'`{candidate.name}` là ứng viên để xác nhận tiếp trong tổ hợp. '
                    'Kết quả lượt này đánh giá mô hình đơn trên các fold đã cố định.')
    else:
        decision = ('Giữ `XGB-D03` làm mô hình cây trong tổ hợp hiện tại. '
                    'Các thay đổi trọng số và cách chọn mẫu khó trong lượt thử này chưa đạt tiêu chí thay thế.')
    incumbent = read_json(REFERENCE / 'final' / 'metrics.json')['oof_auc']
    references = '\n'.join(f'- [`{item.name}`]({item.dir.relative_to(HERE).as_posix()}/metrics.json)'
                           for item in [*tree[1:], *mlp])
    precision = ('Precision@1% là tỷ lệ gian lận trong 1% giao dịch được xếp cao nhất. '
                 'Recall@1% là tỷ lệ giao dịch gian lận được tìm thấy trong nhóm đó.')
    report = f'''# S6E11 · Thử nghiệm xử lý lệch nhãn

Đã chạy **8 cấu hình mới trên NVIDIA RTX 4060** và đối chiếu với `XGB-D03` đã lưu. Cấu hình cây có AUC cao nhất trong nhóm là **`{best_tree.name}` ({best_tree.auc:.6f})**. Cấu hình MLP có AUC cao nhất trong nhóm là **`{best_mlp.name}` ({best_mlp.auc:.6f})**.

Tất cả kết quả dùng cùng **5 fold** trên **699.960** giao dịch, gồm **11.689** giao dịch gian lận. Validation giữ nguyên tỷ lệ gian lận gần **1,67%**. Các chỉ số được tính lại từ dự đoán OOF và khớp với file kết quả từng lượt. Điểm mô hình cuối hiện có là **{incumbent:.6f}**, thu được từ lần xác nhận lồng nhau của tổ hợp. [Kết quả gốc](../plan_runs/results/full/comparison.md)

{precision} Giây là tổng thời gian ghi theo fold. Thời gian `XGB-D03` gồm cả lượt chọn số vòng, còn các biến thể XGBoost mới dùng số vòng đã lưu, nên cột này không đo cùng một phạm vi công việc.

## XGBoost · Trọng số lớp và mẫu hợp lệ dễ bị báo nhầm

`XGB-W2` và `XGB-W4` lần lượt đặt trọng số lớp gian lận là 2 và 4. `XGB-HARD` tăng trọng số lên 3 cho 10% giao dịch hợp lệ có điểm rủi ro cao nhất từ dự đoán ngoài fold trong phần huấn luyện. `XGB-RANDOM` tăng đúng mức đó cho cùng số giao dịch hợp lệ chọn ngẫu nhiên. Các lượt giữ số vòng `[2194, 1536, 1771, 2306, 2131]` theo từng fold của `XGB-D03`, đã chọn trên tập dừng sớm trong phần huấn luyện. Kết quả đánh giá tác động của trọng số ở ngân sách số vòng cố định.

{render_table(tree)}

{tree_text}

**So sánh mẫu khó với chọn ngẫu nhiên.** Lấy `XGB-RANDOM` làm đối chứng trực tiếp. {hard_text}

## MLP · Focal Loss và học xếp hạng theo cặp

Bốn cấu hình dùng cùng 109 đặc trưng, hai tầng ẩn 128 và 64 nút, dropout 0,10, AdamW, batch 4.096 và 12 epoch. Trọng số khởi tạo và thứ tự batch được đối chiếu trong từng fold. `MLP-BCE` dùng binary cross-entropy. `MLP-FOCAL-G1` và `MLP-FOCAL-G2` giảm ảnh hưởng của mẫu dễ với `γ=1` và `γ=2`. `MLP-PAIRWISE` học để giao dịch gian lận được điểm cao hơn giao dịch hợp lệ được lấy cặp trong batch.

{render_table(mlp)}

{mlp_text}

Đầu ra focal và pairwise được dùng như điểm xếp hạng chưa hiệu chỉnh. Việc biến đổi qua sigmoid đưa điểm về khoảng 0–1 để lưu, với mục đích đo AUC và AP. Số batch chỉ có một lớp được bỏ qua khi tính mất mát theo cặp là **{skipped}** trên toàn bộ 5 fold.

## AUC theo từng fold

{fold_table([*tree, *mlp])}

## Kết luận

{decision}

## Hồ sơ kết quả

{references}
'''
    temporary = OUTPUT.with_suffix('.md.tmp')
    temporary.write_text(report, encoding='utf-8')
    temporary.replace(OUTPUT)
    print(f'Đã xác minh {len(tree) + len(mlp)} dòng. Báo cáo tại {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
