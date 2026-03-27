# シナリオ1（mMTC含む）評価のために実施した eSQR (Extended SQR) 実装まとめ

元のコードベースはシナリオ0（eMBB + eMBBの2スライス構成）の検証にのみ特化しており、mMTCを含むシナリオ1（eMBB×3 + mMTC×2の5スライス構成）を正しく評価するために拡張した **eSQR** では、以下の主要な変更を加えた。

> [!NOTE]
> eSQR の「e」は「mMTC（遅延制約）への機能拡張」を指す。評価の公平性を期すため、初期検証時动态的な安全調整ロジック（分位数・マージンの動的変動）は削除され、オリジナルの SQR と同じ固定パラメータ（分位数 0.05/0.95, マージン 2）で動作する。

---

## A. mMTC の SLA 定義・評価方向の修正

### A-1. SLA閾値・制約タイプの修正
**対象:** `experiments_qr.py` / `scenario_creator.py`

**問題:** mMTCのSLAがスループット要件（下限値）と同じ設定になっており、シミュレータの平均遅延要件（上限値）と乖離していた。

**eSQR の実装:**
```python
# mMTC向け（experiments_qr.py）
mmtc_sla = {
    'threshold': 10000, 
    'quantile': quantile_list, 
    'type': 'upper',    # 「しきい値(10s)以下が安全」
    'kpi_key': 'l1_info'
}
```

### A-2. 分位数（Quantile）追跡方向の動的反転
**対象:** `qr_scenario_creator.py`

**問題:** 0.05分位点は「最良の5%」を指すが、遅延においては「最悪の95%（上位側）」を追跡しないとSLAを保証できない。

**eSQR の実装:**
```python
# upper制約（遅延）では quantile を 1.0 - 0.05 = 0.95 に反転させ、最悪ケースをモデルに追跡させる
actual_quantile = 1.0 - quantile if constraint_type == 'upper' else quantile
```

### A-4. 空状態（トラフィック無し）の判定ロジックの汎用化
**対象:** `qr_control.py`

**問題:** オリジナルではeMBBのUE数（`cbr_ue == 0`）のみで空判定を行っており、mMTCスライスが空のときでも強引に学習・制御が行われていた。

**eSQR の実装:**
- **eMBB:** 従来どおり `cbr_ue == 0` で判定。
- **mMTC:** 状態変数の先頭にある `devices` (アクティブデバイス数) が 0 の場合に空と判定し、不要な学習やペナルティの更新をスキップするように拡張。

---

## B. eSQR アルゴリズムの Upper 制約対応（方向性修正）

元のQRアルゴリズムはあらゆる計算フローで「予測値が大きいほど良い」を無条件に前提としていたため、遅延（小さいほど良い）に対応するために拡張した **eSQR** では以下の箇所を修正した。

### B-1. カーネルの「未探索域＝0」バイアスへの悲観補正
**対象:** `qr_control.py`

**問題:** カーネル法は未知領域で予測値が0に収束する。遅延の場合、0msは「完璧に安全」と誤認される。

**eSQR の実装:**
```python
# 不確実性に比例した「悲観的ペナルティ」を追加し、未知領域を危険（高遅延）とみなす
if h.constraint_type == 'upper':
    prediction = prediction + (h.sla_threshold * 2) * uncertainty
```

### B-2. 行動選択（スコアリング）の方向反転
**対象:** `qr_control.py`

**eSQR の実装:**
```python
# 安全な行動の中でも「遅延が小さい（コストは度外視）」行動を優先する符号反転
if h.constraint_type == 'lower':
    random_scores[a] = prediction + self.exploration_factor * uncertainty
else:
    random_scores[a] = -prediction + self.exploration_factor * uncertainty
```

### B-3. 調整ペナルティ（過失割り当て罰則）の方向反転
**対象:** `qr_control.py`

**eSQR の実装:**
```python
# PRB削減（ダウンサイズ）時に予測値を意図的に悪化させる方向を制約タイプで切り替え
dynamic_penalty = self.adjustment_penalty if h.constraint_type == 'lower' else -self.adjustment_penalty
h.algorithm.add_penalty_update(x, dynamic_penalty)
```

### B-4. 学習フェーズ（勾配ペナルティ）の Upper 判定
**対象:** `qr_util.py`

**eSQR の実装:**
```python
# SLA違反（遅延超過）を見逃した際の罰則を学習プロセスに追加
elif constraint_type == 'upper' and y_true > sla_threshold and prediction <= sla_threshold:
    gradient_update *= self.gradient_penalty
```

---

## D. 評価・可視化の整備

### D-1. Per-slice 違反数ログの追加
スライスごとの累積違反件数をリアルタイム表示し、特定の構成（mMTCなど）のみが大きく失敗していないかを確認可能にした。

### D-2. グラフスケールの修正（plot_results.py）
違反率（Violation Rate）のY軸をオートスケール、または最大100%固定に対応させ、大規模な違反も見落とさないようにした。

---

## 修正の全体サマリー（eSQR 主要項目）

| 分類 | 対象ファイル | 変更の本質 |
|------|-------------|-----------|
| A-1 | `experiments_qr.py` | mMTC SLA定義をupper制約（遅延≤10000ms）に修正 |
| A-2 | `qr_scenario_creator.py` | Quantile追跡方向を反転（0.05→0.95） |
| A-3 | `qr_scenario_creator.py` | mMTCのカーネルを MaternKernel に統一 |
| B-1 | `qr_control.py` | 未探索領域に対する悲観ペナルティの追加 |
| B-2 | `qr_control.py` | 行動選択ロジック（スコアリング）の方向反転 |
| B-3 | `qr_control.py` | 過剰割り当てペナルティの符号反転 |
| B-4 | `qr_util.py` | 勾配ペナルティ（学習側）の判定条件追加 |
| D-1 | `experiments_qr.py` | Per-slice違反ログの追加 |
| D-2 | `plot_results.py` | グラフY軸レンジのスケール対応 |
