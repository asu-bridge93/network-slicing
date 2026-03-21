# シナリオ1（mMTC含む）評価のために実施した実装まとめ

元のコードベースはシナリオ0（eMBB + eMBBの2スライス構成）の検証にのみ特化しており、mMTCを含むシナリオ1（eMBB×3 + mMTC×2の5スライス構成）を評価するために以下の変更を加えた。

---

## A. mMTC の SLA 定義・評価方向の修正

### A-1. SLA閾値・制約タイプの修正
**対象:** `experiments_qr.py`

**問題:** mMTCのSLAがスループット（成功率0.99以上）と同じ設定になっており、シミュレータ環境の実態（「1ステップあたりの平均遅延が300ms以下」）と完全に乖離していた。

**修正内容:**
```python
# 修正前（元のコード）
mmtc_sla = {'threshold': 0.99, 'quantile': quantile_list, 'type': 'lower', 'kpi_key': 'l1_info'}

# 修正後
mmtc_sla = {'threshold': 300, 'quantile': quantile_list, 'type': 'upper', 'kpi_key': 'l1_info'}
# type='upper': 「しきい値以下が安全」を意味する上限制約
# threshold=300: 1stepあたりの正規化遅延 (slots_per_step / norm_const_mmtc['delay'])
```

---

### A-2. 分位数（Quantile）追跡方向の反転
**対象:** `qr_scenario_creator.py`

**問題:** `quantile = 0.05` は「最も楽観的な（遅延が一番小さい）5パーセンタイル側」をモデルが追跡することを意味する。スループットではこれでよいが、遅延の場合は「最悪ケース（遅延が膨らむ方向）」を追跡しないとSLA保証ができない。つまり0.05を使い続けると、「たまたま運良く遅延が短かった状況」を基準に学習してしまい、実際にはSLA違反が多発する。

**修正内容:**
```python
# 追加した処理（qr_scenario_creator.py）
# upper制約（遅延）では quantile を反転させ、0.95（上位95%の悪い方）を追跡させる
actual_quantile = 1.0 - quantile if constraint_type == 'upper' else quantile
# quantile=0.05 → actual_quantile=0.95 として KernelizedOnlineQuantileRegressor に渡す
```

---

## B. QRアルゴリズムの Upper 制約対応（演算の方向性修正）

元のQRアルゴリズムはあらゆる計算フローで「予測値が大きいほど良い」を無条件に前提としていた。
遅延は「小さいほど良い」ため、以下の4箇所でその前提を逆転させる必要があった。

### B-1. RBFカーネルの「未探索域＝0（安全）」バイアス補正
**対象:** `qr_control.py`（`select_action` 関数内）

**問題:** ガウシアン（RBF）カーネルは、学習データが存在しない未探索領域に対して予測値を `0.0` に収束させる（「0に引っ張られる」性質）。スループットなら `0 = 最悪` と正しく解釈されるが、遅延なら `0 = 最高（遅延ゼロ）` と誤認され、未探索のPRBが常に「安全」と判断されてしまう。これにより、AIが小さなPRB（例: 2 PRB）しか割り当てず、mMTCのキューが無限に積み上がっていた。

**修正内容:**
```python
prediction, uncertainty = h.algorithm.predict_with_uncertainty(x)

# upper制約では「未探索 = 危険（遅延が大きい）」となるよう、
# 不確実性に比例した悲観的ペナルティを足す
if h.constraint_type == 'upper':
    prediction = prediction + (h.sla_threshold * 2) * uncertainty
# 不確実性が高い（未探索）ほど prediction が大きくなり、安全圏外と判定される
```

---

### B-2. スコアリングとフォールバック探索方向の逆転
**対象:** `qr_control.py`（`select_action` 関数内）

**問題:** 安全な行動を複数候補から選ぶ際、元コードは `optimistic_score = prediction + exploration_bonus` の最大値を選ぶ。これはスループット（大きほど良い）では正しいが、遅延（小さいほど良い）では「最も遅延が大きい（危険な）行動」を選んでしまう。

**修正内容:**
```python
# Lower制約（スループット）：大きい予測値に高スコア
if h.constraint_type == 'lower':
    best_prediction = -np.inf  # 初期値は最小
    random_scores[a] = prediction + self.exploration_factor * uncertainty
    optimistic_score = prediction + self.exploration_factor * uncertainty

# Upper制約（遅延）：小さい予測値に高スコア（符号を反転）
else:
    best_prediction = np.inf   # 初期値は最大
    random_scores[a] = -prediction + self.exploration_factor * uncertainty
    optimistic_score = -prediction + self.exploration_factor * uncertainty

# フォールバック用：制約方向に応じた「最も良い」予測値を選ぶ
if (h.constraint_type == 'lower' and prediction > best_prediction) or \
   (h.constraint_type == 'upper' and prediction < best_prediction):
    best_prediction = prediction
    best_fallback_action = a
```

---

### B-3. 調整ペナルティ（過剰割り当て罰則）の方向反転
**対象:** `qr_control.py`（`penalize_original_actions` 関数）

**問題:** 割り当てたPRBがシステム上限を超えて調整（削減）された場合、「要求したPRBが減らされた → その行動は過剰だった」という学習をモデルに促すために、予測値にマイナスの罰則（ペナルティ）を加える処理がある。スループットでは「予測スループットを下げる（悪化させる）」ことが罰として機能するが、遅延では「予測遅延を下げる（改善させる）」ことになってしまい、逆に過剰割り当てを「褒める」挙動になっていた。

**修正内容:**
```python
# ベースとなるペナルティ係数（マイナス値）
dynamic_penalty = base_penalty_coeff * (adjustment_delta / self.n_prbs)

# Upper制約（遅延）では、遅延を「増加」させることが罰になるため符号を反転
if h.constraint_type == 'upper':
    dynamic_penalty = -dynamic_penalty  # マイナス → プラス（遅延増加方向に罰）

h.algorithm.add_penalty_update(original_x, penalty_coefficient=dynamic_penalty)
```

---

### B-4. 勾配ペナルティの Upper 制約判定追加
**対象:** `qr_util.py`（`KernelizedOnlineQuantileRegressor.update` 内）

**問題:** 「モデルがSLA違反を見逃した（SLA違反しているのに安全と予測した）」場合に勾配を強化する罰則処理が存在したが、判定条件が `lower` 制約のみを想定しており、`upper` 制約では発動しなかった。

**修正内容:**
```python
# Lower制約: 実測値 < 閾値（SLA違反）かつ 予測値 >= 閾値（安全と誤予測）
if constraint_type == 'lower' and y_true < sla_threshold and prediction >= sla_threshold:
    gradient_update *= self.gradient_penalty

# Upper制約（追加）: 実測値 > 閾値（SLA違反）かつ 予測値 <= 閾値（安全と誤予測）
elif constraint_type == 'upper' and y_true > sla_threshold and prediction <= sla_threshold:
    gradient_update *= self.gradient_penalty
```

---

### B-5. 分位数の動的更新方向と上下限のハードコード撤廃
**対象:** `qr_control.py`（更新ループ内）

**問題1:** 分位数の動的調整（不安定・違反時には分位数を変える処理）が、`lower` 制約のみを想定して「常に減少方向」に固定されていた。`upper` 制約では0.95から0.94...と下がり、正しく反転させた分位数を自ら破壊していた。

**問題2:** 分位数の上限を `max_quantile = 0.05` とハードコードしており、`upper` 制約で上限であるべき0.99を超えないようにする代わりに、0.05まで強制リセットするバグがあった。

**修正内容:**
```python
# 方向を制約タイプに合わせて動的化
sign = -1 if h.constraint_type == 'lower' else +1

if final_uncertainty[j] > uncertainty_threshold:
    self.current_quantiles[j] += sign * 0.01   # lowerなら減少、upperなら増加
elif violations[j] > 0:
    self.current_quantiles[j] += sign * 0.005

# ハードコード撤廃：制約タイプに合わせたクリップ範囲を動的設定
if h.constraint_type == 'lower':
    min_q, max_q = 0.01, 0.05
else:  # upper
    min_q, max_q = 0.95, 0.99
self.current_quantiles[j] = np.clip(self.current_quantiles[j], min_q, max_q)
```

---

## C. SQR 提案機構：デススパイラル脱出ジャンプ（探索の安全機構）

### C-1. KNNフォールバック時の強制探索ジャンプ
**対象:** `qr_control.py`（`select_action` 内 KNNフォールバック処理）

**問題:** 既存のQRコードは、安全な行動が一つも見つからない場合にKNNフォールバック（過去の近傍状態から判断する緊急回避）を使う。しかし、近傍の全ての過去データがSLA違反だった場合、「過去に試した中で最大のPRB」を無条件で繰り返す実装になっていた。

例：キューが溢れている状態で「これまで最大6 PRBしか試していなかった」としたら、毎ステップ6 PRBを要求し続け、永遠にキューが詰まる「デススパイラル」に陥る。

比較ベースラインのKBRLは、安全な行動が見つからない場合にループ末尾の最大値（150 PRB）が自然に選ばれる実装になっており、サルベージ能力を持っていた。QRにはこの機能が欠如していた。

**修正内容（SQRの提案機構）:**
```python
# 全近傍がSLA違反だった場合
else:
    # 近傍中の最大PRBを基準に…
    chosen_action_normalized = np.max(relevant_actions[k_nearest_indices])
    best_action_for_slice = int(round(chosen_action_normalized * self.n_prbs))

    # + 10% のPRBを上乗せして未探索領域に強制ジャンプ（デススパイラル脱出）
    jump_prbs = int(max(1, self.n_prbs * 0.10))
    best_action_for_slice = min(self.n_prbs, best_action_for_slice + jump_prbs)
```

この機構により、数ステップでPRBが適切なレベルまで自律的に引き上げられ、モデルが新たな安全圏を発見した後は、コスト最小化ロジックが自動的に最適な（節約した）PRBに戻す。

---

## D. 評価・可視化の整備

### D-1. Per-slice 違反数ログの追加
**対象:** `experiments_qr.py`

**変更内容:** どのスライスで何件の違反が発生しているかを1000ステップごとにコンソール出力するよう追加した（`Total Violations = X`）。これにより、mMTCとeMBBどちらで問題が起きているかをリアルタイムで追跡できるようになった。

---

### D-2. グラフ縦軸スケールの修正
**対象:** `plot_results.py`

**問題:** SLA違反率のY軸が `(0, 0.06)` にハードコードされており、違反率が高い実験段階の結果（最大100%）が全てグラフの上限を超えてしまいグラフが意味をなさなかった。

**修正内容:**
```python
# 修正前
axs[violations_idx].set_ylim((0, 0.06))
axs[violations_idx].set_yticks(np.arange(0, 0.06, 0.01))

# 修正後
axs[violations_idx].set_ylim((0, 1.0))
axs[violations_idx].set_yticks(np.arange(0, 1.0, 0.1))
```

---

## 修正の全体サマリー

| 分類 | 対象ファイル | 変更の本質 |
|------|-------------|-----------|
| A-1 | `experiments_qr.py` | mMTC SLA定義をupper制約（遅延≤300ms）に修正 |
| A-2 | `qr_scenario_creator.py` | Quantile追跡方向を反転（0.05→0.95） |
| B-1 | `qr_control.py` | RBFカーネルの0バイアスに悲観ペナルティを追加 |
| B-2 | `qr_control.py` | スコア・フォールバック探索の方向を反転 |
| B-3 | `qr_control.py` | 過剰割り当てペナルティの符号をupper制約で反転 |
| B-4 | `qr_util.py` | 勾配ペナルティ条件をupper制約にも追加 |
| B-5 | `qr_control.py` | Quantile動的更新の方向とハードコード上限を修正 |
| C-1 | `qr_control.py` | KNNフォールバック時に+10%ジャンプ探索を追加（SQR新機構）|
| D-1 | `experiments_qr.py` | Per-slice違反ログを追加 |
| D-2 | `plot_results.py` | グラフY軸レンジを0〜100%対応に変更 |
