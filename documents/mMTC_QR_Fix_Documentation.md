# mMTC (遅延制約) 対応のための実コード修正ログ

元のQRアルゴリズムは「スループット（値が大きいほど良い＝Lower制約）」専用にハードコードされており、mMTCの「遅延（値が小さいほど良い＝Upper制約）」に対応するため、実際に行った**すべてのコード修正箇所**を以下に列挙します。

## 1. SLA設定と評価指標の食い違い修正
**対象ファイル:** `experiments_qr.py`
mMTCのSLAがスループットと同じ形式で設定されていたため、「遅延 300ms 以下」というシミュレータ環境の実態に合わせました。
```python
# 修正前
mmtc_sla = {'threshold': 0.99, 'quantile': quantile_list, 'type': 'lower', 'kpi_key': 'l1_info'}
# 修正後
mmtc_sla = {'threshold': 300, 'quantile': quantile_list, 'type': 'upper', 'kpi_key': 'l1_info'}
```

## 2. 分位数（Quantile）追跡の反転
**対象ファイル:** `qr_scenario_creator.py`
`quantile = 0.05` は遅延に対して「最も幸運な5%」を追跡し危険なため、遅延の場合は最悪の95パーセンタイル（0.95）を追跡するよう反転させました。
```python
# 追加した修正
actual_quantile = 1.0 - quantile if mmtc_sla['type'] == 'upper' else quantile
```

## 3. RBFカーネルの「未探索＝0」問題の補正
**対象ファイル:** `qr_control.py` ( `select_action` 関数 )
RBFカーネルは未探索領域の予測値を `0.0`（＝遅延ゼロで安全と誤認）に減衰させるため、Upper制約に対し「不確実（未探索）＝危険」となるようペナルティを足し込みました。
```python
prediction, uncertainty = h.algorithm.predict_with_uncertainty(x)
# 追加した修正
if h.constraint_type == 'upper':
    prediction = prediction + (h.sla_threshold * 2) * uncertainty
```

## 4. スコア計算とフォールバック探索の逆転
**対象ファイル:** `qr_control.py` ( `select_action` 関数 )
予測値（遅延）が「小さいほど良い」行動（Upper制約）を正しく選べるよう、スコアリングと最小値探索を実装しました。
```python
# 修正箇所抜粋
if h.constraint_type == 'lower':
    best_prediction = -np.inf
    random_scores[a] = prediction... # (省略)
else: # upper制約
    best_prediction = np.inf
    random_scores[a] = -prediction... # マイナスでスコア化

# Fallbackの最適アクション探索
if (h.constraint_type == 'lower' and prediction > best_prediction) or \
   (h.constraint_type == 'upper' and prediction < best_prediction):
    best_prediction = prediction
    best_fallback_action = a    
```

## 5. 調整ペナルティ（Over-allocation罰則）の逆転
**対象ファイル:** `qr_control.py` ( `penalize_original_actions` 関数 )
PRBを過剰要求して削られた際に加えられる予測値ペナルティ（マイナス値）が、遅延に対しては「予測遅延が減る＝ご褒美」になっていたため反転させました。
```python
dynamic_penalty = base_penalty_coeff * (adjustment_delta / self.n_prbs)
# 追加した修正
if h.constraint_type == 'upper':
    dynamic_penalty = -dynamic_penalty # マイナスのペナルティをプラス（遅延増加）へ反転
h.algorithm.add_penalty_update(original_x, penalty_coefficient=dynamic_penalty)
```

## 6. 分位数の動的更新とハードコード制限の修正
**対象ファイル:** `qr_control.py` ( 更新ループ )
不安時やSLA違反時に分位数を強制的に「0.05上限」「-0.01マイナス」とするスループット専用のハードコードが、せっかく反転させた0.95を破壊していたため、制約に合わせて上限下限と加減算の向きを動的化しました。
```python
sign = -1 if h.constraint_type == 'lower' else 1

if final_uncertainty[j] > uncertainty_threshold:
    self.current_quantiles[j] += sign * 0.01  # upperなら 0.95 -> 0.96... と悲観側に上がる
elif info.get('violations')[j] > 0:
    self.current_quantiles[j] += sign * 0.005

# ハードコード（max_quantile = 0.05）の撤廃と動的クリップ
if h.constraint_type == 'lower':
    min_q, max_q = 0.01, 0.05
else:
    min_q, max_q = 0.95, 0.99
self.current_quantiles[j] = np.clip(self.current_quantiles[j], min_q, max_q)
```

## 7. 勾配ペナルティ条件の修正
**対象ファイル:** `qr_util.py` ( `KernelizedOnlineQuantileRegressor.update` )
モデルがSLA違反を「見逃した」時に課される勾配ペナルティの判定が、Upper制約（予測が小さすぎた時）にも対応できるよう条件分岐を追加しました。
```python
if constraint_type == 'lower' and y_true < sla_threshold and prediction >= sla_threshold:
    gradient_update *= self.gradient_penalty
elif constraint_type == 'upper' and y_true > sla_threshold and prediction <= sla_threshold:
    gradient_update *= self.gradient_penalty
```

## 8. KNNフォールバックのデススパイラル（局所解の罠）脱出機構の追加
**対象ファイル:** `qr_control.py` ( `select_action` 関数内の KNN 処理 )
比較ベースラインである `KBRL` は「見つからなければ最大PRB(150)を要求する」という自己防衛が最初から働いていましたが、QRのKNNフォールバックは「周囲の近傍状態がすべてSLA違反だった場合、近傍が『過去に試した中で最大のPRB』を思考停止で出し続ける」という実装上の欠陥があり、トラフィック急増時にeMBB・mMTC共に永遠にSLA違反から抜け出せなくなるデススパイラルに陥っていました。
これを元のアルゴリズムの本来の意図（探索と自己修復）に合わせるため、**「すべての近傍が失敗だった場合は、過去の最大値に +10% のPRBを上乗せして強制探索（ジャンプ）する」**という修正を追加しました。

```python
# 追加した修正
else:
    successful_actions_normalized = relevant_actions[k_nearest_indices]
    chosen_action_normalized = np.max(successful_actions_normalized)
    best_action_for_slice = int(round(chosen_action_normalized * self.n_prbs))
    
    # Aggressively explore higher PRBs if all known nearest neighbors failed!
    jump_prbs = int(max(1, self.n_prbs * 0.10)) # Jump by 10%
    best_action_for_slice = min(self.n_prbs, best_action_for_slice + jump_prbs)
```

以上が、mMTCにおけるPRB割り当ての枯渇とSLA違反100%バグを根本から治療し、QRアルゴリズムをKBRLと同等の強靭なベースラインとして機能させるために、実際にコードに加えたすべての対応です。
