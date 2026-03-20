# uv 仮想環境セットアップガイド — network-slicing

> **uv** は非常に高速な Python パッケージマネージャです。`pip` + `venv` の代替として使えます。
> このガイドに沿って進めることで、`network-slicing` をクリーンな仮想環境で動かせます。

---

## 前提条件

- macOS / Linux（または WSL2 on Windows）
- Python 3.9 以上がインストールされていること
- `uv` がインストールされていること（↓ インストール方法）

---

## Step 0: uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール後、ターミナルを再起動するか以下を実行してパスを通します:

```bash
source $HOME/.cargo/env
```

インストール確認:

```bash
uv --version
```

---

## Step 1: リポジトリのクローンと移動

```bash
git clone https://github.com/jjalcaraz-upct/network-slicing.git
cd network-slicing
```

（すでにダウンロード済みの場合はそのフォルダに移動してください）

---

## Step 2: 仮想環境の作成

```bash
uv venv .venv --python 3.10
```

> `--python 3.10` の部分は、インストール済みの Python バージョンに合わせて変更してください。
> `3.9` / `3.10` / `3.11` で動作します。

仮想環境を有効化します:

```bash
# macOS / Linux
source .venv/bin/activate
```

プロンプトの先頭に `(.venv)` が表示されれば成功です。

---

## Step 3: 基本依存パッケージのインストール

```bash
uv pip install numpy pandas scipy matplotlib gymnasium
```

---

## Step 4: gym-ran_slice パッケージのインストール（必須）

このリポジトリ独自の Gym 環境をローカルインストールします:

```bash
cd gym-ran_slice
uv pip install -e .
cd ..
```

> `-e` は「編集可能モード（editable install）」です。
> ソースを変更してもインストールし直す必要がなくなります。

---

## Step 5: 強化学習ライブラリのインストール

使用するアルゴリズムに応じてインストールしてください。

### KBRL / BQR / QR 実験を動かす場合（追加不要）
Steps 3〜4 のみで動作します。

### stable-baselines3（DQN / PPO / A2C 等）を使う場合

```bash
uv pip install stable-baselines3
```

### stable-baselines（旧版 v2、TensorFlow 1.x ベース）を使う場合

> ⚠️ TensorFlow 1.x は Python 3.7 以下を必要とします。Python 3.10 以上では動作しません。
> 旧版が必要な場合は Python 3.7 で仮想環境を作成してください:

```bash
uv venv .venv --python 3.7
source .venv/bin/activate
uv pip install tensorflow==1.9.0 stable-baselines==2.10.1
```

### keras-rl / NAF 実験を動かす場合

```bash
uv pip install keras keras-rl
```

---

## Step 6: インストール確認

```bash
python -c "import numpy, pandas, scipy, matplotlib, gymnasium; print('All OK')"
python -c "import gym_ran_slice; print('gym_ran_slice OK')"
```

---

## Step 7: 実験スクリプトの実行

仮想環境が有効化された状態で以下を実行します:

### KBRL アルゴリズムの実験

```bash
python experiments_kbrl.py
```

### DQN アルゴリズムの実験（stable-baselines3 が必要）

```bash
python experiments_dqn.py
```

### stable-baselines の RL 実験（PPO、A2C 等）

```bash
python experiments_rl.py
```

---

## Step 8: 結果のプロット

実験完了後、以下で結果グラフを生成できます:

```bash
# 例: シナリオ0 の学習曲線（論文 Figure 3 相当）
python plot_results.py 0

# 推論フェーズの結果
python plot_trained_results.py

# KBRL の調整率
python plot_adjustment_results.py

# KBRL の精度
python plot_accuracy_results.py
```

---

## まとめ：クイックスタート（最短手順）

```bash
# 1. uv インストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. リポジトリに移動
cd network-slicing

# 3. 仮想環境作成＆有効化
uv venv .venv --python 3.10
source .venv/bin/activate

# 4. 依存パッケージインストール
uv pip install numpy pandas scipy matplotlib gymnasium stable-baselines3

# 5. gym 環境インストール
cd gym-ran_slice && uv pip install -e . && cd ..

# 6. 実験実行
python experiments_kbrl.py
```

---

## トラブルシューティング

### `ModuleNotFoundError: No module named 'gym_ran_slice'`

→ Step 4 の `gym-ran_slice` のインストールが完了していない可能性があります。
仮想環境が有効化された状態で再実行してください:

```bash
cd gym-ran_slice && uv pip install -e . && cd ..
```

### `FileNotFoundError: ./datasets/fading_trace_EPA_3kmph.csv`

→ リポジトリのルートディレクトリから実行してください（`cd network-slicing` 後に実行）。

### 並列実行がハングする

→ `experiments_kbrl.py` 内の `PROCESSES` の値を CPU コア数以下に下げてください:

```python
PROCESSES = 4  # デフォルトの 10 から減らす
```

### Python バージョンエラー

→ `uv python list` で利用可能なバージョンを確認し、インストールします:

```bash
uv python install 3.10
uv venv .venv --python 3.10
```
