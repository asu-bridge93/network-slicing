# Network Slicing Environment — コードベース解説ガイド

> 本ドキュメントは、`network-slicing` リポジトリを初めて触る方向けに、
> コード全体の構造・各ファイルの役割・処理の流れを日本語でわかりやすくまとめたものです。

---

## 0. 研究初心者のための背景知識

> このリポジトリをはじめて読む方は、まずここを読んでください。
> 「何を研究しているのか」「なぜ強化学習を使うのか」を直感的に理解できるよう解説します。

---

### 0.1 この研究が解いている問題を一言で言うと？

> **「限られた電波リソースを、複数のサービスに対してうまく分け合わせる」**

携帯電話基地局には **物理リソースブロック（PRB）** という「電波の枠」があります。この枠は有限で、接続しているすべてのユーザが共有します。この枠を複数の「スライス（サービスグループ）」にどう配分するかがこの研究のテーマです。

```
基地局の電波リソース（総PRB = 100枠）
├── スライスA（動画配信ユーザ向け）→ 何枠割り当てる？
├── スライスB（ゲームユーザ向け）  → 何枠割り当てる？
└── スライスC（IoTセンサ向け）    → 何枠割り当てる？
    合計が100枠を超えてはいけない！
```

---

### 0.2 キーワード解説

| 用語 | 意味 |
|---|---|
| **PRB** (Physical Resource Block) | LTE/5G無線通信における最小リソース単位。「電波の1マス」のイメージ |
| **スライス** (Network Slice) | 同じ物理インフラを論理的に分割したサービス単位。例：動画用・IoT用・音声用 |
| **eMBB** (enhanced Mobile Broadband) | スマホの動画・Web閲覧など高速通信向けのスライス種別 |
| **mMTC** (massive Machine Type Communication) | IoTセンサなど大量デバイス向けのスライス種別。低速だが接続数が多い |
| **SLA** (Service Level Agreement) | 「スループット XX Mbps 以上を保証する」などのサービス品質の約束事 |
| **SLA違反** | そのスライスで約束した品質が守れなかったこと |
| **SINR** (Signal to Interference and Noise Ratio) | 信号対干渉雑音比。値が高いほど通信状況が良い |
| **MCS** (Modulation and Coding Scheme) | SINRに応じて選ぶ変調・符号化方式。SINRが高いほど高効率なMCSが選べる |
| **エージェント** | 意思決定を行うプログラム（ここではリソース割当を決める学習器） |
| **観測（state）** | エージェントが見える情報（各スライスのトラフィック量・スループット等） |
| **行動（action）** | エージェントが決定すること（各スライスへのPRB割当数） |
| **報酬（reward）** | エージェントの行動の良し悪しを示す数値。SLA達成 → +、違反 → − |

---

### 0.3 「強化学習（RL）」をざっくり理解する

強化学習は「試行錯誤しながら上手な意思決定を学ぶ」仕組みです。

```
エージェント ──action──▶ 環境（基地局シミュレータ）
     ▲                        │
     └─────── state + reward ──┘

・state  : 今どんな状況か（各スライスのトラフィック、SINRなど）
・action : 今回の割り当て（各スライスへのPRB数）
・reward : 結果の評価
    SLA達成 → 余った枠の数だけプラス
    SLA違反 → 違反数 × ペナルティ分マイナス
```

学習を重ねるにつれ、エージェントは「どの状況でどう割り当てれば違反せず枠も節約できるか」を自然に学習します。

---

### 0.4 このリポジトリで使われているアルゴリズムの解説

#### 🔵 KBRL（Kernel Model-Based Reinforcement Learning）— 主提案手法

**一言で言うと:** 「過去の経験から基地局の動きを予測するモデルを作り、そのモデルを使って賢く割り当てる」

```
通常のRL（モデルフリー）：
  実際に試す → 結果を見て学ぶ（試行錯誤に多くのステップが必要）

KBRL（モデルベース）：
  実際に試す → 「次回はどうなるか」のモデルを作る → モデルで先読みして賢く決定
  → 少ないステップで効率良く学べる！
```

**カーネル法とは？**  
「過去に似た状況（状態）があったなら、今回も同じような結果になるはず」という考えに基づいて、状態間の「似ている度合い」を数値化する数学的手法です。RBF（放射基底関数）カーネルがよく使われます。

**Projectron とは？**  
カーネル法では「全ての過去データを記憶」すると計算が重くなります。Projectronは「代表的な例だけを辞書として残す」スパース学習アルゴリズムで、計算量を抑えながら精度を維持します。

---

#### 🟢 DQN（Deep Q-Network）

**一言で言うと:** 「ニューラルネットワークで『この状態でこの行動を取ったらどれだけ得か』を学ぶ」

強化学習の代表的アルゴリズムで、Google DeepMindがAtariゲームで人間を超えた手法です。行動価値関数 Q(s,a) をDNNで近似します。本リポジトリでは `stable-baselines3` の実装を使います。

---

#### 🟡 PPO（Proximal Policy Optimization）

**一言で言うと:** 「方針（policy）を少しずつ安全に更新しながら学ぶ」

行動方針を直接学習するアクター・クリティック系の手法です。DQNより安定して学習できることが多く、連続行動空間にも対応しています。OpenAIが提案した手法で、実用的なRLの標準手法の一つです。

---

#### 🟠 NAF（Normalized Advantage Function）

**一言で言うと:** 「連続値の行動空間でQ学習を行う」

通常のDQNは離散的な行動しか扱えません。NAFは「スライスAに53.7枠」のような連続値の行動を直接扱える手法です。`keras-rl` ライブラリで実装されています。

---

#### 🔴 BQR / QR（Binary/Quantile Regression）

**一言で言うと:** 「SLA達成・不達成を2値分類として学ぶ」

「このPRB数を割り当てたらSLAを達成できるか？（Yes/No）」という問いに答えるモデルを学習し、Yesになる最小のPRB数を行動として選びます。KBRLと同じく独自実装のモデルベース手法です。

---

### 0.5 シミュレーションが1ステップで何をしているか

```
【1ステップ = slots_per_step 個のスロット（デフォルト100スロット = 100ms）】

エージェントが action を決定（各スライスへのPRB数）
      ↓
NodeB.step(action) を呼ぶ
  ↓ 100回ループ（1スロット = 1ms ずつ）
  ├─ ユーザの到着・離脱（ポアソン過程）
  ├─ トラフィック生成（パケット到着）
  ├─ SNR/SINRの計算（チャネルモデル）
  ├─ スケジューラがPRBをUEに割当
  └─ パケット受信成否の判定
      ↓
100スロット分の積算結果からSLAを評価
      ↓
  SLA達成 → reward = 余剰PRB数
  SLA違反 → reward = −penalty × 違反数
      ↓
エージェントが次のactionを学習・決定
```

---


## 1. このリポジトリが何をしているか

無線基地局（eNB / gNB）で複数の「ネットワークスライス」に対して **無線リソース（PRB）を割り当てる** 強化学習（RL）の研究コードです。

論文: **"Model-Based Reinforcement Learning with Kernels for Resource Allocation in RAN Slices"**
(IEEE Transactions on Wireless Communications, 2023)

### 基本コンセプト

```
                      エージェント（RL）
                      ↓ action: 各スライスへのPRB数を決定
   ┌─────────────────────────────────────────────────────┐
   │              NodeB（基地局シミュレータ）              │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
   │  │Slice L1-0│  │Slice L1-1│  │Slice L1-2│   ...     │
   │  │(eMBB)    │  │(eMBB)    │  │(mMTC)    │           │
   │  └──────────┘  └──────────┘  └──────────┘           │
   └─────────────────────────────────────────────────────┘
                      ↑ state（現在の状態）+ reward（SLA達成度）
```

- **PRB (Physical Resource Block)**: LTE/5Gの最小無線リソース単位
- **SLA (Service Level Agreement)**: スライスごとに定められたスループット・遅延の品質保証条件
- **スライス**: eMBB（高速通信）または mMTC（大量デバイス通信）の2種類
- **ゴール**: SLA違反を最小化しながら、使用PRBを最小化する

---

## 2. ディレクトリ構成

```
network-slicing/
│
├── gym-ran_slice/              # OpenAI Gym 互換の環境パッケージ
│   ├── gym_ran_slice/
│   │   ├── __init__.py
│   │   └── ran_slice.py        # Gym環境クラス (RanSlice)
│   └── setup.py
│
├── algorithms/                 # KBRLアルゴリズムの実装
│   ├── kernel.py               # カーネル関数
│   └── projectron.py           # Projectronアルゴリズム
│
├── datasets/                   # フェージングトレースCSV、MCSテーブルCSV
│   ├── fading_trace_EPA_3kmph.csv
│   ├── mcs_codeset.csv
│   └── srslte_v19.03.csv
│
├── results/                    # 実験結果保存ディレクトリ
├── figures/                    # 論文の図
├── trained_models/             # 学習済みモデルの保存先
│
│ ── 環境コア層（シミュレータ） ──────────────────────────
├── node_b.py                   # 基地局（NodeB）クラス
├── slice_l1.py                 # L1スライス（PRB割当・スケジューリング）
├── slice_ran.py                # RANスライス（ユーザ到着・離脱）
├── channel_models.py           # 無線チャネルモデル
├── traffic_generators.py       # トラフィック生成
├── schedulers.py               # スケジューラ
│
│ ── 実験構築層 ──────────────────────────────────────────
├── scenario_creator.py         # シナリオ・環境の生成ファクトリ
├── kbrl_scenario_creator.py    # KBRLエージェント生成
├── qr_scenario_creator.py      # QRエージェント生成
├── bqr_scenario_creator.py     # BQRエージェント生成
├── naf_agent_creator.py        # NAFエージェント生成
├── wrapper.py                  # Stable-Baselines向けラッパー
├── wrapper_sppo.py             # SPPO向けラッパー
│
│ ── エージェント・アルゴリズム ──────────────────────────
├── kbrl_control.py             # KBRL（主提案手法）
├── qr_control.py               # QR手法
├── bqr_control.py              # BQR手法
├── svgp_control.py             # SVGP手法
├── ActorCritic.py              # Actor-Criticネットワーク
├── PPO_Safe.py                 # Safe PPO実装
│
│ ── 実験スクリプト ──────────────────────────────────────
├── experiments_kbrl.py         # KBRL実験
├── experiments_rl.py           # stable-baselines RL実験
├── experiments_dqn.py          # DQN実験
├── experiments_naf.py          # NAF実験
├── experiments_bqr.py          # BQR実験
├── experiment_sppo.py          # Safe PPO実験
│
│ ── 結果プロット ─────────────────────────────────────────
├── plot_results.py             # 学習曲線のプロット
├── plot_trained_results.py     # 推論フェーズの結果プロット
├── plot_adjustment_results.py  # KBRL調整率のプロット
├── plot_accuracy_results.py    # KBRL精度のプロット
├── plot_oracle_results.py      # Oracle比較プロット
└── plot_slices.py              # スライス別プロット
```

---

## 3. 環境コア層の詳細

### 3.1 `channel_models.py` — 無線チャネルモデル

ユーザの位置をランダムに生成し、電波伝搬モデルに基づいてSINR（信号対干渉雑音比）を計算します。

| クラス / 関数 | 役割 |
|---|---|
| `macro_cell(rng)` | 3GPP TS 36.942 準拠のマクロセルパスロスモデルでSINRを計算 |
| `free_space(rng)` | 自由空間伝搬モデルでSINRを計算 |
| `NominalSINR` | 上記の伝搬関数をラップするクラス（都市2GHz / 都市900MHz / 農村の3モード対応） |
| `SINRSelectiveFading` | CSVファイルの実測フェージングトレースを使いSINR時系列を生成するクラス |
| `SNRGenerator` | srslte から取得した実測SNRデータを用いたSNR生成クラス |
| `MCSCodeset` | MCSテーブルを読み込み、SNRに応じた変調・符号化方式の選択、スループットの計算を行うクラス |

**処理の流れ（`SINRSelectiveFading`）:**
```
1. CSV（fading_trace_EPA_3kmph.csv等）からフェージングパターンを読み込む
2. ユーザが挿入されると、ランダムな位置・名目SINR・フェージング列が割り当てられる
3. get_snr(user_id) でスロットごとにフェージング列を1ステップ進め、SINR配列を返す
```

---

### 3.2 `traffic_generators.py` — トラフィック生成

スライス内のUEが発生させるビット列（パケット到着過程）を実装します。

| クラス | 種別 | 説明 |
|---|---|---|
| `PeriodicSource` | 周期的 | 一定周期でパケットを送信 |
| `OnOffSource` | On/Off | On/Off 2状態を幾何分布で切り替え |
| `CbrSource` | CBR（固定ビットレート） | PeriodicSourceを継承、bpsで指定 |
| `VbrSource` | VBR（可変ビットレート） | バースト到着過程（ポアソン過程）をシミュレート |

---

### 3.3 `slice_ran.py` — RANスライス（ユーザ管理）

スライスレベルでのUEの到着・離脱を管理します。

| クラス | 種別 | 説明 |
|---|---|---|
| `UE` | ユーザ端末 | SNR推定、バッファ管理、スループット計算を担う |
| `MTCdevice` | MTCデバイス | 繰り返し送信数を持つシンプルなデバイス |
| `SliceRANeMBB` | eMBBスライス | CBR/VBRユーザの到着（ポアソン過程）・CAC・離脱を管理 |
| `SliceRANmMTC` | mMTCスライス | 多数のMTCデバイスの周期的メッセージ到着をシミュレート |

**eMBBスライスの動作（1スロット）:**
```
slot() 呼び出し
  ├── cbr_arrivals(): ポアソン過程でCBRユーザが到着 → CAC（許可制御）でフィルタ
  └── departures(): 保持時間が尽きたユーザを離脱させる
```

---

### 3.4 `slice_l1.py` — L1スライス（リソース割当・スケジューリング）

`slice_ran.py` の上に乗り、実際の **PRB割当とMCS選択** を行うレイヤーです。

- **スケジューラ（`schedulers.py` から利用）** を呼び出してPRB割当を実施
- 各スロットで `slice_ran.py` からUEの到着・離脱情報を受け取る
- チャネルモデルへSINR取得をリクエスト
- SLA（スループット、PRB使用率など）の評価を行い報酬計算に利用する

---

### 3.5 `schedulers.py` — スケジューラ

L1スライスが各UEへPRBを割り当てる具体的なアルゴリズムを実装します。

主なスケジューラ:
- **比例公平 (Proportional Fair)**: 各UEのスループットの公平性を考慮して割当
- **ラウンドロビン**: 順番に割当
- **GBR保証型**: GBRトラフィックを優先的に割当

---

### 3.6 `node_b.py` — 基地局（NodeB）

全てのスライスを束ねる **基地局クラス** です。ここが環境のトップレベルコントローラとなります。

```python
class NodeB:
    def reset()        # 全スライスをリセット、初期状態を返す
    def step(action)   # action（各スライスへのPRB数配列）で1ステップ進める
    def get_state()    # 全スライスの状態ベクトルを結合して返す
    def compute_reward() # 各スライスのSLA達成状況を評価
```

**`step()` の処理フロー:**
```
action = [prb_slice0, prb_slice1, ...]
1. 各スライスにPRB数を設定 (set_prbs)
2. slots_per_step 回スロットをシミュレート
3. 全スライスの状態を取得 (get_state)
4. SLA達成状況を評価し SLA_labels, violations を返す
```

---

### 3.7 `gym-ran_slice/gym_ran_slice/ran_slice.py` — OpenAI Gym 環境

`NodeB` を **OpenAI Gymnasium** のインターフェースでラップします。

- `action_space`: 各スライスへのPRB割当（0〜n_prbs の整数ベクトル）
- `observation_space`: 全スライスの状態変数（連続値ベクトル）
- **報酬関数**:
  - SLA違反あり → `-penalty × 違反数`（負の報酬）
  - SLA達成 → `+余剰PRB数`（正の報酬）

---

## 4. 実験構築層の詳細

### 4.1 `scenario_creator.py` — シナリオ生成

実験で使う環境（NodeB + Gym環境）を作るファクトリ関数群です。

4つのシナリオが定義されており、それぞれ PRB数・eMBBスライス数・mMTCスライス数が異なります:

| シナリオ | PRB数 | eMBBスライス | mMTCスライス |
|---------|-------|------------|------------|
| 0       | 100   | 3          | 0          |
| 1       | 150   | 3          | 2          |
| 2       | 100   | 1          | 4          |
| 3       | 70    | 1          | 1          |

---

### 4.2 `wrapper.py` — Stable-Baselines ラッパー

DQN などの Stable-Baselines3 エージェントが扱える形式に環境を変換し、学習結果の記録も行います。

---

## 5. アルゴリズム層の詳細

### 5.1 KBRL（主提案手法）— `kbrl_control.py` + `algorithms/`

**Kernel Model-Based Reinforcement Learning** の実装です。

| ファイル | 役割 |
|---|---|
| `kbrl_control.py` | KBRLエージェントの本体。モデル学習・探索・制御のループ |
| `algorithms/kernel.py` | RBFカーネル関数（状態間の類似度計算） |
| `algorithms/projectron.py` | Projectronアルゴリズム（スパースカーネルモデル） |

**KBRLの動作原理:**
```
1. 各(状態, アクション) → カーネルでマッピング
2. Projectronで スパースなモデル（辞書ベース）を逐次更新
3. 学習したモデルを使って次状態を予測 → 最良アクションを選択
4. accuracy パラメータでモデルの精度と計算コストのトレードオフを調整
```

---

### 5.2 その他のアルゴリズム

| ファイル | アルゴリズム | ライブラリ |
|---|---|---|
| `experiments_rl.py` | PPO, A2C, TRPO など | stable-baselines |
| `experiments_dqn.py` | DQN | stable-baselines3 |
| `experiments_naf.py` | NAF (Normalized Advantage Function) | keras-rl |
| `experiment_sppo.py` | Safe PPO | 独自実装 |

### 5.3 BQR / QR（Binary / Quantile Regression）の実装

分位点回帰（Quantile Regression = QR）およびバイナリ分位点回帰（BQR）の実装は、ファイル名が `qr_` や `bqr_` で始まるモジュール群に分かれて独自実装されています。モデルの「学習・予測」の数学的実体は `qr_esn.py` などのモデルファイル内に、その予測を「どう基地局のPRB割り当てに活かすか」の強化学習的（バンディット的）なロジックは `qr_control.py` に記述されているという構造です。

| ファイル | 役割 |
|---|---|
| `qr_control.py` | **QRエージェント（コントローラ）のメインロジック。**<br>推論結果（分位予測と不確実性）に基づき、マージンを持たせたリソース割当や、K近傍法（KNN）を利用した安全な最適アクション（Safe Action）の選定などを行います。 |
| `qr_dqrrn.py` | **Deep Quantile Regression Neural Network**<br>ディープラーニングとピンボールロス関数を組み合わせた分位点回帰予測モデルの実装です。 |
| `qr_esn.py` | **Quantile Regression Echo State Network**<br>RNNの一種であるリザバーコンピューティング（ESN）を利用した時系列予測に対応する分位点回帰モデルの実装です。 |
| `qr_scenario_creator.py` | シミュレーション環境のパラメータから、特定のQR予測モデルを組み込んでQRエージェントを構成・初期化するファクトリ（生成用スクリプト）として機能します。 |
| `bqr_control.py` など | SLAの「達成・未達成」を2値（バイナリ）で分類・確率予測し、安全な行動を行うための類似手法（Binary Quantile Regression）のスクリプト群です。 |
| `experiments_qr.py` | 上記QRエージェントおよびシミュレータを呼び出して、学習と評価を実行する検証用スクリプトです。 |

---

## 6. 実験スクリプトの実行フロー

```
experiments_kbrl.py（例）
│
├── create_env(rng, scenario)      ← scenario_creator.py
│   └── NodeB + RanSlice を構築
│
├── create_kbrl_agent(rng, ...)    ← kbrl_scenario_creator.py
│   └── KBRLControl オブジェクトを構築
│
└── kbrl_agent.run(node_env, STEPS)
    ├── ステップごとにアクションを選択
    ├── env.step(action) を呼ぶ
    └── 結果を ./results/scenario_N/KBRL_97/results_K.npz に保存

最後に：
plot_results.py でグラフを生成
```

---

## 7. データフロー図

```
[CSV フェージングデータ]       [CSV MCSテーブル]
        │                            │
        ▼                            ▼
  SINRSelectiveFading          MCSCodeset
        │                            │
        └──────────┬─────────────────┘
                   ▼
              SliceL1 (PRB割当)
              ├── Scheduler（スケジューラ）
              └── SliceRAN（UE管理）
                   ├── UE（トラフィック生成）
                   └── SliceRANeMBB / mMTC
                   ▼
              NodeB（基地局）
                   ▼
              RanSlice（Gym環境）
                   ▼
          エージェント（KBRL / DQN / PPO 等）
```

---

## 8. 主要パラメータ一覧

| パラメータ | 説明 | 典型値 |
|---|---|---|
| `n_prbs` | 基地局の総PRB数 | 100 |
| `slots_per_step` | 1ステップあたりのスロット数 | 100 |
| `penalty` | SLA違反時のペナルティ係数 | 10〜1000 |
| `RUNS` | 実験の繰り返し回数 | 30 |
| `TRAIN_STEPS` | 学習ステップ数 | 10240〜50000 |
| `accuracy` (KBRL) | KBRLのモデル精度パラメータ | 0.97〜0.999 |
| `SEED` | 乱数シード（再現性確保） | 実行番号i を使用 |

---

## 9. 結果の保存形式

実験結果は以下のパスに `.npz` 形式（NumPy圧縮アーカイブ）で保存されます:

```
./results/scenario_{N}/{アルゴリズム名}/results_{K}.npz
```

- `N`: シナリオ番号（0〜3）
- `K`: 実行番号（0〜29）

プロットスクリプトはこれらのファイルを読み込み、matplotlib で学習曲線・SLA違反率などをグラフ化します。

---

## 10. リソース逼迫時の一律削減ロジック（`adjust_action`）

各エージェントがスライスへのPRB割当を決定した後、**各スライスへの要求PRB数の合計が基地局の総PRB数 `n_prbs` を超えた場合**、`adjust_action()` メソッドが呼び出されてアクションを強制縮小します。

### 該当ファイルと呼び出し箇所

| ファイル | クラス | 呼び出し箇所 |
|---|---|---|
| `kbrl_control.py` | `KBRL_Control` | `select_action()` L.67–70 |
| `bqr_control.py` | `BQR_Control` | `select_action()` L.65–67 |
| `qr_control.py` | `QR_Control` | `select_action()` L.207–209 |

### 共通のロジック（比例縮小）

```python
# 逼迫判定（kbrl_control.py L.66–70 を例に）
assigned_prbs = action.sum()
if assigned_prbs > self.n_prbs:   # 要求合計が総PRBを超えた場合
    action = self.adjust_action(action, assigned_prbs)

# adjust_action の本体（全アルゴリズム共通の考え方）
relative_p = action / assigned_prbs   # 各スライスの要求比率を計算
new_action  = floor(n_prbs * relative_p)  # 比率を保ったまま総PRBにスケールダウン
```

**具体例:**
```
要求      = [60, 50, 40]   合計 = 150（総PRB 100 を超過）
比率      = [0.40, 0.33, 0.27]
スケール後 = [40,  33,  27]    合計 = 100
```

→ 要求の大きいスライスほど削減量も大きくなります（比例縮小）。

### KBRL と BQR / QR の実装の違い

| 実装 | floor 後の端数処理 | 合計の保証 |
|---|---|---|
| **KBRL** (`kbrl_control.py`) | なし（切り捨てのみ） | 合計が `n_prbs` を**下回ることがある** |
| **BQR** (`bqr_control.py`) | 端数PRBを先頭スライスから順に +1 で補填 | 合計が厳密に `n_prbs` になる |
| **QR** (`qr_control.py`) | 同上 | 合計が厳密に `n_prbs` になる |

```python
# BQR / QR の端数補填処理（bqr_control.py L.75–77）
remainder = self.n_prbs - new_action.sum()
for i in range(int(remainder)):
    new_action[i % self.n_slices] += 1   # スライス0, 1, 2, ... の順に1つずつ加算
```

### `adjusted` フラグについて（KBRL のみ）

KBRL では `adjust_action` が実行されたステップを `self.adjusted = 1` として記録し、**そのステップではセキュリティマージン（`security_factors`）の更新を行わない**ようになっています（`update_control()` L.100）。これは、強制縮小されたアクションを学習データとして使うとモデルが歪む恐れがあるための保護措置です。

---

## 11. SLA計算ロジック（SLA Violation の判定）

各スライスのSLA（Service Level Agreement：通信品質の保証条件）を守れたかどうかの判定（`compute_reward`での評価）は、L1レベル（主に `slice_ran.py`）で各スロット単位での通信状況を集計した結果をもとに行われます。

### mMTC スライスの基準
mMTCスライスでは **「平均遅延（Delay）」** がSLAの基準となります。
- 1ステップ（デフォルト100スロット）間の全デバイスの平均遅延が、SLAで定められた上限値 `SLA['delay']` より**小さい（短い）**かどうかで判定されます。
- `SLA_fulfilled = self.info['delay']/self.slots_per_step < self.SLA['delay']`
- 遅延が上限を下回っていればSLA達成、上回っていればSLA違反となります。

### eMBB スライスの基準
eMBBスライスでは **「平均スループット（Throughput）」** がSLAの基準となっています。
- 1ステップ間に送信できた合計ビット数に基づくスループット `cbr_th` が、SLAで定められた下限値 `SLA['cbr_th']` を**上回っている**かどうかで判定されます。
- `cbr_th = self.info['cbr_th']/self.observation_time > self.SLA['cbr_th']`
- コード上では、各ユーザのパケットキュー長（`cbr_queue`）や利用PRB数（`cbr_prb`）も計算はされていますが、最終的なSLA達成フラグ（`SLA_fulfilled`）にはスループットの条件のみが使われています。

これらの判定結果が各ステップの終わりに `node_b.py` に集約され、SLA違反があった場合には違反数に応じた「マイナスのペナルティ（負の報酬 `reward`）」としてエージェントの学習フィードバックに利用されます。

---

## 12. 結果可視化スクリプト（plot_*.py）の役割

各種実験で得られた `.npz` のログデータを可視化・比較するためのスクリプト群です。目的や見たい指標に応じて細かく使い分けられます。

| スクリプト名 | 役割・可視化される内容 | 出力先 |
|---|---|---|
| **`plot_results.py`** | **全体の性能推移（報酬・PRB割当数・SLA違反数）** を時間軸に沿って可視化します。<br>`python plot_results.py 0` のようにシナリオ番号を渡して実行します。複数アルゴリズムの収束の速さや時間的安定性を並べて比べるための最もメインのスクリプトです。 | `./figures/` 内 |
| **`plot_trained_results.py`** | **「リソース消費量（横軸）」と「SLA違反数（縦軸）」のトレードオフ（パレート図）** を描画します。<br>学習後半の収束後のデータ点だけを全シナリオから抽出し、「どのアルゴリズムが一番少ないPRB通信枠で安全にSLAを守れているか」を点とエラーバーで比較・評価します。 | `./figures/trained_figure.png` |
| **`plot_slices.py`** | 基地局全体ではなく、**スライス個別での詳細な性能** を見るためのスクリプトです。<br>「eMBB単体の処理量」「mMTC単体の遅延」など、スライスごとに切り分けて要求を満たせているかを深く分析します。 | `./figures/` 内 |
| **`plot_adjustment_results.py`** | **PRB強制調整（Adjustment）の発生率** の推移をプロットします。<br>AIエージェントが基地局の上限を超えた無理なPRB枠を要求し、シミュレータに強制的に削られた割合がどれくらいかを評価します（0に近いほど優秀）。 | `./figures/adjustments.png` |
| **`plot_accuracy_results.py`** | **内部モデルの予測精度（Accuracy / Hit率）** の向上推移をプロットします。<br>KBRLなど、シミュレーション内部で「将来のSLA達成率」を推論する予測モデルを持っているアルゴリズムの純粋な学習性能分析用です。 | `./figures/accuracies.png` |
| **`plot_oracle_results.py`** | 未来のトラフィックを完璧に先読みできる**「オラクル最適解（理想的な上限）」と各アルゴリズムを厳密に比較する** プロットを行います。 | `./figures/` 内 |

---

## 13. QRアルゴリズムのパラメータ「cost / nocost」について

実験結果のディレクトリ名（例：`QR_15_0_0_2_01_nocost` や `QR_15_0_3_2_01_cost`）の末尾につく `_cost` と `_nocost` は、QR（分位点回帰）エージェントの内部計算である **「リソース（PRB）要求量に対するコスト（ペナルティ）の有無」** を意味しています。

コード内部（主となる `qr_control.py` 内など）でのアクション（要求PRB量）のスコア決定時は、以下のようにスコア付けされます。

> `最終評価スコア = (SLA到達期待度) − (リソースコスト係数 × 基地局の何割のPRBを要求しているか)`

*   **`_nocost`（リソースコスト係数：`0`）の場合**:
    どれだけ余分に通信枠（PRB）を要求したとしても、スコア上のペナルティが発生しません。つまり「SLAさえ守れれば、コスト度外視でリソースを確保しておく」という強気な学習になります。SLA違反は起きにくく安全な反面、インフラを浪費しやすくなります。
*   **`_cost`（リソースコスト係数：`0.3`など）の場合**:
    要求するPRBが多いほど、最終評価スコアから罰金が引かれます。「SLAはなんとか守りつつも、極力少ないPRB数に抑えてギリギリで通信を済ませる（余った分を他に譲る）」という全体最適や省エネを重視した振る舞いを獲得するエージェントになります。

---

## 14. 参考文献

- [論文 (IEEE TWC)](https://doi.org/10.1109/TWC.2022.3195570)
- [GitHub リポジトリ](https://github.com/jjalcaraz-upct/network-slicing/)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
