
# openpiの環境構築
## フォルダ構成
ICRA2026/
├── airoa-evaluation-ICRA/
└── openpi-icra2026/

## 環境構築手順
``` bash
cd ICRA2026
git clone https://github.com/LTS-AnalyticsTeam/openpi-icra2026.git
cd openpi-icra2026
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 学習
``` bash
# 何度も同じデータセットをダウンロードしないように、環境変数でデータセットの保存先を指定することができる
export HF_LEROBOT_HOME=/srv/shared/ICRA2026/datasets
# トレーニングを始める前に統計量の計算が必要（このとき学習データが指定位置にない場合は、Hugging Face Hubから自動的にダウンロードされる）
uv run scripts/compute_norm_stats.py --config-name pi0_hsr --max-frames 200000

# トレーニング開始
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_hsr --exp-name=my_experiment --overwrite
```

TrainingConfigで指定した`--config-name repo_id`のデータが存在しない場合、Hugging Face Hubから自動的にダウンロードされるが、データ形式が下記のように加工去れた状態で保存される。
```
# LeRobotの下記スクリプトで定義される学習時の形式
# lerobot/common/datasets/lerobot_dataset.py
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...
```

HSRデータセットの学習向けに整備済みのTrainingConfig一覧
- `pi0_hsr`