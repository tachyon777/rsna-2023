# RSNA 2023 Abdominal Trauma Detection

kaggleの[RSNA2023コンペ](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)の実験管理用リポジトリ。

# Prerequisites
必ずしも統一する必要はありません。
- python 3.9.12
- pytorch 1.11.0
- その他機械学習に用いられる一般的なライブラリを必要に応じてinstall
- 特別必要に思われるモジュールはrequirements.txtに記載

# Getting Started
dataディレクトリ直下に`rsna-2023-abdominal-trauma-detection`というディレクトリを配置し、その中にデータセットを配置してください。
なお、このディレクトリは直接学習には使用しないかつデータ量が大きいので、外付けHDDなどに配置の上シンボリックリンクを作成することをオススメします。  
例：
```
.
└── data
    └── rsna-2023-abdominal-trauma-detection
        ├── image_level_labels.csv
        ├── sample_submission.csv
        ├── segmentations
        ├── test_dicom_tags.parquet
        ├── test_images
        ├── test_series_meta.csv
        ├── train.csv
        ├── train_dicom_tags.parquet
        ├── train_images
        └── train_series_meta.csv
```
学習用データは別に整形して用意し、dataディレクトリ配下に作成します。学習用データは再現できるようにコードをnotebookに配置しておきます。

## ファイル命名規則
**ファイル名に意味を含めない**こと。[Notionの実験管理ページ](https://www.notion.so/RSNA2023-2c6adfcd57594dd69532a2ff347d55fe?pvs=4)と紐付けて内容を管理する。
深層学習を行うファイル名：
exp000_train.ipynb
exp000_eval.ipynb
その他探索的データ分析・実験・検証を行うファイル名：
exp000.ipynb