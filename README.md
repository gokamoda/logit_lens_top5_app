# README

## Limitations
- ローカルで動かす前提（M1 macbook なら大きくてもGPT2-mediumくらい?）
  - よって、backendは複数アクセスする想定をしていない
- LogitLens部分のみ
  - Attentionの描画は一旦保留
- とにかく動く事優先
  - デザイン、コードの綺麗さは後回し

## Usage
1. クローン

1. 必要なライブラリのインストール
    ```
    pip install -r requirements.txt
    ```

1. バックエンド実行
    ```
    python backend.py
    ```

1. index.htmlをブラウザで開く(ファイルパスをアドレスバーに入力する)



