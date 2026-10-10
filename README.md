# onset-grid-aligner

音源のオンセット（アタックの頭）が拍グリッドに対して何msずれているかを推定し、0ms開始の音源を何ms移動すればグリッドに揃うかを解析・推定するPython CLIツールです。

## 主な用途

- 音源（ドラムステム）のオンセットがグリッドより前後どちらに何msずれているかを推定
- BPMが時間経過で変化する音源のドリフトを解析
- 複数の検出器・帯域・因果フィルタを用いて結果を検算
- 最終的に「音源を何ms遅らせればよいか」を表示

## 必要環境

- Python 3.10+
- NumPy
- SciPy
- SoundFile
- FFmpeg (mp3/wmaを扱う場合)

## インストール

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

python -m pip install -r requirements.txt
```

## 使い方

BPM を指定して解析 (--bpm):

```bash
python src/onset_grid_aligner.py input.wav --bpm 120
```

BPM を自動推定:

```bash
python src/onset_grid_aligner.py input.wav
```

主なオプション:

```text
--bpm BPM (BPMの指定)
--bpm-min BPM推定範囲の下限
--bpm-max BPM推定範囲の上限
--low-hz 周波数帯域の下限
--high-hz 周波数帯域の上限
--head-level オンセット頭の定義 {デフォルトは0.15}
--drift-window-sec 時間ドリフト解析の窓長
--reference-group 音群の選択
--no-band-check 帯域別検算を無効化
--no-refine 局所精密化を無効化
--filter-mode {zero,causal}
```

## 入力形式

直接読み込み:

- WAV
- FLAC
- AIFF / AIF
- OGG

FFmpeg 経由:

- MP3
- WMA

## 結果の読み方

出力の

```text
音源を遅らせる量 +X.XX ms
```

が、0 ms 開始の音源をグリッドに揃えるための推奨移動量です。

符号は次の意味です。

- 正: 音源を後ろへずらす (遅らせる)
- 負: 音源を前へずらす

## 注意

- 基本的にはドラムステムの音声ファイルの使用を前提としています。特に、生音源のアタックが弱いドラムを使用している場合や、音声ファイルがドラムステムではない場合は結果の信憑性が著しく落ちる可能性があります。
- 拍の分割・判定の仕様上、5連符や7連符には対応していません。
- 表示値の小数点以下すべてが測定精度を意味するわけではありません。
- オンセットの検出方法、周波数帯域、head-level、時間ドリフト、フィルタ方式、音源の構成によって結果が変化し得るため、出力される検算結果と推定誤差も確認してください。
- MP3はエンコーダ／デコーダ遅延の扱いが再生環境によって異なるため、本ツールは複数の遅延補正ケースを表示します。使用する再生環境に対応する結果を採用してください。
