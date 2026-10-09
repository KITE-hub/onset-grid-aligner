# onset-grid-aligner

音源のオンセット（アタックの頭）が拍グリッドに対して何msずれているかを推定し、0 ms 開始の音源を何ms移動すればグリッドに揃うかを解析・推定するPython CLIツールです。

## 主な用途

- 音源(ドラムステム)のオンセットがグリッドより前後どちらに何 ms ずれているかを推定
- BPM が時間経過で変化する音源のドリフトを解析
- ハイハット、スネア、キックなどのオンセット群を帯域によって推定して分割して比較
- 複数の検出器・帯域・因果フィルタを使って結果を検算
- 最終的に「音源を何 ms 遅らせればよいか」を表示

## 必要環境

- Python 3.10+
- NumPy
- SciPy
- SoundFile
- FFmpeg (mp3/wma を扱う場合)

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

例: 1/4 拍単位ではなく 1/8 拍単位のグリッドで評価 (--subdivision):

```bash
python src/onset_grid_aligner.py input.wav --bpm 120 --subdivision 8
```

主なオプション:

```text
--bpm BPM (BPMの指定)
--bpm-min BPM推定範囲の下限
--bpm-max BPM推定範囲の上限
--subdivision {1,2,3,4,6,8, デフォルトは4}
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

ただし、グリッド間隔の剰余としてしか評価しないので、BPMや subdivisionの設定によって等価な位相が存在します。最終的な拍頭位置は DAW 上でも確認してください。

## 注意

- 表示値の小数点以下すべてが測定精度を意味するわけではありません。
- このツールはオンセットの検出方法、周波数帯域、head-level、時間ドリフト、フィルタ方式、音源の構成によって結果が変化し得るため、出力される検算結果と推定誤差も確認してください。
- 特に、生音源のアタックが弱いドラムを使用している場合や、音声ファイルがドラムステムではない場合は結果の信憑性が著しく落ちる可能性があります。
- MP3 はエンコーダ／デコーダ遅延の扱いが再生環境によって異なるため、本ツールは複数の遅延補正ケースを表示します。使用する再生環境に対応する結果を採用してください。
