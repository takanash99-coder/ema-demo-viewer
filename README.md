# EMA Demo Viewer Ver.0.2.2

EMA (Expert Motion Coaching AI) の完成イメージを体験するための Streamlit デモアプリです。

Ver.0.2.2 では、同じ Wi-Fi 内のアクセスに加えて、Cloudflare Tunnel を使った一時的な外部デモ公開手順と、Cloudflare Tunnel URL から QR コードを生成する機能を追加しています。


## Public Demo Repository

このリポジトリは EMA Demo Viewer の公開デモ版として、Streamlit Community Cloud での公開を想定しています。

- 実在する被験者データは含めません。
- `demo_data/public_sample` は匿名ダミーデータです。
- 実研究データはローカル環境のみで管理します。
- GitHub 公開時は `.gitignore` により動画、EMG、Excel、CSV、secrets などを除外します。

## Project Location

正式ルート:

```text
G:\マイドライブ\EMA_Project\03_Tools\ema_demo_viewer
```

現在のアプリ本体バージョンは Ver.0.2.2 です。

## 推奨端末

- iPad 横向き
- PC

スマホでも閲覧は可能ですが、EMA はスマホ最適化アプリではありません。教育・解析デモとしては iPad 横向きまたは PC での利用を推奨します。

## Run

PC のみで確認する場合:

```powershell
cd 03_Tools/ema_demo_viewer
streamlit run app.py
```

同じ Wi-Fi 内のスマホ・iPad から接続する場合:

```powershell
cd 03_Tools/ema_demo_viewer
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Folder Roles

`assets`:
将来のロゴ、デモ画像、動画、UI素材を格納する予定のフォルダです。現時点では空で問題ありません。

`data`:
将来のデモ用JSON、CSV、EMGダミーデータ、被験者設定データを格納する予定のフォルダです。現時点では空で問題ありません。

## スマホ・iPadから同じWi-Fiでアクセスする方法

1. PC とスマホ（iPad）を同じ Wi-Fi へ接続する。
2. EMA Demo Viewer を `streamlit run app.py --server.address 0.0.0.0 --server.port 8501` で起動する。
3. 画面上部の Network URL QR コードを読み取る。
4. または `http://<PCのIPアドレス>:8501` へアクセスする。

例:

```text
http://192.168.1.35:8501
```

## スマホ・iPadから大学Wi-Fiなしでアクセスする方法

Cloudflare Tunnel を使用します。これは本番公開ではなく、学内発表・デモ用途の一時的な外部公開手段です。

### 1. EMA Demo Viewerを起動

```powershell
cd 03_Tools/ema_demo_viewer
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 2. 別のターミナルでCloudflare Tunnelを起動

```powershell
cloudflared tunnel --url http://localhost:8501
```

### 3. 表示されたURLをスマホで開く

Cloudflare Tunnel を起動すると、次のような一時URLが表示されます。

```text
https://xxxx.trycloudflare.com
```

この URL をスマホや iPad のブラウザで開きます。

### 4. QRコードを使う場合

EMA Demo Viewer 画面上部の `External Demo URL` 入力欄に `https://xxxx.trycloudflare.com` を貼り付けると、外部公開URL用の QR コードが生成されます。スマホや iPad で読み取ってアクセスできます。


## 外部URLで閲覧する方法

PC とスマートフォン／iPad が同じ Wi-Fi に接続されていない場合は、Cloudflare Tunnel で一時的な外部URLを発行します。この方法は一時的なデモ確認用です。

### 1. Streamlitを起動する

PowerShell で正式ルートへ移動し、EMA Demo Viewer を起動します。

```powershell
cd G:\マイドライブ\EMA_Project\03_Tools\ema_demo_viewer
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 2. cloudflaredを起動する

別のPowerShellまたはターミナルを開き、以下を実行します。

```powershell
cloudflared tunnel --url http://localhost:8501
```

### 3. trycloudflare.com URLを確認する

cloudflared の出力に、以下のようなURLが表示されます。

```text
https://xxxx.trycloudflare.com
```

このURLは恒久URLではなく、cloudflaredを実行している間だけ使える一時URLです。

### 4. スマートフォンまたはiPadで開く

スマートフォン／iPad はPCと同じWi-Fiである必要はありません。発行された `https://xxxx.trycloudflare.com` をスマートフォン／iPadのブラウザで開きます。

### 5. External Demo URL欄へ貼り付ける

EMA Demo Viewer 画面上部の `External Demo URL` 入力欄に、発行された `https://xxxx.trycloudflare.com` を貼り付けます。

### 6. QRコードを生成する

`External Demo URL` 欄へURLを貼り付けると、そのURL用のQRコードが生成されます。スマートフォン／iPadでQRコードを読み取ると、外部URLへアクセスできます。

### 7. 終了方法

cloudflared を停止するには、cloudflared を実行しているターミナルで `Ctrl + C` を押します。

Streamlit を停止するには、Streamlit を実行しているターミナルで `Ctrl + C` を押します。

### セキュリティ上の注意

- URLを知っている人はアクセスできます。
- URLは一時的なものです。
- cloudflaredを停止するとアクセスできなくなります。
- 実在する患者情報、学生個人情報、未公開研究データは使用しないでください。
- 現段階ではデモ用ダミーデータのみを使用してください。
- 発表や確認終了後は必ずCloudflare Tunnelを停止してください。

## cloudflared のインストール方法（Windows）

### 方法A: 公式サイトからダウンロード

Cloudflare 公式サイトから `cloudflared.exe` をダウンロードし、PowerShell から実行できる場所に配置します。

### 方法B: winget を使う

```powershell
winget install --id Cloudflare.cloudflared
```

### インストール確認

```powershell
cloudflared --version
```

## Cloudflare Tunnel 使用時の注意

- Cloudflare Tunnel は一時的に外部公開するため、URLを知っている人はアクセスできます。
- この URL は一時的なデモURLです。
- PCで Streamlit と cloudflared が起動している間だけ有効です。
- URL は毎回変わる可能性があります。
- 実在の患者情報、学生個人情報、未公開研究データは表示しないでください。
- デモ用のダミーデータのみで使用してください。
- 学内発表・デモ用途に限定してください。
- 発表後は Cloudflare Tunnel を停止してください。
- cloudflared を起動しているターミナルで `Ctrl + C` を押すと停止できます。

## LAN接続時の注意

- 異なるネットワークからは LAN URL では接続できません。
- Windows Defender Firewall の許可が必要な場合があります。
- 推奨端末は iPad 横向きです。

## Current Limitations

- カメラ機能はブラウザ、OS、端末、権限設定により制限される場合があります。
- 現時点では本物の AI 解析、OpenPose、MediaPipe、EMG 同期、リアルタイム音声ガイドは未実装です。
- Motion Capture Overlay、Motion Report、AI Coaching はデモ表示です。
- `st.camera_input` はライブ解析ではなく、ブラウザ経由のカメラキャプチャです。
- Cloudflare Tunnel はデモ用の一時公開手段であり、本番公開手段ではありません。

## Ver.0.2.2 Contents

- Connection Information 表示
- Local URL 表示: `http://localhost:8501`
- Network URL 自動表示: `http://<PCのIPv4アドレス>:8501`
- Network URL の QR コード生成
- External Demo Access 表示
- Cloudflare Tunnel コマンド表示
- External Demo URL 入力欄
- Cloudflare Tunnel URL の QR コード生成
- Start Screen / Splash Screen
- iPad 横向き・PC 前提のダークテーマ Dashboard
- Camera Mode / Demo Mode / Video Upload の切り替え
- Motion Capture Demo Overlay の ON / OFF
- Analyze Motion の解析演出
- Event Timeline
- Dummy EMG waveform
- Motion Report with six educational scores
- AI Coaching cards








