# youtube_dualsub

在 YouTube 原生播放器上疊加**雙語字幕**：原文（日文或英文）與繁體中文同時顯示，全部在自己的 GPU 上運算，不依賴任何付費雲端服務。

專為**吵雜、快節奏、多人同時說話**的內容而做——遊戲實況、VTuber 精華、Podcast、Vlog。那正是免費方案表現最差的地方。語言會自動判斷，日文與英文是主要調校對象。

---

## 為什麼不直接用現成的擴充功能

好用的那幾個，「沒字幕就自己轉錄」都是**付費功能，且跑在他們的伺服器上**。免費層一律退回 YouTube 的自動字幕——那東西沒有標點、沒有大小寫，餵給翻譯模型會把辨識的小錯誤放大成一整段自信的胡言亂語。

本地管線能做、而逐句翻譯器**結構上做不到**的事，是**先把整支影片讀完**。全片摘要與自動抽取的術語表會被塞進每一個翻譯批次，所以 `pods` 不會變成「豆莢」，`creeper` 也不會在第 3 分鐘是「爬行者」、第 40 分鐘變「苦力怕」，而ツナ會被認出是成員猫汰つな，不是鮪魚。

術語表**不需要手動建立**。模型找出影片中的人名、地名與專有名詞後，會到 Wikipedia（以及該主題的 wiki，例如 minecraft.wiki）查出台灣正體的正式名稱。

---

## 需求

- NVIDIA 顯示卡，12 GB VRAM 以上：gemma4:12b 約佔 10 GB（開發環境為 16 GB 的 RTX 4070 Ti SUPER）
- Python 3.12
- [Ollama](https://ollama.com/download)，並安裝一個模型
- Chrome

---

## 安裝

```powershell
.\setup.ps1              # 要 Demucs 人聲分離就加 -WithVocals（額外約 3 GB）
```

腳本會安裝 uv、ffmpeg 與 Python 依賴，然後檢查在 Windows 上真正會出事的三件事：ffmpeg 是否在 PATH、CTranslate2 需要的 cuDNN 9 DLL、以及 Ollama 有沒有可用的模型。可重複執行。

接著載入擴充功能：`chrome://extensions` → 開啟開發人員模式 → **載入未封裝項目** → 選擇 `extension/` 資料夾。

---

## 使用

**按兩下 `start.cmd`**，然後讓那個視窗開著。

它會依序備妥後端需要的每一件事：關掉殘留的舊 server、`uv sync`、確認 yt-dlp 沒有過期、必要時在背景啟動 `ollama serve`、拉取設定檔指定的翻譯模型，最後等 `/api/health` 回應才宣告就緒。關掉視窗就是關掉後端。

其中 yt-dlp 那步值得說明：它的版本號本身就是日期，而 YouTube 每隔幾週就會弄壞它。過期時的症狀不是明顯的錯誤，而是每支影片都 `HTTP Error 403: Forbidden`。讀日期不需要連網也不花時間，所以只要超過 14 天就會在後端啟動前先升級。

```powershell
.\start.ps1 -UpdateYtDlp        # 不管幾天，強制升級 yt-dlp
.\start.ps1 -SkipYtDlpCheck     # 離線時完全不碰 yt-dlp
.\start.ps1 -YtDlpMaxAgeDays 7  # 改成 7 天就算過期
```

原本的方式仍然有效，只是不會幫你檢查上面那些：

```powershell
uv run python -m youtube_dualsub.main
```

打開影片，按播放器控制列上的 **雙語**。第一批翻譯完成就會開始出字幕，而翻譯速度遠快於播放速度，所以它會一直領先你。

也可以完全不碰瀏覽器：

```powershell
uv run dualsub IqcS1d3eXYc --export srt
```

---

## 運作方式

```
audio ──▶ vocals ──▶ asr ──────▶ sentences ──▶ context ──────▶ translate ──▶ shape
yt-dlp    Demucs     Whisper      評分式斷句    摘要 + 術語     gemma4       合併、
          （可選）    large-v3                  + wiki 查譯名   逐批串流      切分、夾制
                     batched
```

**嚴格串行**：每個階段釋放自己的 VRAM 之後，下一個才載入。峰值約 10 GB（gemma4:12b），16 GB 顯示卡綽綽有餘。

- **語言**：採用 YouTube 標示的影片語言，沒有標示時交給 Whisper 偵測。
- **Whisper** 使用 batched 模式，並把標題中的人名當成提示。這樣比逐段模式快 2–7 倍，還能讓日文輸出標點。出現重複迴圈的片段，會改用逐段模式重新辨識。
- **斷句**：在所有可能的切點裡找整體最佳解，綜合考慮標點、停頓、下一個字是否小寫、句尾是否停在冠詞或助詞上。日文段落內部的「停頓」是對齊誤差，不會拿來斷句。
- **說話者分離**：用 sherpa-onnx 在 CPU 上分辨是誰在說話，不佔 VRAM，7.5 分鐘的影片約 15 秒。結果不會顯示在字幕上，只拿來斷句：換人說話的地方會優先斷開，兩個人的話不會被接成同一句。日文的換人點常落在字詞中間，會移到最近可以斷句的位置（0.6 秒內）。第一次使用時會從 Hugging Face 下載約 35 MB 的模型。
- **日文快速接話**：Whisper 在接話很快、沒有標點的地方只會留一個空格（「繋いで遊ぶ 初めて話してる 話したことないよ」），這個空格會被當成斷句點。
- **用 YouTube 自動字幕補漏**：大喊大叫的句子常常被語音偵測（VAD）當成噪音，根本沒交給 Whisper。這些地方會改用 YouTube 自己的自動字幕補上，只補「Whisper 附近幾乎沒寫到這些字、而且這段時間 Whisper 也沒在說話」的句子，避免重複。實測在有人工字幕的片段上補了 2 句、2 句都對。設定 `asr.youtube_gap_fill` 可關閉。
- **人工字幕**：有人工上傳字幕的影片會**跳過音訊與 Whisper**，並**保留原本的斷行**。翻譯時模型會看到後面幾行，所以只有半句話的行也能翻對。

實測（RTX 4070 Ti SUPER，gemma4:12b）：

| 影片 | 來源 | 總耗時 | 倍率 |
|---|---|---|---|
| 33 分鐘英文 Minecraft 實況（1244 行） | 人工字幕 | 2.4 分 | 13.5× |
| 3 分鐘日文多人連動（6 人） | Whisper | 31 秒 | 6.3× |

同一支 33 分鐘影片改用 Whisper 辨識只要 30 秒。總耗時主要花在翻譯：同時送 4 個請求給 Ollama，它會一起解碼，每行約 0.09 秒（一次一個請求時是 0.18 秒），而且 VRAM 不會增加。


---

## 調整

所有參數集中在 `youtube_dualsub/config.py`。在專案根目錄建立 `config.local.json` 就能覆寫，結構相同：

```json
{
  "translate": { "model": "qwen3:14b" },
  "asr": { "language": "ja" },
  "context": { "wiki_lookup": false }
}
```

擴充功能的 popup 可即時調整：中文與原文字級、哪一行在上、翻譯模型、OpenCC 開關、全片摘要開關。

### 術語表

每支影片的術語都是自動產生的：

1. 模型讀完逐字稿和影片資訊（標題、說明欄、標籤），列出人名、地名、作品名與專有名詞。逐字稿裡沒出現的會被丟掉。
2. 到 Wikipedia 查人名、地名、團體與作品，到主題 wiki 查遊戲術語，取台灣正體（zh-tw）的頁面標題。查詢結果會跨影片快取在 SQLite，同主題的第二支影片不需要連網。
3. 查不到的就用模型自己的翻譯。英文等拉丁字母的人名一律保留原文。

以 Minecraft 的 99 個術語實測：模型單靠記憶只對 55 個（例如把 Nether 翻成中國譯名「下界」），查 wiki 後對 76 個，只花 1.8 秒。其餘不一致的項目，多半是舊的手寫詞彙表本身寫錯。

主題 wiki 定義在 `context.domain_wikis`，預設只有 Minecraft。任何連到中文版、且支援 zh-tw 轉換的 MediaWiki 都可以加進來。

若想**強制**某個譯名，寫進 `youtube_dualsub/glossaries/user.yaml`，它的優先權高於一切：

```yaml
terms:
  - source: gapple
    target: 金蘋果
    aliases: [金蘋]        # 出現這些寫法就改寫成 target
```

`aliases` 在中文是**純子字串比對**，所以避免使用本身就是常用詞的別名。

改完只要重跑翻譯，逐字稿不用重來：

```powershell
uv run dualsub <video_id> --retranslate
```

---

## 用量測代替猜測

有兩個開關預設是開的，因為它們**大概**是對的，不是因為在你的內容上證明過。CLI 每次跑完都會印出報告，讓你用數字定案：

```powershell
uv run dualsub <id> --vocals             # Demucs 人聲分離值不值得（需 -WithVocals）？
uv run dualsub <id> --no-opencc   --retranslate   # 這個模型真的會吐簡體嗎？
uv run dualsub <id> --no-context  --retranslate   # 摘要換來的一致性值不值那十幾秒？
uv run dualsub <id> --no-wiki     --retranslate   # wiki 查譯名到底改了多少？
uv run dualsub <id> --model qwen3:14b --retranslate
uv run dualsub <id> --language ja        # YouTube 標錯語言時強制指定
```

報告包含辨識出的語言、自動術語數與主題、簡體字偵測比例、未翻譯的行數，以及實際的即時倍率。

---

## 測試

```powershell
uv run --extra dev pytest
```

不需要 GPU、不需要 Ollama、不需要網路——模型與下載都是假的，真正被測的是**決定螢幕上出現什麼**的那些邏輯。

---

## 已知限制

**重疊語音。** 說話者分離能分辨輪流講話的人，但 Whisper 只會寫下最大聲的那一個，四個人同時講話仍然會變成一串沒有主詞的字。更重要的是，重疊處**辨識出來的詞本身就是錯的**，所以翻譯只能忠實地把垃圾翻成垃圾。WhisperX 也解決不了——它的 ASR 後端就是 faster-whisper，官方文件明講 *"Overlapping speech is not handled particularly well by whisper nor whisperx"*。

實際影響：以一支兩人雜談、附人工字幕的影片為準，48 行裡仍有 7 行辨識不到，多半是另一個人同時插的一兩個字。試過把「VAD 判定有人聲、卻沒有任何字幕」的片段單獨重新辨識，補回來的約七成是幻覺（例如對著笑聲寫出「ごちそうさまでした」），所以沒有採用：漏一句比多一句假的好。

OpenMOSS 的 MOSS-Transcribe-Diarize 也實測過。它能標出說話者，時間也最準，但召回率沒有比較好，而且在 Windows 上比影片本身還慢（452 秒的三人片段跑了 544 秒），還要多裝 3 GB 的 PyTorch，所以目前不採用。

**時間軸。** Whisper 常把一句話的第一個字標在前一個人講話的尾巴上，字幕會提早好幾秒出現。現在每個字的時間會先對齊到 VAD 偵測到的人聲範圍，被遺留在前面的假名也會移回原位。在一支三人雜談上，被拉長超過 6 秒的短句從 5 句降到 0 句。


**yt-dlp 隨時可能失效。** 2026 年起 YouTube 的 WEB client player response 已不再提供 adaptiveFormats 播放連結，只剩 SABR URL。所以所有下載路徑都被隔離在單一函式 `pipeline.audio.get_audio` 之後。預定的替代方案（`sources/mse_source.py`、`extension/inject.js`）是攔截 YouTube 播放器**自己已經下載**的音訊片段——那是 YouTube 無法封鎖而不封鎖自己的唯一路徑。

**模型會安靜地騙你。** 被要求「翻得自然」的模型會腦補細節，而你抓不到——因為你正是看不懂原文才在用這個工具。Prompt 明確禁止，而且**原文那一行永遠並列顯示**當作你的防呆機制。中文讀起來太順的時候，往上瞄一眼。

---

## 授權

MIT，見 [LICENSE](LICENSE)。
