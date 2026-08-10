# youtube_dualsub

在 YouTube 原生播放器上疊加**雙語字幕**：原文與繁體中文同時顯示，全部在自己的 GPU 上運算，不依賴任何付費雲端服務。

專為**吵雜、快節奏、多人同時說話**的內容而做——遊戲實況、Podcast、Vlog。那正是免費方案表現最差的地方。

---

## 為什麼不直接用現成的擴充功能

好用的那幾個，「沒字幕就自己轉錄」都是**付費功能，且跑在他們的伺服器上**。免費層一律退回 YouTube 的自動字幕——那東西沒有標點、沒有大小寫，餵給翻譯模型會把辨識的小錯誤放大成一整段自信的胡言亂語。

本地管線能做、而逐句翻譯器**結構上做不到**的事，是**先把整支影片讀完**。全片摘要與自動抽取的術語表會被塞進每一個翻譯批次，所以 `pods` 不會變成「豆莢」，`creeper` 也不會在第 3 分鐘是「爬行者」、第 40 分鐘變「苦力怕」。

---

## 需求

- NVIDIA 顯示卡，8 GB VRAM 以上（開發環境為 16 GB 的 RTX 4070 Ti SUPER）
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

啟動後端並保持執行：

```powershell
uv run python -m youtube_dualsub.main
```

打開影片，按播放器控制列上的 **中/EN**。第一批翻譯完成就會開始出字幕，而翻譯速度遠快於播放速度，所以它會一直領先你。

也可以完全不碰瀏覽器：

```powershell
uv run dualsub IqcS1d3eXYc --export srt
```

---

## 運作方式

```
audio ──▶ vocals ──▶ asr ──▶ sentences ──▶ context ──▶ translate ──▶ shape
yt-dlp    Demucs     Whisper  碎片重組      摘要 +      Ollama       合併、
          （可選）    large-v3              術語表      逐批串流      切分、夾制
```

**嚴格串行**：每個階段釋放自己的 VRAM 之後，下一個才載入。峰值約 9 GB，儘管三個模型加起來遠不只如此。

實測（52 分鐘影片，RTX 4070 Ti SUPER）：

| 階段 | 耗時 |
|---|---|
| 抓音訊 | ~1 分 |
| Whisper large-v3（857 句） | 134 秒 |
| 全片摘要與術語抽取 | ~45 秒 |
| 翻譯（86 批） | ~3.5 分 |
| **總計** | **4.1 分鐘，12.5 倍即時速度** |

有人工上傳字幕的影片會**跳過音訊與 Whisper**，直接使用該字幕——品質更好、時間軸更準、零 GPU 成本。

---

## 調整

所有參數集中在 `youtube_dualsub/config.py`。在專案根目錄建立 `config.local.json` 就能覆寫，結構相同：

```json
{
  "translate": { "model": "gemma4:12b" },
  "vocals": { "enabled": false },
  "context": { "enabled": false }
}
```

擴充功能的 popup 可即時調整四項：翻譯模型、字級、OpenCC 開關、全片摘要開關。

### 術語表

`youtube_dualsub/glossaries/user.yaml` 的優先權高於一切，包含領域術語表與模型自己抽取的結果。名字翻錯就改這裡：

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
uv run dualsub <id> --no-vocals   --retranslate   # Demucs 值不值那 3 分鐘？
uv run dualsub <id> --no-opencc   --retranslate   # 這個模型真的會吐簡體嗎？
uv run dualsub <id> --no-context  --retranslate   # 摘要換來的一致性值不值 45 秒？
uv run dualsub <id> --model gemma4:12b --retranslate
```

報告包含簡體字偵測比例、退回英文的行數，以及實際的即時倍率。

---

## 測試

```powershell
uv run --extra dev pytest
```

不需要 GPU、不需要 Ollama、不需要網路——模型與下載都是假的，真正被測的是**決定螢幕上出現什麼**的那些邏輯。

---

## 已知限制

**重疊語音。** Whisper 沒有語者分離，四個人同時講話會變成一串沒有主詞的字。更重要的是，重疊處**辨識出來的詞本身就是錯的**，所以翻譯只能忠實地把垃圾翻成垃圾。WhisperX 也解決不了——它的 ASR 後端就是 faster-whisper，官方文件明講 *"Overlapping speech is not handled particularly well by whisper nor whisperx"*。`Sentence.speaker` 欄位與渲染都已預留，接 pyannote 是可行的，但那只會讓胡言亂語被精確標註是誰講的。

**yt-dlp 隨時可能失效。** 2026 年起 YouTube 的 WEB client player response 已不再提供 adaptiveFormats 播放連結，只剩 SABR URL。所以所有下載路徑都被隔離在單一函式 `pipeline.audio.get_audio` 之後。預定的替代方案（`sources/mse_source.py`、`extension/inject.js`）是攔截 YouTube 播放器**自己已經下載**的音訊片段——那是 YouTube 無法封鎖而不封鎖自己的唯一路徑。

**模型會安靜地騙你。** 被要求「翻得自然」的模型會腦補細節，而你抓不到——因為你正是看不懂原文才在用這個工具。Prompt 明確禁止，而且**原文那一行永遠並列顯示**當作你的防呆機制。中文讀起來太順的時候，往上瞄一眼。

---

## 授權

MIT，見 [LICENSE](LICENSE)。
