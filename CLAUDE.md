# youtube_dualsub

Local bilingual YouTube subtitles. Pipeline, setup and tuning are in `README.md`;
this file carries only what the code and config cannot tell you on their own.

## Chesterton's fences

Each of these looks like an oversight or an easy improvement. Each was measured,
and removing it costs hours. Keep the fence; if you must move one, re-measure and
update the number here.

**Reasoning stays off** (`translate.think = False`). Both gemma4 and qwen3 are
reasoning models. Asked to translate a three-word line, gemma4 spent up to 4000
thinking tokens and filled the entire context before writing any answer — the
request returns an empty string ~100 seconds later with no error. Turning
reasoning off took a batch of ten from 60–100 s to ~2 s.

**The generation budget is enforced client-side by streaming, never with
`num_predict`.** Setting `num_predict` makes *every* request behave like a
runaway: the model emits exactly the cap and returns nothing, including requests
that succeeded moments earlier without it. `llm._stream_bounded` counts answer
chunks and closes the connection instead.

**Reasoning tokens arrive in a separate `thinking` field**, not as inline
`<think>` tags. Anything counting tokens must skip them, or it cuts models off
before they write a single character. The `_THINK` regex in `llm.py` is a
fallback for models that do inline it, and it never fires on Ollama.

**`asr.condition_on_previous_text = False`.** It reads like a coherence
regression. With it on, Whisper feeds its own output back as a prompt, so one
hallucination during a music sting seeds the next window and the damage cascades
for minutes of noisy multi-speaker audio.

**Manual captions are collapsed** (`ytdlp_source._collapse_rolling`). YouTube
serves captions as a rolling highlight: the same line re-sent every time the
highlighted word advances. On a real 6-minute video, 85 % of consecutive cues
were exact repeats and the median cue lasted 0.18 s — 262 real lines arriving as
1729 cues. Without the collapse the subtitles strobe and the translator is billed
for every duplicate.

**A cue never outlives the next cue's start** (`pipeline/shape.py`). The
readability floor `min_duration_s` is a soft bound that only fills genuine gaps.
Holding a line longer covers the next speaker, which makes the subtitle lie about
who is talking. Fix flicker by merging short cues, never by extending them.

**Splitting protects known words.** Chinese has no spaces, so an index-based
split severs 種子碼 into 種 / 子碼. `shape._forbidden_cuts` protects glossary
renderings and Latin/number tokens, and balance beats filling: 23 characters
under a 20-character limit split 12 + 11, not 20 + 3.

**Chrome match patterns carry no port.** `http://127.0.0.1:8756/*` in
`manifest.json` is invalid and Chrome rejects the *whole manifest*, so the
symptom is an extension that does nothing at all rather than a blocked request.
Use `http://127.0.0.1/*`.

**The content script uses `pagehide`.** YouTube's permissions policy blocks
`unload` and `beforeunload`; registering them throws rather than doing nothing.

**yt-dlp is upgraded at launch, never mid-job** (`start.ps1`). A stale yt-dlp
does not fail loudly; it 403s on every video, because YouTube's SABR migration
retires whatever client that release still knew how to use. Measured on
`nNWM9a-SNTQ`: 2026.07.04 could list formats via `android_vr` but every media URL
returned 403, and `ios`/`mweb`/`web_safari` returned no formats at all;
2026.08.19 switched to `visionos` and downloaded on the first format spec with
`format_chain` untouched. The version string is a date, so the staleness check
costs no network. Moving the upgrade into the pipeline's `AudioUnavailable`
handler looks obvious and does nothing: the running process already holds the old
`yt_dlp` module in memory, so the fix cannot take effect until the next restart.

**Stages run serially and release their VRAM.** Demucs (~4 GB), Whisper (~3 GB)
and a 12–14 B model (~9 GB) never coexist. Peak stays under 9 GB on a 16 GB card.

## Cache contract

Two fingerprints in `config.py` decide what a change retires. Transcription costs
minutes of GPU time and translation costs more, so they retire independently —
swapping the LLM must not re-run Whisper. `tests/test_fingerprints.py` pins this.

- `asr_fingerprint` covers everything shaping the stored sentences: the Whisper
  model, vocal isolation, the caption source, the sentence-building thresholds,
  and `INGEST_VERSION`.
- `translation_fingerprint` covers the LLM, prompt version, style, target
  language, and whether the summary pass ran.

**Bump `INGEST_VERSION` when ingestion changes in a way settings do not
capture** — a caption parser fix, a change to hallucination filtering. Without
the bump the pipeline silently serves stale rows and the fix looks ignored. This
has already burned two debugging sessions.

## Verifying a change

**Python edits need a backend restart.** The running server holds the old code in
memory; the CLI picks up edits immediately because each run is a fresh process.
This is the usual reason a fix "did not work" in the browser.

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*youtube_dualsub.main*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
uv run python -m youtube_dualsub.main
```

`start.cmd` does exactly this kill-then-start on every launch, so a double-click
is also the shortest way to pick up a Python edit.

**Extension edits need both a reload and a page refresh.** The old content script
keeps running in open tabs until the page reloads, so stale errors keep appearing
from code that no longer exists.

**Subtitle rendering can only be confirmed on a visible tab.** Timing is driven by
`requestAnimationFrame`, which does not fire in a background tab — automation
sees `visibilityState: "hidden"`, zero frames, and an empty overlay while the code
is perfectly fine. Verify the backend by reading what the WebSocket sends; ask a
human to look at the screen for the rest.

**Measure on real output before concluding.** Every wrong diagnosis in this
project's history came from reasoning about the code instead of instrumenting it,
and three of them were measurements of a limit the debugging had itself
introduced. Print the raw value.

## Conventions

- Python: `uv run` for everything; tests are `uv run --extra dev pytest`.
- The extension is plain JavaScript with no build step, so subtitle styling stays
  a save-and-reload loop.
- Anything touching how audio is obtained belongs behind
  `pipeline.audio.get_audio`. YouTube is actively breaking that path, and the
  planned replacement swaps one function body.
