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

**Whisper prompts go to the batched decoder only, and stay short.** Measured
against human captions: the title's names as a batched `initial_prompt` lowered
CER (English 2.0 → 1.5 %, Japanese 24.3 → 21.8 %) and tripled Japanese
punctuation, which segmentation lives on. The same text as sequential `hotwords`
raised it (English 1.0 → 4.3 %). A clip channel's hashtag pile (精華, 翻譯, 中文)
as the prompt sent the batched decoder into loops, so `build_prompt` keeps at most
six bracketed title names. Batched is 2–7× faster; the sequential decoder is the
repair path for chunks that loop.

**Japanese gaps inside a Whisper segment are not pauses** (`sentences._inside_cjk_word`).
Whisper reported 3.7 s between そ and こに, and 0.8 s between 金 and の玉. Cutting
there severs words; Japanese sentences break only at punctuation or segment ends.

**Word timestamps are corrected after decoding** (`asr.snap_to_speech`). Whisper
stamps the first kana of a phrase at the tail of the *previous* speaker's audio
(シ at 79.72 s, ャル、スマブラ… at 84.44 s), so a subtitle appeared 5 s early.
Words are clamped to Silero speech regions and a stranded kana run is moved back
next to its word. On a real three-speaker clip: lines stretched past 6 s went
5 → 0. Measure timing against human captions with a harness, not by eye.
Two more shapes of the same fault (`asr._unstrand`, 2026-09-29): a prefix
before a digit (で at 228.13 s, 1ラウンド at 235.04 s), and a gap hidden inside
an over-long first token (で stretched 598.53–600.41 s). On `v-V9NpCvPDk` 17 of
469 lines moved later; of those with a YouTube word time, 6 moved to within
0.4 s of it and 2 "moved away" only because YouTube's time was for an earlier
sentence in the same line. English (`km1IlBAZJH0`) is untouched. A harness that
matches a line's first words anywhere nearby is biased early (いや, これ recur);
match the first four letters and take the nearest occurrence.

**Why Whisper "misses" shouting: Silero, not Whisper** (measured 2026-09-29).
On `v-V9NpCvPDk` 418–452 s, Silero scored ≈ 0.00 over loud, clear speech that
Whisper transcribes well when handed the clip. Silero is stateful: the same 16 s
scored max 0.89 alone and 0.10 after the 30 s before it. Resetting its state
every 5–30 s only raised coverage of that stretch from 1 % to 13 %. pyannote's
segmentation (already run for diarization) covered 63 % of it. Unmeasured idea:
Silero ∪ pyannote as Whisper's speech regions — check recall on `0RcKbPmnvRU`
first, since longer chunks already hurt once (vad_threshold 0.2).

**Untranscribed speech is not re-transcribed.** Re-decoding the VAD speech no word
covers looked like the fix for lines lost in overlapping talk; measured, about
30 % of what it recovered was right and the rest was hallucination
(ありがとうございました, バイバーイ, ごちそうさまでした over laughter). A missed line
costs less than an invented one the viewer cannot check.

**Missed lines are filled from YouTube's auto-captions instead** (`pipeline/gapfill.py`).
That is a second recogniser, not a second try of Whisper. Shouting is what goes
missing: Silero scored a scream at 0.29 at most, under the 0.35 threshold, so
Whisper never saw it. **Do not lower `vad_threshold` to catch it:** 0.2 helped on
one clip and, on the human-captioned `0RcKbPmnvRU`, sent Whisper's recall from
85.3 % to 70.2 % (longer VAD chunks, more skipped inside them). YouTube's track
on that clip matched Whisper (recall 84.9 %, precision 93.1 % vs 92.0 %) and
mostly missed 4 of 48 lines to Whisper's 9. A line is added only when Whisper's
nearby words hold under a third of its text *and* cover under half its time:
2 of 2 added lines were right on `0RcKbPmnvRU`; on `VlaTbwhSSqk`, 19 added,
mostly the shouting, with one duplicate spelled differently (チャク様 / さくさま).
Only the original-language track (`ja-orig`, `ja`) is used; `[笑い]` tags are not
speech and must not end the line before them.

**Added lines keep YouTube's text but take Whisper's timing** (`gapfill.retime`).
Most auto-caption lines have no word offsets (372 of 505 on `v-V9NpCvPDk`), only
a display window: 何それ?何それ?やばい。ついてきてる。 was windowed 433.68–436.68 s
and spoken until 440.2 s, so it vanished mid-sentence. Whisper decodes each
line's window widened by 1.5 s / 5 s with the VAD off, and the line moves only
if half its letters are found there — so this is not re-transcription, and a
hallucination cannot reach the screen. On that video: 80 of 165 lines moved,
median start −0.7 s / end −0.6 s, none onto Whisper's own words, +88 s of GPU
time. Unverified against human timing: the one human-captioned clip adds only
two lines, both too short to align.

**Lines decoding cannot place are force-aligned** (`gapfill.retime_forced`,
`asr.align_texts`, 2026-10-28). Decoding a shouted window loops (ああ ×30,
はぁっ ×40) or hears something else, so about half the added lines used to keep
YouTube's window, whose end is the *next* line's start (median 0.7 s late).
Forced alignment puts YouTube's own text through Whisper's alignment heads
without decoding. On 199 YouTube lines Whisper's main pass also heard (truth =
Whisper's word times), for the 29 decoding missed: ends over 1 s off 34 % →
14 %, starts 28 % → 17 %. Lines left on YouTube timing: 89 → 11 of 171 on
`v-V9NpCvPDk`, 14 → 2 of 20 on `VlaTbwhSSqk`. The edge words absorb window
silence, hence the trims in `forced_span`; a start more than 1 s before
YouTube's is not trusted; an alignment piled onto the window's last frames
(text not in the window) keeps YouTube's timing. A probability gate was tried
and changed nothing.

**A json3 line's first piece has no time of its own** (measured 2026-09-30). It
carries the line's start: on `d1MzjzSTwvc` これ was stamped 19.40 s and said at
12.72 s (Whisper had it), so a line opened with it after seven silent seconds.
Later pieces match Whisper within 0.3 s. `gapfill._drop_heard_edges` drops a
first piece Whisper heard back to the previous line, and last pieces Whisper
heard in place. Widening the retime window back 7 s to catch an early first word
(投げえ, 1.1 s late) was tried and reverted: lines matched earlier repeats of
their own words (何それ?何それ? 433.6 → 426.7 s on `v-V9NpCvPDk`). The
windows' total must stay under the audio length or faster-whisper asserts
(`asr._within`).

**Short lines repeated by several people are speech, not a loop.** The repeat
filter only drops a line of 8+ characters said three times running; it used to
drop two speakers' "え？え？". Zero-length lines with 4+ characters are dropped
instead — that is the actual signature of a spliced-in loop.

**Human caption lines are kept as the human cut them** (`sentences_from_captions`),
even mid-sentence. The translator sees the next lines as look-ahead instead.
Re-gluing them into sentences produced 10-second run-ons on unpunctuated captions
and cues whose source and translation no longer matched.

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
The one sanctioned exception is the overlay's dialogue row (`content.js`
`heldLine`, on trial since 2026-09-28): across a change of voice, the previous
line's translation stays on a dimmed, smaller row *above* the current cue for at
most 1.5 s. Its position and dimming say "already said", so it does not lie
about who is talking; the cues themselves are unchanged. Whisper never emits
truly simultaneous lines (3 overlaps > 0.3 s in ~7,500 cached lines), so this is
the only form "two rows" can take.

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
A candidate model must fit in about 9 GB; Qwen3.8-27B (15 GB) does not run here.

**Terminology is never hand-written; it is grounded in wiki titles** (`terms.py`).
gemma4:12b alone rendered 55/99 Minecraft terms the Taiwanese way (下界 for
Nether); wiki langlinks + `varianttitles["zh-tw"]` got 76/99 in 1.8 s, and most
"misses" were the old hand-made glossary being wrong. Each acceptance rule in
`terms._chinese_titles` came from a real wrong answer: section redirects (crit →
Melee attack#Critical hit), renaming redirects (猫汰つな → her agency's page),
disambiguation and tutorial pages. Only the model's canonical `lookup` name is
queried — the spoken form ツナ is also tuna — and Wikipedia only for people,
places, groups, works and organisations (the item タバコ became 菸草, the plant).

**Diarization is sherpa-onnx on the CPU** (`pipeline/diarize.py`), four measured
choices in it. `window_shift_ratio` is 0.5, not the example's 0.1: 4.5× faster
and closer to reference labels, because the fine grid invented speakers. CAM++
(zh/en) beat ERes2NetV2, TitaNet and WeSpeaker (which merged everyone into one
voice). Threshold 0.9 plus dropping clusters under 5 s / 3 % of speech gave the
right count on two- and three-person clips; without the drop, laughter and game
audio became four extra "speakers". It runs *after* Whisper — concurrently, the
two fought over the CPU and the pair was slower than in sequence.

**`diarize._preload_onnxruntime` must run before sherpa-onnx loads.** The wheel
ships no `onnxruntime.dll`, so Windows picks System32's 1.17 copy (Windows ML)
and sherpa fails with "API version [28] is not available". `import onnxruntime`
alone does not load the DLL; `ctypes.WinDLL` on the pip package's copy does.

**Models download from Hugging Face, not GitHub releases** — ~7 MB/s against
~35 KB/s for the same files.

**A speaker change is a strong sentence boundary, not a forced one.** Turn
edges are only word-accurate, so forcing the cut stranded single words; the
"unfinished sentence" penalties are skipped across a change of voice. In
Japanese the edge usually lands mid-word (わかん|ない), where no cut is
allowed, so `sentences._voice_changes` moves it to the nearest breakable gap
within 0.6 s. Before that move, diarization barely changed Japanese
segmentation at all. Speakers are never displayed: the numbers only mark
boundaries (and keep `shape` from merging two voices).

**Whisper's spaces in Japanese are sentence boundaries** (`sentences._cjk_space`).
In fast cross-talk it writes "繋いで遊ぶ 初めて話してる 話したことないよ" with
no punctuation and word timings a uniform 0.1 s apart, so the space is the only
signal. Treated as "inside a word", it produced 4-voice run-ons; on `VlaTbwhSSqk`
honouring it (plus the voice move) took 72 lines to 91. An unpunctuated
segment end is weaker on purpose: at full strength it split 先輩が | ちょっと先に
| 火つけられます; it now cuts unless the line ends on a particle.
Cluster threshold was re-measured on the same clip (0.9 / 0.7 / 0.5): lower
values split one voice into several but moved only 2–4 of ~90 boundaries, so
0.9 stays.

**Demucs vocal isolation was measured and not adopted** (2026-09-28). The hope
was that Silero would hear shouting once the game audio was gone. It did not:
on `v-V9NpCvPDk` the VAD found 1144 s of speech on vocals vs 1143 s raw, and
still nothing in 415–453 s, where Whisper on the bare clip hears clear lines.
Whisper's output shrank (478 → 398 fragments), and on the human-captioned
`0RcKbPmnvRU` recall fell 84.1 % → 73.8 %. Separation itself is cheap (26 s for
27 min). Note `vocals.py` imports `demucs.api`, which PyPI's demucs 4.0.1 does
not ship, so `vocals.enabled` has only ever fallen back to the raw mix.

**OpenMOSS (MOSS-Transcribe-Diarize) was measured and not adopted.** On a 196 s
clip it took 130.7 s (Whisper 4.1 s); on a 452 s three-speaker clip, 544 s —
slower than real time — on Windows transformers, plus a 3 GB PyTorch dependency.
Against human captions its timing was the best measured (median error 0.68 s vs
0.83 s) but its recall was no better than Whisper's (8 vs 7 missing lines of 48).
It does label speakers. Revisit only with a fast runtime (vLLM under WSL2).


## Cache contract

Two fingerprints in `config.py` decide what a change retires. Transcription costs
minutes of GPU time and translation costs more, so they retire independently —
swapping the LLM must not re-run Whisper. `tests/test_fingerprints.py` pins this.

- `asr_fingerprint` covers everything shaping the stored sentences: the Whisper
  model, the forced language, batching and the prompt, vocal isolation,
  diarization, the caption source, the sentence-building thresholds, and
  `INGEST_VERSION`.
- `translation_fingerprint` covers the LLM, prompt version, style, target
  language, whether the summary pass ran, and the wiki lookup.

**Bump `INGEST_VERSION` when ingestion changes in a way settings do not
capture** — a caption parser fix, a change to hallucination filtering. Without
the bump the pipeline silently serves stale rows and the fix looks ignored. This
has already burned two debugging sessions.

**Bump `TERMS_VERSION` when a wiki acceptance rule changes.** It keys the
cross-video lookup cache, the per-video context and the translations built on it;
without it a fixed rule keeps serving the answer cached under the old one.

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
