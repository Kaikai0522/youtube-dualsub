/**
 * YouTube DualSub — content script.
 *
 * Draws bilingual subtitles over the normal YouTube player and talks to the
 * local backend on 127.0.0.1. Deliberately plain JavaScript with no build step:
 * subtitle styling is the kind of thing you tweak thirty times in a row, and a
 * bundler between the edit and the result is pure friction.
 *
 * Three things here are less obvious than they look:
 *
 *  - Timing is driven by requestAnimationFrame, not the `timeupdate` event.
 *    `timeupdate` fires about four times a second, which is a visible quarter
 *    second of subtitle lag on fast dialogue.
 *
 *  - Ads are detected explicitly. YouTube plays them through the *same* <video>
 *    element, so `currentTime` during an ad is the ad's clock. Rendering
 *    through that would desynchronise the entire video.
 *
 *  - Navigation is hooked on `yt-navigate-finish`. YouTube is a single-page
 *    app; moving to the next video never reloads the document, so without this
 *    the overlay would keep showing the previous video's subtitles.
 *
 * The backend reports progress in English (it also feeds logs and the CLI);
 * everything the viewer reads is translated here, in `describeProgress`.
 */

(() => {
  "use strict";

  const API = "http://127.0.0.1:8756";
  const WS = "ws://127.0.0.1:8756";
  const RECONNECT_DELAYS = [500, 1000, 2000, 4000];

  const DEFAULT_STYLE = { zhOnTop: true, fontSizeZh: 30, fontSizeEn: 22 };
  const KANA = /[\p{Script=Hiragana}\p{Script=Katakana}]/u;
  const ERROR_VISIBLE_MS = 6000;

  // Dialogue layout: in fast cross-talk the previous speaker's line stays on a
  // dimmed upper row while the reply plays, so a line cut short by the next
  // voice can still be finished. Only across a change of voice (the same
  // person's lines never stack), only when the reply follows within
  // HOLD_GAP_S, and never longer than HOLD_MAX_S past the line's own end.
  const HOLD_GAP_S = 0.6;
  const HOLD_MAX_S = 1.5;
  const PREV_SCALE = 0.8;

  const STAGE_LABELS = {
    queued: "排隊中",
    audio: "取得音訊",
    vocals: "分離人聲",
    asr: "語音辨識",
    diarize: "分析對話輪替",
    sentences: "斷句",
    context: "通讀全片",
    translate: "翻譯",
    shape: "排版字幕",
    done: "完成",
    paused: "已暫停",
    failed: "失敗",
  };

  /** Backend messages worth more than their stage label, as [pattern, Chinese]. */
  const MESSAGE_LABELS = [
    [/^Loading Whisper.*/, "載入語音辨識模型"],
    [/^Loading Demucs.*/, "載入人聲分離模型"],
    [/^Downloading audio.*/, "下載音訊"],
    [/^Downloading (\S+).*/, "下載模型 $1（僅第一次）"],
    [/^Checking for human-uploaded captions.*/, "檢查是否有官方字幕"],
    [/^Filling gaps from YouTube captions.*/, "用 YouTube 自動字幕補上漏掉的句子"],
    [/^Retiming YouTube lines.*/, "校正補上句子的時間"],
    [/^Using (\d+) human-uploaded captions.*/, "使用官方字幕（$1 行）"],
    [/^Reusing (\d+) cached transcript lines.*/, "使用已存的逐字稿（$1 句）"],
    [/^Reusing cached context.*/, "使用已存的全片摘要"],
    [/^All (\d+) lines already translated.*/, "已翻譯完成（$1 句）"],
    [/^Translating (\d+) lines.*/, "翻譯中（共 $1 句）"],
    [/^Looking up (\d+) terms.*/, "查詢專有名詞（$1 個）"],
    [/^Summarising part (\d+)\/(\d+).*/, "通讀全片（$1/$2）"],
    [/^No translation.*/, "翻譯失敗，只顯示原文"],
    [/^(\d+) sentences.*/, "斷句完成（$1 句）"],
  ];

  const state = {
    videoId: null,
    active: false,
    cues: [],
    index: -1,
    socket: null,
    reconnects: 0,
    style: { ...DEFAULT_STYLE },
    rafId: null,
    lastKey: "",
    statusTimer: null,
    overlay: null,
    resizeObserver: null,
  };

  // ---------------------------------------------------------------- dom ---

  const player = () => document.getElementById("movie_player");
  const video = () => document.querySelector("#movie_player video");

  /**
   * The overlay, attached to the current player. Called every frame, so the
   * common case (already attached) must stay cheap.
   */
  function ensureOverlay() {
    const host = player();
    if (!host) return null;
    const cached = state.overlay;
    if (cached && cached.parentElement === host) return cached;

    let overlay = document.getElementById("dualsub-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "dualsub-overlay";
      overlay.hidden = true;
      overlay.innerHTML =
        '<div class="dualsub-line dualsub-prev"></div>' +
        '<div class="dualsub-line dualsub-zh"></div><div class="dualsub-line dualsub-en"></div>';
    }
    // An overlay left by an older copy of this script (extension reloaded,
    // page not) lacks the held-line row.
    if (!overlay.querySelector(".dualsub-prev")) {
      const row = document.createElement("div");
      row.className = "dualsub-line dualsub-prev";
      overlay.prepend(row);
    }
    if (overlay.parentElement !== host) host.appendChild(overlay);
    applyOrder(overlay);
    scaleText(overlay);
    state.overlay = overlay;

    // Fullscreen, theater mode and the miniplayer resize the player without
    // changing the cue; without this the text kept its old size until the
    // next line.
    if (state.resizeObserver) state.resizeObserver.disconnect();
    state.resizeObserver = new ResizeObserver(() => scaleText(overlay));
    state.resizeObserver.observe(host);
    return overlay;
  }

  function ensureStatus() {
    const host = player();
    if (!host) return null;
    let chip = document.getElementById("dualsub-status");
    if (!chip) {
      chip = document.createElement("div");
      chip.id = "dualsub-status";
      chip.hidden = true;
      chip.innerHTML = '<span class="dualsub-text"></span><div class="dualsub-bar"><i></i></div>';
    }
    if (chip.parentElement !== host) host.appendChild(chip);
    return chip;
  }

  function applyOrder(overlay) {
    const zh = overlay.querySelector(".dualsub-zh");
    const en = overlay.querySelector(".dualsub-en");
    const first = state.style.zhOnTop ? zh : en;
    const second = state.style.zhOnTop ? en : zh;
    // The held previous line always sits above the current pair.
    if (first.nextElementSibling !== second || overlay.lastElementChild !== second) {
      overlay.appendChild(first);
      overlay.appendChild(second);
    }
  }

  function scaleText(overlay) {
    // Sizes are authored against a ~720p player and scaled from there, so the
    // subtitles look the same in the miniplayer and in fullscreen.
    const host = player();
    if (!host) return;
    const factor = Math.max(0.55, Math.min(2.2, host.clientHeight / 720));
    overlay.querySelector(".dualsub-zh").style.fontSize = `${state.style.fontSizeZh * factor}px`;
    overlay.querySelector(".dualsub-en").style.fontSize = `${state.style.fontSizeEn * factor}px`;
    overlay.querySelector(".dualsub-prev").style.fontSize =
      `${state.style.fontSizeZh * PREV_SCALE * factor}px`;
  }

  function ensureButton() {
    const controls = document.querySelector("#movie_player .ytp-right-controls");
    if (!controls || document.getElementById("dualsub-button")) return;

    const button = document.createElement("button");
    button.id = "dualsub-button";
    button.className = "ytp-button dualsub-button";
    button.title = "雙語字幕（本機生成）";
    button.setAttribute("aria-label", "雙語字幕");
    button.dataset.active = "false";
    button.innerHTML = '<span class="dualsub-badge">雙語</span>';

    button.addEventListener("click", () => (state.active ? stop() : start()));

    // YouTube nests the control buttons inside .ytp-right-controls-right on
    // current layouts, so the settings button is a descendant rather than a
    // child. insertBefore() demands a direct child and throws NotFoundError
    // otherwise; .before() places the button relative to its actual parent.
    const settings = controls.querySelector(".ytp-settings-button");
    if (settings) {
      settings.before(button);
    } else {
      controls.prepend(button);
    }
  }

  function setButton(active, busy) {
    const button = document.getElementById("dualsub-button");
    if (!button) return;
    button.dataset.active = String(active);
    button.dataset.busy = String(busy);
    button.setAttribute("aria-pressed", String(active));
    button.title = active ? "關閉雙語字幕" : "雙語字幕（本機生成）";
  }

  /**
   * Show `text` in the corner chip. Errors fade on their own: an accidental
   * click with the backend off must not leave "cannot reach" pinned over the
   * video. `hideAfterMs` overrides the delay; 0 keeps the chip up.
   */
  function setStatus(text, fraction, isError, hideAfterMs) {
    clearTimeout(state.statusTimer);
    const chip = ensureStatus();
    if (!chip) return;
    const delay = hideAfterMs != null ? hideAfterMs : isError ? ERROR_VISIBLE_MS : 0;
    if (text && delay > 0) {
      state.statusTimer = setTimeout(() => setStatus("", null, false), delay);
    }
    if (!text) {
      chip.hidden = true;
      return;
    }
    chip.hidden = false;
    chip.classList.toggle("dualsub-error", Boolean(isError));
    chip.querySelector(".dualsub-text").textContent = text;
    const bar = chip.querySelector(".dualsub-bar");
    // No fraction means "working, amount unknown": a moving bar rather than
    // one that looks stuck at zero.
    bar.classList.toggle("dualsub-indeterminate", fraction == null && !isError);
    bar.firstElementChild.style.width =
      fraction == null ? "0%" : `${Math.round(fraction * 100)}%`;
  }

  /** A progress event as one Chinese line: "語音辨識 45%", "下載音訊". */
  function describeProgress(event) {
    const message = event.message || "";
    let label = STAGE_LABELS[event.stage] || "處理中";
    for (const [pattern, chinese] of MESSAGE_LABELS) {
      if (pattern.test(message)) {
        label = message.replace(pattern, chinese);
        break;
      }
    }
    const fraction = event.fraction;
    if (fraction != null && fraction > 0 && fraction < 1) {
      return `${label} ${Math.round(fraction * 100)}%`;
    }
    return label;
  }

  // -------------------------------------------------------------- cues ----

  /** Replace every cue inside [lo, hi] with `incoming`, keeping the list sorted. */
  function spliceCues(incoming, lo, hi, replaceAll) {
    if (replaceAll) {
      state.cues = incoming.slice();
    } else {
      const kept = state.cues.filter((c) => c.s < lo || c.s > hi);
      state.cues = kept.concat(incoming);
    }
    state.cues.sort((a, b) => a.s - b.s);
    state.index = -1;
    state.lastKey = "";
  }

  function findCue(t) {
    const cues = state.cues;
    if (!cues.length) return -1;

    // Playback is mostly sequential, so check where we were before searching.
    const i = state.index;
    if (i >= 0 && i < cues.length && t >= cues[i].s && t < cues[i].e) return i;
    if (i + 1 < cues.length && t >= cues[i + 1].s && t < cues[i + 1].e) return i + 1;

    let lo = 0;
    let hi = cues.length - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (cues[mid].s <= t) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (found >= 0 && t < cues[found].e) return found;
    return -1;
  }

  /** The previous speaker's line, if it should stay up while cue `idx` plays. */
  function heldLine(idx, t) {
    if (idx < 1) return null;
    const cue = state.cues[idx];
    const prev = state.cues[idx - 1];
    // No diarization means no known change of voice, so nothing stacks.
    if (!prev.sp || !cue.sp || prev.sp === cue.sp) return null;
    if (cue.s - prev.e > HOLD_GAP_S || t - prev.e > HOLD_MAX_S) return null;
    return prev;
  }

  // --------------------------------------------------------------- loop ---

  function tick() {
    state.rafId = requestAnimationFrame(tick);
    const overlay = ensureOverlay();
    const v = video();
    if (!overlay || !v) return;

    const host = player();
    // During an ad the <video> is playing the ad, so its clock means nothing
    // for our cues. Hide rather than guess.
    const adShowing =
      host && (host.classList.contains("ad-showing") || host.classList.contains("ad-interrupting"));
    overlay.classList.toggle("dualsub-ad", Boolean(adShowing));
    if (adShowing) return;

    const t = v.currentTime;
    const idx = findCue(t);
    const cue = idx >= 0 ? state.cues[idx] : null;
    state.index = idx;
    const prev = cue ? heldLine(idx, t) : null;

    const key = cue ? `${cue.s}|${cue.zh}|${cue.en}|${prev ? prev.s : ""}` : "";
    if (key === state.lastKey) return;
    state.lastKey = key;

    if (!cue) {
      overlay.hidden = true;
      return;
    }
    overlay.hidden = false;
    // Only the translation is held: four lines of text would bury the picture.
    overlay.querySelector(".dualsub-prev").textContent = prev ? prev.zh || prev.en : "";
    overlay.classList.toggle("dualsub-pending", !cue.zh);
    const source = overlay.querySelector(".dualsub-en");
    overlay.querySelector(".dualsub-zh").textContent = cue.zh || "";
    source.textContent = cue.en;
    // Kana means Japanese: render it with a Japanese font, not the Chinese one.
    source.lang = KANA.test(cue.en) ? "ja" : "";
  }

  function startLoop() {
    if (state.rafId == null) state.rafId = requestAnimationFrame(tick);
  }

  function stopLoop() {
    if (state.rafId != null) cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }

  // ----------------------------------------------------------- backend ----

  async function loadStyle() {
    try {
      const stored = await chrome.storage.local.get(["style"]);
      if (stored.style) state.style = { ...DEFAULT_STYLE, ...stored.style };
    } catch (_) {
      /* storage is optional */
    }
  }

  async function start() {
    const id = currentVideoId();
    if (!id) return;
    state.videoId = id;
    state.active = true;
    document.body.classList.add("dualsub-active");
    setButton(true, true);
    setStatus("啟動中…", null, false);

    try {
      const response = await fetch(`${API}/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: id }),
      });
      if (!response.ok) throw new Error(`backend returned ${response.status}`);
    } catch (error) {
      setStatus(`連不上本機後端，請先執行 start.cmd（${error.message}）`, null, true);
      setButton(false, false);
      state.active = false;
      // Hand YouTube's own captions back; they were hidden for ours.
      document.body.classList.remove("dualsub-active");
      return;
    }
    connect();
    startLoop();
  }

  function connect() {
    if (!state.videoId) return;
    const socket = new WebSocket(`${WS}/ws/jobs/${state.videoId}`);
    state.socket = socket;

    socket.addEventListener("open", () => {
      state.reconnects = 0;
    });
    socket.addEventListener("message", (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch (_) {
        return;
      }
      handleEvent(payload);
    });
    socket.addEventListener("close", () => {
      if (!state.active) return;
      const delay = RECONNECT_DELAYS[Math.min(state.reconnects, RECONNECT_DELAYS.length - 1)];
      state.reconnects += 1;
      if (state.reconnects <= 6) {
        setTimeout(connect, delay);
      } else {
        // Give up cleanly: button off, YouTube's captions back, message fades.
        stop();
        setStatus("與本機後端的連線中斷了。", null, true);
      }
    });
  }

  function handleEvent(event) {
    switch (event.type) {
      case "progress": {
        setStatus(describeProgress(event), event.fraction, false);
        setButton(true, event.stage !== "done");
        break;
      }
      case "cues": {
        const [lo, hi] = event.window || [0, Infinity];
        spliceCues(event.cues || [], lo, hi, Boolean(event.replace_all));
        break;
      }
      case "done": {
        const captions = event.used_manual_captions ? "，使用官方字幕" : "";
        setStatus(
          `字幕完成：${event.cue_count} 句（${formatElapsed(event.elapsed_s)}${captions}）`,
          1,
          false,
          4000,
        );
        setButton(true, false);
        break;
      }
      case "paused": {
        setStatus("已暫停，進度已保存。", null, false);
        setButton(true, false);
        break;
      }
      case "error": {
        stop();
        setStatus(`字幕產生失敗：${event.message}`, null, true);
        break;
      }
      default:
        break;
    }
  }

  function formatElapsed(seconds) {
    const s = Math.round(seconds || 0);
    return s < 60 ? `${s} 秒` : `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  }

  function stop() {
    state.active = false;
    setButton(false, false);
    setStatus("", null, false);
    document.body.classList.remove("dualsub-active");
    stopLoop();
    const overlay = document.getElementById("dualsub-overlay");
    if (overlay) overlay.hidden = true;
    if (state.resizeObserver) state.resizeObserver.disconnect();
    state.resizeObserver = null;
    state.overlay = null;
    // Closing the socket is what tells the backend to pause: it keeps its
    // checkpoints and picks up where it left off next time.
    if (state.socket) {
      state.socket.close();
      state.socket = null;
    }
  }

  // ------------------------------------------------------- navigation -----

  function currentVideoId() {
    const match = location.search.match(/[?&]v=([A-Za-z0-9_-]{11})/);
    return match ? match[1] : null;
  }

  function onNavigate() {
    ensureButton();
    const id = currentVideoId();
    if (id === state.videoId) return;
    // A different video: everything we were showing is now wrong.
    const wasActive = state.active;
    stop();
    state.videoId = id;
    state.cues = [];
    state.index = -1;
    state.lastKey = "";
    if (wasActive && id) start();
  }

  document.addEventListener("yt-navigate-finish", onNavigate);

  // The control bar is rebuilt on layout changes (theater, fullscreen), which
  // silently drops our button; re-adding it is cheap enough to just poll.
  const keepAlive = setInterval(() => {
    ensureButton();
    if (state.active) ensureOverlay();
  }, 1500);

  // `unload` and `beforeunload` are blocked by YouTube's permissions policy —
  // registering them throws rather than doing nothing. `pagehide` is allowed,
  // fires on the same journeys, and is what closes the socket so the backend
  // knows to pause the job.
  window.addEventListener("pagehide", () => {
    clearInterval(keepAlive);
    if (state.socket) state.socket.close();
  });

  loadStyle().then(() => {
    ensureButton();
    onNavigate();
  });

  chrome.storage.onChanged.addListener((changes) => {
    if (!changes.style) return;
    state.style = { ...DEFAULT_STYLE, ...changes.style.newValue };
    const overlay = document.getElementById("dualsub-overlay");
    if (overlay) {
      applyOrder(overlay);
      scaleText(overlay);
    }
  });
})();
