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
 */

(() => {
  "use strict";

  const API = "http://127.0.0.1:8756";
  const WS = "ws://127.0.0.1:8756";
  const RECONNECT_DELAYS = [500, 1000, 2000, 4000];

  const DEFAULT_STYLE = { zhOnTop: true, fontSizeZh: 30, fontSizeEn: 22 };

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
  };

  // ---------------------------------------------------------------- dom ---

  const player = () => document.getElementById("movie_player");
  const video = () => document.querySelector("#movie_player video");

  function ensureOverlay() {
    const host = player();
    if (!host) return null;
    let overlay = document.getElementById("dualsub-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "dualsub-overlay";
      overlay.hidden = true;
      overlay.innerHTML =
        '<div class="dualsub-line dualsub-zh"></div><div class="dualsub-line dualsub-en"></div>';
    }
    if (overlay.parentElement !== host) host.appendChild(overlay);
    applyOrder(overlay);
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
    if (overlay.firstElementChild !== first) {
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
  }

  function ensureButton() {
    const controls = document.querySelector("#movie_player .ytp-right-controls");
    if (!controls || document.getElementById("dualsub-button")) return;

    const button = document.createElement("button");
    button.id = "dualsub-button";
    button.className = "ytp-button dualsub-button";
    button.title = "Bilingual subtitles (local)";
    button.dataset.active = "false";
    button.innerHTML = '<span class="dualsub-badge">中/EN</span>';
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
  }

  function setStatus(text, fraction, isError) {
    const chip = ensureStatus();
    if (!chip) return;
    if (!text) {
      chip.hidden = true;
      return;
    }
    chip.hidden = false;
    chip.classList.toggle("dualsub-error", Boolean(isError));
    chip.querySelector(".dualsub-text").textContent = text;
    const bar = chip.querySelector(".dualsub-bar > i");
    bar.style.width = fraction == null ? "0%" : `${Math.round(fraction * 100)}%`;
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

    const idx = findCue(v.currentTime);
    const cue = idx >= 0 ? state.cues[idx] : null;
    state.index = idx;

    const key = cue ? `${cue.s}|${cue.zh}|${cue.en}` : "";
    if (key === state.lastKey) return;
    state.lastKey = key;

    if (!cue) {
      overlay.hidden = true;
      return;
    }
    overlay.hidden = false;
    overlay.classList.toggle("dualsub-pending", !cue.zh);
    const prefix = cue.sp ? `${cue.sp}: ` : "";
    overlay.querySelector(".dualsub-zh").textContent = cue.zh ? prefix + cue.zh : "";
    overlay.querySelector(".dualsub-en").textContent = cue.zh ? cue.en : prefix + cue.en;
    scaleText(overlay);
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
    setStatus("Starting…", null, false);

    try {
      const response = await fetch(`${API}/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: id }),
      });
      if (!response.ok) throw new Error(`backend returned ${response.status}`);
    } catch (error) {
      setStatus(`Cannot reach the local backend — is it running? (${error.message})`, null, true);
      setButton(false, false);
      state.active = false;
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
      if (state.reconnects <= 6) setTimeout(connect, delay);
      else setStatus("Lost the connection to the backend.", null, true);
    });
  }

  function handleEvent(event) {
    switch (event.type) {
      case "progress": {
        const label = event.message || event.stage;
        setStatus(label, event.fraction, false);
        setButton(true, event.stage !== "done");
        break;
      }
      case "cues": {
        const [lo, hi] = event.window || [0, Infinity];
        spliceCues(event.cues || [], lo, hi, Boolean(event.replace_all));
        break;
      }
      case "done": {
        const mins = (event.elapsed_s / 60).toFixed(1);
        setStatus(`${event.cue_count} subtitles ready (${mins} min)`, 1, false);
        setButton(true, false);
        setTimeout(() => setStatus("", null, false), 4000);
        break;
      }
      case "paused": {
        setStatus(event.message || "Paused.", null, false);
        setButton(true, false);
        break;
      }
      case "error": {
        setStatus(event.message, null, true);
        setButton(false, false);
        break;
      }
      default:
        break;
    }
  }

  function stop() {
    state.active = false;
    setButton(false, false);
    setStatus("", null, false);
    document.body.classList.remove("dualsub-active");
    stopLoop();
    const overlay = document.getElementById("dualsub-overlay");
    if (overlay) overlay.hidden = true;
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
