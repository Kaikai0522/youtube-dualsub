/**
 * Popup: the few settings worth exposing (decision Q25).
 *
 * Display settings (font sizes, which line goes on top) live in
 * chrome.storage, where the content script picks them up immediately. Model,
 * OpenCC and the whole-video pass are pipeline concerns and live in the
 * backend's config.local.json. Everything else stays in the config file on
 * purpose — a settings panel is a bottomless pit, and a week of actual use is
 * what tells you which knobs you really wanted.
 */

const API = "http://127.0.0.1:8756";

// Keep in step with DEFAULT_STYLE in content.js.
const DEFAULT_STYLE = { zhOnTop: true, fontSizeZh: 30, fontSizeEn: 22 };

const els = {
  status: document.getElementById("status"),
  font: document.getElementById("font"),
  fontValue: document.getElementById("fontValue"),
  fontEn: document.getElementById("fontEn"),
  fontEnValue: document.getElementById("fontEnValue"),
  zhOnTop: document.getElementById("zhOnTop"),
  backend: document.getElementById("backendSettings"),
  model: document.getElementById("model"),
  opencc: document.getElementById("opencc"),
  context: document.getElementById("context"),
  footer: document.getElementById("footer"),
};

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.className = `status ${kind || ""}`;
}

async function patch(body) {
  const response = await fetch(`${API}/api/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`後端回應 ${response.status}`);
  return response.json();
}

async function loadStyle() {
  const stored = await chrome.storage.local.get(["style"]);
  return { ...DEFAULT_STYLE, ...(stored.style || {}) };
}

async function saveStyle(change) {
  const style = { ...(await loadStyle()), ...change };
  await chrome.storage.local.set({ style });
}

async function load() {
  const style = await loadStyle();
  els.font.value = style.fontSizeZh;
  els.fontValue.textContent = `${style.fontSizeZh}px`;
  els.fontEn.value = style.fontSizeEn;
  els.fontEnValue.textContent = `${style.fontSizeEn}px`;
  els.zhOnTop.checked = style.zhOnTop;

  let settings;
  try {
    const response = await fetch(`${API}/api/settings`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    settings = await response.json();
  } catch (error) {
    els.status.className = "status bad";
    els.status.innerHTML = "連不上本機後端。請雙擊 <code>start.cmd</code> 啟動。";
    els.backend.disabled = true;
    els.footer.textContent = `顯示設定仍可調整。（${error.message}）`;
    return;
  }

  setStatus("本機後端已連線。", "ok");
  els.opencc.checked = settings.opencc_enabled;
  els.context.checked = settings.context_enabled;

  const models = settings.models.length ? settings.models : [settings.translate_model];
  els.model.innerHTML = "";
  for (const name of models) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.selected = name === settings.translate_model;
    els.model.appendChild(option);
  }

  els.footer.textContent = `人聲分離目前${
    settings.vocals_enabled ? "開啟" : "關閉"
  }，可在 config.local.json 修改。`;
}

els.font.addEventListener("input", () => {
  const value = Number(els.font.value);
  els.fontValue.textContent = `${value}px`;
  saveStyle({ fontSizeZh: value });
});

els.fontEn.addEventListener("input", () => {
  const value = Number(els.fontEn.value);
  els.fontEnValue.textContent = `${value}px`;
  saveStyle({ fontSizeEn: value });
});

els.zhOnTop.addEventListener("change", () => {
  saveStyle({ zhOnTop: els.zhOnTop.checked });
});

/** Apply one backend setting, reverting the control if the backend refuses. */
function bindBackend(control, key, read, describe) {
  control.addEventListener("change", async () => {
    const value = read();
    try {
      await patch({ [key]: value });
      setStatus(describe(value), "ok");
    } catch (error) {
      setStatus(`設定失敗：${error.message}`, "bad");
      load();
    }
  });
}

bindBackend(els.model, "translate_model", () => els.model.value, (v) => `改用 ${v} 翻譯。`);
bindBackend(
  els.opencc,
  "opencc_enabled",
  () => els.opencc.checked,
  (on) => (on ? "已開啟 OpenCC 正體轉換。" : "已關閉 OpenCC 正體轉換。"),
);
bindBackend(
  els.context,
  "context_enabled",
  () => els.context.checked,
  (on) => (on ? "已開啟全片通讀。" : "已關閉全片通讀：較快，但術語可能前後不一。"),
);

load();
