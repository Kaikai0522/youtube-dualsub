/**
 * Popup: the three settings worth exposing (decision Q25).
 *
 * Font size is a display concern and lives in chrome.storage, where the content
 * script picks it up immediately. Model and OpenCC are pipeline concerns and
 * live in the backend's config.local.json. Everything else stays in the config
 * file on purpose — a settings panel is a bottomless pit, and a week of actual
 * use is what tells you which knobs you really wanted.
 */

const API = "http://127.0.0.1:8756";

const els = {
  status: document.getElementById("status"),
  model: document.getElementById("model"),
  font: document.getElementById("font"),
  fontValue: document.getElementById("fontValue"),
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
  if (!response.ok) throw new Error(`backend returned ${response.status}`);
  return response.json();
}

async function load() {
  const stored = await chrome.storage.local.get(["style"]);
  const fontSize = (stored.style && stored.style.fontSizeZh) || 30;
  els.font.value = fontSize;
  els.fontValue.textContent = `${fontSize}px`;

  let settings;
  try {
    const response = await fetch(`${API}/api/settings`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    settings = await response.json();
  } catch (error) {
    setStatus(`Backend unreachable. Start it with:  uv run python -m youtube_dualsub.main`, "bad");
    els.model.disabled = true;
    els.opencc.disabled = true;
    els.context.disabled = true;
    els.footer.textContent = error.message;
    return;
  }

  setStatus("Backend connected.", "ok");
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

  els.footer.textContent = `Vocal isolation is ${
    settings.vocals_enabled ? "on" : "off"
  }; change it in config.local.json.`;
}

els.font.addEventListener("input", async () => {
  const value = Number(els.font.value);
  els.fontValue.textContent = `${value}px`;
  const stored = await chrome.storage.local.get(["style"]);
  const style = { ...(stored.style || {}), fontSizeZh: value };
  await chrome.storage.local.set({ style });
});

els.model.addEventListener("change", async () => {
  try {
    await patch({ translate_model: els.model.value });
    setStatus(`Translating with ${els.model.value}.`, "ok");
  } catch (error) {
    setStatus(error.message, "bad");
  }
});

els.opencc.addEventListener("change", async () => {
  try {
    await patch({ opencc_enabled: els.opencc.checked });
    setStatus(els.opencc.checked ? "OpenCC on." : "OpenCC off.", "ok");
  } catch (error) {
    setStatus(error.message, "bad");
  }
});

els.context.addEventListener("change", async () => {
  try {
    await patch({ context_enabled: els.context.checked });
    setStatus(
      els.context.checked
        ? "Whole-video summary on."
        : "Summary off — faster, but terminology may drift.",
      "ok",
    );
  } catch (error) {
    setStatus(error.message, "bad");
  }
});

load();
