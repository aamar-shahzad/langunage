import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowRemoteModels = true;
env.useBrowserCache = true;

const $ = (id) => document.getElementById(id);

const ui = {
  loadModelBtn: $("load-model-btn"),
  runBtn: $("run-btn"),
  clearBtn: $("clear-btn"),
  copyBtn: $("copy-btn"),
  downloadBtn: $("download-btn"),
  clearHistoryBtn: $("clear-history-btn"),
  runsToday: $("runs-today"),
  streakCount: $("streak-count"),
  totalRuns: $("total-runs"),
  dailyGoalInput: $("daily-goal-input"),
  goalProgress: $("goal-progress"),
  markDoneBtn: $("mark-done-btn"),
  llmDot: $("llm-dot"),
  llmLabel: $("llm-label"),
  statusText: $("status-text"),
  modelSelect: $("model-select"),
  toolSelect: $("tool-select"),
  toneSelect: $("tone-select"),
  lengthSelect: $("length-select"),
  autoRunTemplate: $("autorun-template"),
  templateSelect: $("template-select"),
  applyTemplateBtn: $("apply-template-btn"),
  inspireCards: document.querySelectorAll(".inspire-card"),
  contextInput: $("context-input"),
  sourceInput: $("source-input"),
  outputBox: $("output-box"),
  charCount: $("char-count"),
  historyList: $("history-list"),
};

const state = {
  llmPipe: null,
  modelsLoading: false,
  modelReady: false,
  busy: false,
  modelId: "onnx-community/Qwen2.5-0.5B-Instruct",
  history: [],
  streamJob: 0,
  usage: {
    totalRuns: 0,
    lastActiveDate: null,
    dailyRuns: 0,
    streak: 0,
    completedDays: [],
  },
};

const HISTORY_KEY = "qwendesk-history-v1";
const HISTORY_LIMIT = 8;
const USAGE_KEY = "qwendesk-usage-v1";
const SETTINGS_KEY = "qwendesk-settings-v1";
const DRAFT_KEY = "qwendesk-draft-v1";

const templates = {
  "sales-followup": {
    tool: "reply",
    tone: "professional",
    context: "B2B SaaS follow-up after a product demo; goal is to move toward next meeting.",
    source:
      "Hi team, thanks for your time earlier. I wanted to follow up and see if there are any questions from your side.",
  },
  "support-apology": {
    tool: "reply",
    tone: "empathetic",
    context: "Customer reported delayed refund and is upset. Need a clear apology and timeline.",
    source:
      "I know this delay has been frustrating and we appreciate your patience. I am checking with billing and will update you soon.",
  },
  "interview-followup": {
    tool: "rewrite",
    tone: "professional",
    context: "Polish this into a concise thank-you email after a software engineer interview.",
    source:
      "Thanks for interview today I really liked learning about your team and I think I can help with the platform migration work.",
  },
  "meeting-recap": {
    tool: "summarize",
    tone: "direct",
    context: "Create a short recap for cross-functional weekly sync with clear owners and deadlines.",
    source:
      "Today we reviewed launch blockers: API docs still incomplete, design waiting on legal copy, and QA found payment edge-case failures. Maya will finalize docs by Wednesday, Jon will coordinate legal feedback by Thursday, and Ravi will retest after patch deployment.",
  },
};

function setModelState(kind, label) {
  ui.llmDot.className = `dot ${kind}`;
  ui.llmLabel.textContent = label;
}

function setStatus(text) {
  ui.statusText.textContent = text;
}

function getTodayKey() {
  return new Date().toISOString().slice(0, 10);
}

function isYesterday(dateStr) {
  if (!dateStr) return false;
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().slice(0, 10) === dateStr;
}

function setBusy(next) {
  state.busy = next;
  ui.runBtn.disabled = !state.modelReady || next;
  ui.loadModelBtn.disabled = next || state.modelsLoading;
}

function esc(text) {
  return (text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function updateCount() {
  ui.charCount.textContent = `${(ui.outputBox.textContent || "").length} chars`;
}

async function streamTextToOutput(text) {
  const content = text || "";
  const myJob = ++state.streamJob;
  ui.outputBox.textContent = "";
  updateCount();

  // Fast path for short outputs.
  if (content.length < 140) {
    ui.outputBox.textContent = content;
    updateCount();
    return;
  }

  const chunkSize = 8;
  for (let i = 0; i < content.length; i += chunkSize) {
    if (myJob !== state.streamJob) return;
    ui.outputBox.textContent = content.slice(0, i + chunkSize);
    updateCount();
    await new Promise((resolve) => setTimeout(resolve, 14));
  }
}

function updateUsageUI() {
  ui.runsToday.textContent = String(state.usage.dailyRuns || 0);
  ui.streakCount.textContent = `${state.usage.streak || 0} days`;
  ui.totalRuns.textContent = String(state.usage.totalRuns || 0);
  const goal = Number.parseInt(ui.dailyGoalInput.value || "5", 10);
  ui.goalProgress.textContent = `${state.usage.dailyRuns || 0} / ${goal}`;
}

function saveUsage() {
  localStorage.setItem(USAGE_KEY, JSON.stringify(state.usage));
}

function loadUsage() {
  try {
    const raw = localStorage.getItem(USAGE_KEY);
    if (raw) {
      state.usage = { ...state.usage, ...JSON.parse(raw) };
    }
  } catch (_) {
    // Ignore invalid stored data.
  }
  if (state.usage.lastActiveDate !== getTodayKey()) {
    state.usage.dailyRuns = 0;
  }
  updateUsageUI();
}

function saveSettings() {
  const payload = {
    modelId: ui.modelSelect.value,
    tool: ui.toolSelect.value,
    tone: ui.toneSelect.value,
    length: ui.lengthSelect.value,
    autoRunTemplate: ui.autoRunTemplate.checked,
    dailyGoal: ui.dailyGoalInput.value || "5",
  };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(payload));
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const settings = JSON.parse(raw);
    if (settings.modelId) ui.modelSelect.value = settings.modelId;
    if (settings.tool) ui.toolSelect.value = settings.tool;
    if (settings.tone) ui.toneSelect.value = settings.tone;
    if (settings.length) ui.lengthSelect.value = settings.length;
    if (typeof settings.autoRunTemplate === "boolean") ui.autoRunTemplate.checked = settings.autoRunTemplate;
    if (settings.dailyGoal) ui.dailyGoalInput.value = settings.dailyGoal;
  } catch (_) {
    // Ignore invalid settings.
  }
}

function saveDraft() {
  const payload = {
    context: ui.contextInput.value,
    source: ui.sourceInput.value,
  };
  localStorage.setItem(DRAFT_KEY, JSON.stringify(payload));
}

function loadDraft() {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return;
    const draft = JSON.parse(raw);
    if (draft.context) ui.contextInput.value = draft.context;
    if (draft.source) ui.sourceInput.value = draft.source;
    if (draft.context || draft.source) {
      setStatus("Draft restored");
    }
  } catch (_) {
    // Ignore invalid draft.
  }
}

function recordRun() {
  const today = getTodayKey();
  const previous = state.usage.lastActiveDate;
  if (previous !== today) {
    if (isYesterday(previous)) state.usage.streak += 1;
    else state.usage.streak = 1;
    state.usage.dailyRuns = 0;
    state.usage.lastActiveDate = today;
  }
  state.usage.dailyRuns += 1;
  state.usage.totalRuns += 1;
  saveUsage();
  updateUsageUI();
}

function buildSystemPrompt(tool, tone, context, length) {
  const lengthRule = {
    short: "Keep output compact and direct.",
    balanced: "Keep output concise but complete.",
    detailed: "Provide richer detail while staying practical.",
  };
  const base = `You are a practical writing assistant. Tone: ${tone}. ${lengthRule[length] || lengthRule.balanced}`;
  if (tool === "rewrite") {
    return `${base}
Rewrite the user text to be clearer and cleaner while preserving intent.
Output only the rewritten result.
${context ? `Context: ${context}` : ""}`;
  }
  if (tool === "summarize") {
    return `${base}
Summarize user text in markdown with sections:
- Summary
- Key points
- Action items
Keep it concise.
${context ? `Context: ${context}` : ""}`;
  }
  return `${base}
Generate exactly 3 business-ready reply options in markdown:
1) Short
2) Balanced
3) Detailed
Rules:
- Preserve user's intent and key facts.
- Do not invent names or events not present in the input/context.
- Keep each option realistic and ready to send.
- Avoid generic filler lines.
${context ? `Context: ${context}` : ""}`;
}

function buildPrompt(systemPrompt, userText) {
  return `<|im_start|>system
${systemPrompt}<|im_end|>
<|im_start|>user
${userText}<|im_end|>
<|im_start|>assistant
`;
}

function extractAssistantText(generatedText, prompt) {
  let text = generatedText || "";

  if (prompt && text.startsWith(prompt)) {
    text = text.slice(prompt.length);
  }

  // Handle role echoes with or without colon.
  if (/(\n|^)assistant\s*:?\s*\n/i.test(text)) {
    const parts = text.split(/(?:\n|^)assistant\s*:?\s*\n/i);
    text = parts[parts.length - 1] || "";
  }

  // Remove any remaining chat template tokens or role echoes.
  text = text
    .replace(/<\|im_start\|>system[\s\S]*?<\|im_end\|>/gi, "")
    .replace(/<\|im_start\|>user[\s\S]*?<\|im_end\|>/gi, "")
    .replace(/<\|im_start\|>assistant/gi, "")
    .replace(/<\|im_end\|>/gi, "")
    .replace(/(?:^|\n)system\s*:?\s*\n[\s\S]*?(?=(?:\nuser\s*:?\s*\n|\nassistant\s*:?\s*\n|$))/gi, "")
    .replace(/(?:^|\n)user\s*:?\s*\n[\s\S]*?(?=(?:\nassistant\s*:?\s*\n|$))/gi, "")
    .replace(/^You are a practical writing assistant\.[\s\S]*?(?=\n(?:assistant\s*:?\s*\n|1\)\s*Short|$))/i, "")
    .trim();

  const tailMatch = text.match(/assistant\s*:?\s*\n([\s\S]*)$/i);
  if (tailMatch && tailMatch[1]) {
    text = tailMatch[1].trim();
  }

  return text;
}

async function loadModel() {
  if (state.modelReady || state.modelsLoading) return;
  state.modelsLoading = true;
  setBusy(true);
  state.modelId = ui.modelSelect.value || state.modelId;
  const selectedLabel = ui.modelSelect.options[ui.modelSelect.selectedIndex]?.text || "Selected model";
  setModelState("loading", "Model loading");
  setStatus(`Loading ${selectedLabel}...`);

  try {
    state.llmPipe = await pipeline("text-generation", state.modelId, {
      progress_callback: (p) => {
        if (p.status === "progress" && p.progress) {
          setStatus(`Loading model: ${Math.round(p.progress)}%`);
        }
      },
    });
    state.modelReady = true;
    setModelState("ready", "Model ready");
    setStatus("Ready");
  } catch (error) {
    setModelState("error", "Model failed");
    setStatus(`Model load failed: ${error.message}`);
  } finally {
    state.modelsLoading = false;
    setBusy(false);
  }
}

async function runTool() {
  if (!state.modelReady || state.busy) return;

  const source = ui.sourceInput.value.trim();
  if (!source) {
    setStatus("Please add input text");
    return;
  }

  const tool = ui.toolSelect.value;
  const tone = ui.toneSelect.value;
  const length = ui.lengthSelect.value;
  const context = ui.contextInput.value.trim();
  const systemPrompt = buildSystemPrompt(tool, tone, context, length);
  const prompt = buildPrompt(systemPrompt, source);
  const lengthTokenBudget = { short: 120, balanced: 220, detailed: 360 };

  setBusy(true);
  setModelState("loading", "Generating");
  setStatus("Generating result...");

  try {
    const generationOptions =
      tool === "reply"
        ? {
            max_new_tokens: lengthTokenBudget[length] || 220,
            do_sample: false,
            repetition_penalty: 1.06,
            eos_token_id: 151645,
          }
        : {
            max_new_tokens: lengthTokenBudget[length] || 220,
            temperature: 0.5,
            do_sample: true,
            repetition_penalty: 1.08,
            eos_token_id: 151645,
          };
    const result = await state.llmPipe(prompt, generationOptions);

    const generated = extractAssistantText(result?.[0]?.generated_text || "", prompt);

    setStatus("Streaming output...");
    await streamTextToOutput(generated || "No output generated.");
    recordRun();
    saveHistoryItem({
      tool,
      tone,
      length,
      context,
      source,
      output: generated || "No output generated.",
      ts: Date.now(),
    });
    setModelState("ready", "Model ready");
    setStatus("Done");
  } catch (error) {
    ui.outputBox.textContent = `Generation failed: ${error.message}`;
    updateCount();
    setModelState("error", "Generation error");
    setStatus("Generation failed");
  } finally {
    setBusy(false);
  }
}

function clearAll() {
  state.streamJob += 1;
  ui.sourceInput.value = "";
  ui.contextInput.value = "";
  ui.outputBox.textContent = "Your result will appear here.";
  saveDraft();
  updateCount();
  setStatus(state.modelReady ? "Ready" : "Load the model to start");
}

function copyOutput() {
  const text = ui.outputBox.textContent || "";
  navigator.clipboard.writeText(text);
  setStatus("Output copied");
}

function applyTemplate() {
  const key = ui.templateSelect.value;
  if (!key || !templates[key]) {
    setStatus("Select a template first");
    return;
  }
  const preset = templates[key];
  ui.toolSelect.value = preset.tool;
  ui.toneSelect.value = preset.tone;
  ui.contextInput.value = preset.context;
  ui.sourceInput.value = preset.source;
  saveDraft();
  setStatus("Template applied");
  if (ui.autoRunTemplate.checked && state.modelReady && !state.busy) {
    runTool().catch(() => undefined);
  }
}

function downloadOutput() {
  const content = ui.outputBox.textContent || "";
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  a.href = url;
  a.download = `qwendesk-output-${stamp}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setStatus("Output downloaded");
}

function renderHistory() {
  ui.historyList.innerHTML = "";
  if (state.history.length === 0) {
    ui.historyList.innerHTML = `<div class="history-item"><div class="history-text">No history yet.</div></div>`;
    return;
  }
  for (const item of state.history) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "history-item";
    const date = new Date(item.ts);
    el.innerHTML = `<div class="history-meta">${item.tool} · ${item.tone} · ${item.length} · ${date.toLocaleString()}</div>
      <div class="history-text">${esc(item.output)}</div>`;
    el.addEventListener("click", () => restoreHistoryItem(item));
    ui.historyList.appendChild(el);
  }
}

function saveHistoryItem(item) {
  state.history.unshift(item);
  state.history = state.history.slice(0, HISTORY_LIMIT);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history));
  renderHistory();
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    state.history = raw ? JSON.parse(raw) : [];
  } catch (_) {
    state.history = [];
  }
  renderHistory();
}

function restoreHistoryItem(item) {
  state.streamJob += 1;
  ui.toolSelect.value = item.tool;
  ui.toneSelect.value = item.tone;
  ui.lengthSelect.value = item.length || "balanced";
  ui.contextInput.value = item.context || "";
  ui.sourceInput.value = item.source || "";
  ui.outputBox.textContent = item.output || "";
  updateCount();
  setStatus("History item restored");
}

function clearHistory() {
  state.history = [];
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  setStatus("History cleared");
}

function markTodayDone() {
  const today = getTodayKey();
  const set = new Set(state.usage.completedDays || []);
  set.add(today);
  state.usage.completedDays = Array.from(set).slice(-90);
  saveUsage();
  setStatus("Great work. Today marked complete.");
}

ui.loadModelBtn.addEventListener("click", () => {
  loadModel().catch(() => undefined);
});

ui.runBtn.addEventListener("click", () => {
  runTool().catch(() => undefined);
});

ui.clearBtn.addEventListener("click", clearAll);
ui.copyBtn.addEventListener("click", copyOutput);
ui.downloadBtn.addEventListener("click", downloadOutput);
ui.applyTemplateBtn.addEventListener("click", applyTemplate);
ui.clearHistoryBtn.addEventListener("click", clearHistory);
ui.markDoneBtn.addEventListener("click", markTodayDone);
ui.inspireCards.forEach((btn) => {
  btn.addEventListener("click", () => {
    ui.templateSelect.value = btn.dataset.template;
    applyTemplate();
  });
});
[
  ui.toolSelect,
  ui.toneSelect,
  ui.lengthSelect,
  ui.autoRunTemplate,
  ui.dailyGoalInput,
].forEach((el) => {
  el.addEventListener("change", () => {
    saveSettings();
    updateUsageUI();
  });
});
ui.contextInput.addEventListener("input", saveDraft);
ui.sourceInput.addEventListener("input", saveDraft);

ui.sourceInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    runTool().catch(() => undefined);
  }
});
ui.modelSelect.addEventListener("change", () => {
  saveSettings();
  if (state.modelReady && !state.busy) {
    state.modelReady = false;
    if (state.llmPipe) {
      state.llmPipe.dispose().catch(() => undefined);
      state.llmPipe = null;
    }
    setModelState("idle", "Model idle");
    setStatus("Model changed. Load model again.");
    setBusy(false);
  }
});

window.addEventListener("beforeunload", () => {
  if (state.llmPipe) {
    state.llmPipe.dispose().catch(() => undefined);
  }
});

setModelState("idle", "Model idle");
setStatus("Load the model to start");
setBusy(false);
loadSettings();
loadUsage();
loadDraft();
loadHistory();
updateCount();
