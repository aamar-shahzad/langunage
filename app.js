import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowRemoteModels = true;
env.useBrowserCache = true;

const $ = (id) => document.getElementById(id);

const ui = {
  loadModelBtn: $("load-model-btn"),
  runBtn: $("run-btn"),
  clearBtn: $("clear-btn"),
  copyBtn: $("copy-btn"),
  llmDot: $("llm-dot"),
  llmLabel: $("llm-label"),
  statusText: $("status-text"),
  toolSelect: $("tool-select"),
  toneSelect: $("tone-select"),
  templateSelect: $("template-select"),
  applyTemplateBtn: $("apply-template-btn"),
  contextInput: $("context-input"),
  sourceInput: $("source-input"),
  outputBox: $("output-box"),
  charCount: $("char-count"),
};

const state = {
  llmPipe: null,
  modelsLoading: false,
  modelReady: false,
  busy: false,
};

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

function buildSystemPrompt(tool, tone, context) {
  const base = `You are a practical writing assistant. Tone: ${tone}.`;
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
Generate 3 reply options in markdown:
1) Short
2) Balanced
3) Detailed
Keep each option ready to send.
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

async function loadModel() {
  if (state.modelReady || state.modelsLoading) return;
  state.modelsLoading = true;
  setBusy(true);
  setModelState("loading", "Model loading");
  setStatus("Loading Qwen2.5-0.5B-Instruct...");

  try {
    state.llmPipe = await pipeline("text-generation", "onnx-community/Qwen2.5-0.5B-Instruct", {
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
  const context = ui.contextInput.value.trim();
  const systemPrompt = buildSystemPrompt(tool, tone, context);
  const prompt = buildPrompt(systemPrompt, source);

  setBusy(true);
  setModelState("loading", "Generating");
  setStatus("Generating result...");

  try {
    const result = await state.llmPipe(prompt, {
      max_new_tokens: 320,
      temperature: 0.55,
      do_sample: true,
      repetition_penalty: 1.08,
      eos_token_id: 151645,
    });

    const generated = (result?.[0]?.generated_text || "")
      .replace(prompt, "")
      .replace(/<\|im_end\|>.*/s, "")
      .trim();

    ui.outputBox.innerHTML = esc(generated || "No output generated.");
    updateCount();
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
  ui.sourceInput.value = "";
  ui.contextInput.value = "";
  ui.outputBox.textContent = "Your result will appear here.";
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
  setStatus("Template applied");
}

ui.loadModelBtn.addEventListener("click", () => {
  loadModel().catch(() => undefined);
});

ui.runBtn.addEventListener("click", () => {
  runTool().catch(() => undefined);
});

ui.clearBtn.addEventListener("click", clearAll);
ui.copyBtn.addEventListener("click", copyOutput);
ui.applyTemplateBtn.addEventListener("click", applyTemplate);

window.addEventListener("beforeunload", () => {
  if (state.llmPipe) {
    state.llmPipe.dispose().catch(() => undefined);
  }
});

setModelState("idle", "Model idle");
setStatus("Load the model to start");
setBusy(false);
updateCount();
