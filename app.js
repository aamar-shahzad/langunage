import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowRemoteModels = true;
env.useBrowserCache = true;

const MAX_HISTORY_MESSAGES = 12;

const $ = (id) => document.getElementById(id);

const ui = {
  dotSTT: $('dot-stt'),
  lblSTT: $('lbl-stt'),
  psSTT: $('ps-stt'),
  dotLLM: $('dot-llm'),
  lblLLM: $('lbl-llm'),
  psLLM: $('ps-llm'),
  dotTTS: $('dot-tts'),
  lblTTS: $('lbl-tts'),
  psTTS: $('ps-tts'),
  statusMsg: $('status-msg'),
  micBtn: $('mic-btn'),
  micHintText: $('mic-hint-text'),
  micHintSub: $('mic-hint-sub'),
  transcriptArea: $('transcript-area'),
  transcriptBody: $('transcript-body'),
  responseArea: $('response-area'),
  responseBody: $('response-body'),
  feedbackBar: $('feedback-bar'),
  chatLog: $('chat-log'),
  textInput: $('text-input'),
  textSendBtn: $('text-send-btn'),
  waveformWrap: $('waveform-wrap'),
  waveCanvas: $('waveform'),
  loaderOverlay: $('loader-overlay'),
  loadModelsBtn: $('load-models-btn'),
  appInner: $('app-inner'),
  skipLoadBtn: $('skip-load-btn'),
  nativeLang: $('native-lang'),
  targetLang: $('target-lang'),
  level: $('level'),
  replayBtn: $('replay-btn'),
  welcomeBubble: $('welcome-bubble'),
};

const state = {
  sttPipe: null,
  llmPipe: null,
  ttsPipe: null,
  ttsAvailable: false,
  modelsLoaded: false,
  modelsLoading: false,
  isBusy: false,
  isRecording: false,
  conversationHistory: [],
  practiceMode: 'conversation',
  lastAudioBuffer: null,
  lastAudioSampleRate: 16000,
  monitorCtx: null,
  monitorAnalyser: null,
  monitorSource: null,
  monitorAnimation: null,
  playbackCtx: null,
  mediaRecorder: null,
  mediaStream: null,
  audioChunks: [],
  revealObserver: null,
};

function setDot(dot, lbl, ps, status, label) {
  const classes = { idle: 'idle', loading: 'loading', ready: 'ready', active: 'active', error: 'error' };
  dot.className = `ss-dot ${classes[status] || ''}`;
  ps.className = `pipe-status ps-${status}`;
  ps.textContent = status;
  if (label) lbl.textContent = label;
}

function setStatus(msg) {
  ui.statusMsg.textContent = msg;
}

function toast(msg, color = 'var(--a)') {
  const el = document.createElement('div');
  el.style.cssText = `position:fixed;bottom:1.5rem;left:50%;transform:translateX(-50%);
    background:var(--ink3);border:0.5px solid var(--line3);border-radius:var(--r-sm);
    padding:8px 18px;font-size:12px;font-family:var(--mono);color:${color};
    z-index:999;animation:msgIn 0.3s ease both;white-space:nowrap`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function escHtml(str) {
  return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function formatTime() {
  return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
}

function clampHistory() {
  if (state.conversationHistory.length > MAX_HISTORY_MESSAGES) {
    state.conversationHistory = state.conversationHistory.slice(-MAX_HISTORY_MESSAGES);
  }
}

function setBusy(nextBusy) {
  state.isBusy = nextBusy;
  const disabled = !state.modelsLoaded || nextBusy;
  ui.textSendBtn.disabled = disabled;
  ui.micBtn.disabled = disabled || state.isRecording;
}

function resetMicHint() {
  ui.micHintText.textContent = 'Click to start speaking';
  ui.micHintSub.textContent = 'Click once to record, click again to process';
  ui.micBtn.textContent = '🎙️';
}

function updateWelcome() {
  const lang = ui.targetLang.value;
  const level = ui.level.value;
  const modeDesc = {
    conversation: 'have a natural conversation',
    correction: 'correct your grammar and suggest better phrasing',
    immersion: 'respond only in the target language (no translations)',
  };
  ui.welcomeBubble.textContent = `Bonjour! I'm your ${lang} coach. We'll ${modeDesc[state.practiceMode]}. You're at ${level} level. Go ahead — speak or type something in ${lang}.`;
}

function closeContext(ctx) {
  if (ctx && ctx.state !== 'closed') {
    ctx.close().catch(() => undefined);
  }
}

function cleanupRecordingResources() {
  if (state.monitorAnimation) {
    cancelAnimationFrame(state.monitorAnimation);
    state.monitorAnimation = null;
  }
  if (state.monitorSource) {
    state.monitorSource.disconnect();
    state.monitorSource = null;
  }
  if (state.monitorAnalyser) {
    state.monitorAnalyser.disconnect();
    state.monitorAnalyser = null;
  }
  closeContext(state.monitorCtx);
  state.monitorCtx = null;
  ui.waveformWrap.style.display = 'none';
}

function stopStreamTracks() {
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((t) => t.stop());
    state.mediaStream = null;
  }
}

function maybeInitPlaybackContext() {
  if (!state.playbackCtx || state.playbackCtx.state === 'closed') {
    state.playbackCtx = new AudioContext();
  }
}

function playFloat32Audio(float32, sampleRate) {
  if (!float32) return;
  maybeInitPlaybackContext();
  const ctx = state.playbackCtx;
  const buf = ctx.createBuffer(1, float32.length, sampleRate);
  buf.copyToChannel(float32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
}

function fallbackTTS(text) {
  if (!('speechSynthesis' in window) || !text) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const langMap = {
    French: 'fr-FR',
    Spanish: 'es-ES',
    German: 'de-DE',
    Japanese: 'ja-JP',
    'Mandarin Chinese': 'zh-CN',
    Italian: 'it-IT',
    Portuguese: 'pt-BR',
    Korean: 'ko-KR',
    Arabic: 'ar-SA',
    Hindi: 'hi-IN',
    Russian: 'ru-RU',
    English: 'en-US',
  };
  utterance.lang = langMap[ui.targetLang.value] || 'en-US';
  utterance.rate = 0.9;
  speechSynthesis.speak(utterance);
}

function addReplayChip(msgEl, text) {
  const bubble = msgEl.querySelector('.chat-bubble');
  if (!bubble) return;
  const chip = document.createElement('button');
  chip.type = 'button';
  chip.className = 'play-chip';
  chip.textContent = state.ttsAvailable ? '▶ replay voice' : '▶ play (browser voice)';
  chip.addEventListener('click', () => {
    if (state.ttsAvailable) {
      runTTS(text, null).catch(() => fallbackTTS(text));
    } else {
      fallbackTTS(text);
    }
  });
  bubble.appendChild(chip);
}

function addChatBubble(role, text) {
  const el = document.createElement('div');
  el.className = `chat-msg ${role}`;
  const displayText =
    role === 'ai'
      ? text.replace(
          /\[correction:\s*(.+?)\s*→\s*(.+?)\]/g,
          (_, original, fix) =>
            `<span class="original-err">${escHtml(original)}</span><span class="corrected">${escHtml(fix)}</span>`,
        )
      : escHtml(text);
  el.innerHTML = `
    <div class="chat-avatar ${role === 'ai' ? 'ai' : 'user'}">${role === 'ai' ? '🤖' : '👤'}</div>
    <div>
      <div class="chat-bubble ${role === 'ai' ? 'ai' : 'user'}">${displayText}</div>
      <div class="chat-meta">${role === 'ai' ? 'AI Coach' : 'You'} · ${formatTime()}</div>
    </div>`;
  ui.chatLog.appendChild(el);
  ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
  return el;
}

function showTranscript(text) {
  ui.transcriptArea.style.display = 'block';
  ui.transcriptBody.innerHTML = `<span style="color:var(--snow)">${escHtml(text)}</span>`;
  addChatBubble('user', text);
}

function showScores(fluency, grammar, vocab, tip) {
  ui.feedbackBar.style.display = 'block';
  const set = (id, textId, val) => {
    setTimeout(() => {
      $(id).style.width = `${val}%`;
      $(textId).textContent = `${val}/100`;
    }, 100);
  };
  set('sc-fluency', 'sc-fluency-n', fluency);
  set('sc-grammar', 'sc-grammar-n', grammar);
  set('sc-vocab', 'sc-vocab-n', vocab);
  $('feedback-tips').innerHTML = tip ? `<strong>Tip:</strong> ${escHtml(tip)}` : '';
}

async function runTTS(text, msgEl) {
  const cleanText = (text || '').replace(/<[^>]+>/g, '').replace(/\[.*?\]/g, '').slice(0, 300);
  if (!cleanText) return;

  if (state.ttsAvailable && state.ttsPipe) {
    setDot(ui.dotTTS, ui.lblTTS, ui.psTTS, 'active', 'TTS speaking');
    setStatus('Generating speech...');
    try {
      const out = await state.ttsPipe(cleanText, {
        speaker_embeddings:
          'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin',
      });
      state.lastAudioBuffer = out.audio;
      state.lastAudioSampleRate = out.sampling_rate || 16000;
      playFloat32Audio(out.audio, state.lastAudioSampleRate);
      setDot(ui.dotTTS, ui.lblTTS, ui.psTTS, 'ready', 'TTS ✓');
      if (msgEl) addReplayChip(msgEl, cleanText);
      return;
    } catch (_) {
      state.ttsAvailable = false;
      setDot(ui.dotTTS, ui.lblTTS, ui.psTTS, 'ready', 'TTS (browser)');
      toast('SpeechT5 failed, using browser voice', 'var(--speak)');
    }
  }

  fallbackTTS(cleanText);
  if (msgEl) addReplayChip(msgEl, cleanText);
}

async function runLLM(userText) {
  if (!state.llmPipe || state.isBusy) return;

  setBusy(true);
  ui.responseArea.style.display = 'block';
  ui.responseBody.innerHTML = '<span class="typing-cursor"></span>';
  setDot(ui.dotLLM, ui.lblLLM, ui.psLLM, 'active', 'LLM thinking');
  setStatus('AI is thinking...');

  const lang = ui.targetLang.value;
  const native = ui.nativeLang.value;
  const level = ui.level.value;

  const modeInstructions = {
    conversation:
      'Have a natural conversation. If the user makes minor grammar errors, gently note them inline with [correction: original → fix]. Keep it conversational and encouraging.',
    correction:
      `Focus on correcting the user's ${lang}. First show their corrected sentence, then explain each error briefly, then give a natural reply. Use [correction: original → fix] markup inline.`,
    immersion: `Respond only in ${lang}. Do not use ${native}. If the user writes in ${native}, still respond in ${lang}. Keep it brief and natural.`,
  };

  const systemPrompt = `You are a friendly, encouraging ${lang} language coach.
The user is a ${level} ${lang} learner whose native language is ${native}.
${modeInstructions[state.practiceMode]}
After your response, on a new line add: SCORES: fluency=N grammar=N vocab=N (N is 0-100)
Then on a new line: TIP: one short actionable tip.
Keep responses concise (2-4 sentences max for conversation).`;

  state.conversationHistory.push({ role: 'user', content: userText });
  clampHistory();

  const messages = [{ role: 'system', content: systemPrompt }, ...state.conversationHistory.slice(-8)];
  let prompt = '';
  for (const message of messages) {
    if (message.role === 'system') prompt += `<|im_start|>system\n${message.content}<|im_end|>\n`;
    else if (message.role === 'user') prompt += `<|im_start|>user\n${message.content}<|im_end|>\n`;
    else prompt += `<|im_start|>assistant\n${message.content}<|im_end|>\n`;
  }
  prompt += '<|im_start|>assistant\n';

  try {
    const result = await state.llmPipe(prompt, {
      max_new_tokens: 280,
      temperature: 0.7,
      do_sample: true,
      repetition_penalty: 1.1,
      eos_token_id: 151645,
    });
    const fullResponse = (result[0]?.generated_text || '')
      .replace(prompt, '')
      .replace(/<\|im_end\|>.*/s, '')
      .trim();

    const scoresMatch = fullResponse.match(/SCORES:\s*fluency=(\d+)\s+grammar=(\d+)\s+vocab=(\d+)/i);
    const tipMatch = fullResponse.match(/TIP:\s*(.+)/i);
    const cleanResponse = fullResponse.replace(/SCORES:.*$/m, '').replace(/TIP:.*$/m, '').trim();
    const displayResponse = cleanResponse.replace(
      /\[correction:\s*(.+?)\s*→\s*(.+?)\]/g,
      (_, original, fix) =>
        `<span class="original-err">${escHtml(original)}</span><span class="corrected">${escHtml(fix)}</span>`,
    );

    ui.responseBody.innerHTML = displayResponse || '<span class="area-placeholder">No response</span>';
    setDot(ui.dotLLM, ui.lblLLM, ui.psLLM, 'ready', 'LLM ✓');

    if (scoresMatch) {
      showScores(
        Number.parseInt(scoresMatch[1], 10),
        Number.parseInt(scoresMatch[2], 10),
        Number.parseInt(scoresMatch[3], 10),
        tipMatch ? tipMatch[1] : null,
      );
    }

    state.conversationHistory.push({ role: 'assistant', content: cleanResponse });
    clampHistory();

    const msgEl = addChatBubble('ai', cleanResponse);
    await runTTS(cleanResponse, msgEl);
    setStatus('Ready — your turn');
  } catch (error) {
    toast(`LLM error: ${error.message}`, '#ff5c5c');
    setDot(ui.dotLLM, ui.lblLLM, ui.psLLM, 'error', 'LLM ✗');
  } finally {
    setBusy(false);
    resetMicHint();
  }
}

function startWaveform() {
  const ctx = ui.waveCanvas.getContext('2d');
  const w = ui.waveCanvas.clientWidth;
  const h = ui.waveCanvas.clientHeight;
  ui.waveCanvas.width = Math.max(1, w);
  ui.waveCanvas.height = Math.max(1, h);

  const buffer = new Uint8Array(state.monitorAnalyser.frequencyBinCount);
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--a').trim() || '#ff7b54';

  const draw = () => {
    state.monitorAnimation = requestAnimationFrame(draw);
    state.monitorAnalyser.getByteTimeDomainData(buffer);
    ctx.clearRect(0, 0, ui.waveCanvas.width, ui.waveCanvas.height);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const sliceWidth = ui.waveCanvas.width / buffer.length;
    let x = 0;
    for (let i = 0; i < buffer.length; i += 1) {
      const v = buffer[i] / 128;
      const y = (v * ui.waveCanvas.height) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.stroke();
  };

  draw();
}

async function startRecording() {
  if (!state.modelsLoaded || state.isBusy || state.isRecording) return;
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    state.monitorCtx = new AudioContext();
    state.monitorAnalyser = state.monitorCtx.createAnalyser();
    state.monitorAnalyser.fftSize = 256;
    state.monitorSource = state.monitorCtx.createMediaStreamSource(state.mediaStream);
    state.monitorSource.connect(state.monitorAnalyser);

    ui.waveformWrap.style.display = 'block';
    startWaveform();

    state.mediaRecorder = new MediaRecorder(state.mediaStream);
    state.audioChunks = [];
    state.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) state.audioChunks.push(e.data);
    };
    state.mediaRecorder.onstop = () => {
      processAudio().catch((e) => {
        toast(`Audio processing failed: ${e.message}`, '#ff5c5c');
        setBusy(false);
        resetMicHint();
      });
    };
    state.mediaRecorder.start();

    state.isRecording = true;
    ui.micBtn.classList.add('recording');
    ui.micBtn.textContent = '⏹️';
    ui.micHintText.textContent = 'Recording... click to stop';
    ui.micHintSub.textContent = 'Speak clearly into your mic';
  } catch (_) {
    toast('Microphone access denied', '#ff5c5c');
    cleanupRecordingResources();
    stopStreamTracks();
  }
}

function stopRecording() {
  if (!state.isRecording) return;
  state.isRecording = false;
  ui.micBtn.classList.remove('recording');
  ui.micBtn.textContent = '🎙️';
  ui.micHintText.textContent = 'Processing...';
  ui.micHintSub.textContent = 'Transcribing with Whisper...';

  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
    state.mediaRecorder.stop();
  } else {
    cleanupRecordingResources();
    stopStreamTracks();
  }
}

async function processAudio() {
  cleanupRecordingResources();
  stopStreamTracks();

  if (!state.sttPipe) {
    toast('STT model not loaded', '#ff5c5c');
    resetMicHint();
    return;
  }
  if (state.audioChunks.length === 0) {
    toast('No speech captured');
    resetMicHint();
    return;
  }
  if (state.isBusy) return;

  setBusy(true);
  setDot(ui.dotSTT, ui.lblSTT, ui.psSTT, 'active', 'STT running');
  setStatus('Transcribing speech...');

  try {
    const blob = new Blob(state.audioChunks, { type: 'audio/webm' });
    const arrayBuffer = await blob.arrayBuffer();
    const tmpCtx = new AudioContext({ sampleRate: 16000 });
    const decoded = await tmpCtx.decodeAudioData(arrayBuffer);
    const float32 = decoded.getChannelData(0);
    closeContext(tmpCtx);

    const result = await state.sttPipe(float32, {
      language: null,
      task: 'transcribe',
      chunk_length_s: 30,
      return_timestamps: false,
    });

    const transcript = (result.text || '').trim();
    if (!transcript) {
      toast('No speech detected');
      setDot(ui.dotSTT, ui.lblSTT, ui.psSTT, 'ready', 'STT ✓');
      setBusy(false);
      resetMicHint();
      return;
    }

    showTranscript(transcript);
    setDot(ui.dotSTT, ui.lblSTT, ui.psSTT, 'ready', 'STT ✓');
    await runLLM(transcript);
  } catch (error) {
    toast(`Transcription failed: ${error.message}`, '#ff5c5c');
    setDot(ui.dotSTT, ui.lblSTT, ui.psSTT, 'error', 'STT ✗');
    setBusy(false);
    resetMicHint();
  }
}

async function loadAllModels() {
  if (state.modelsLoading) return;
  state.modelsLoading = true;
  setStatus('Loading models...');

  try {
    setDot(ui.dotSTT, ui.lblSTT, ui.psSTT, 'loading', 'STT loading');
    $('ld-stt-detail').textContent = 'Downloading whisper-tiny...';
    state.sttPipe = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
      progress_callback: (p) => {
        if (p.status === 'progress' && p.progress) {
          $('ld-stt-bar').style.width = `${Math.round(p.progress)}%`;
          $('ld-stt-detail').textContent = `${p.file || ''} ${Math.round(p.progress)}%`;
        }
      },
    });
    $('ld-stt-bar').style.width = '100%';
    $('ld-stt-detail').textContent = 'Ready';
    setDot(ui.dotSTT, ui.lblSTT, ui.psSTT, 'ready', 'STT ✓');
  } catch (error) {
    $('ld-stt-detail').textContent = `Failed: ${error.message}`;
    setDot(ui.dotSTT, ui.lblSTT, ui.psSTT, 'error', 'STT ✗');
    toast(`STT load failed: ${error.message}`, '#ff5c5c');
  }

  try {
    setDot(ui.dotLLM, ui.lblLLM, ui.psLLM, 'loading', 'LLM loading');
    $('ld-llm-detail').textContent = 'Downloading Qwen2.5-0.5B...';
    state.llmPipe = await pipeline('text-generation', 'onnx-community/Qwen2.5-0.5B-Instruct', {
      progress_callback: (p) => {
        if (p.status === 'progress' && p.progress) {
          $('ld-llm-bar').style.width = `${Math.round(p.progress)}%`;
          $('ld-llm-detail').textContent = `${p.file || ''} ${Math.round(p.progress)}%`;
        }
      },
    });
    $('ld-llm-bar').style.width = '100%';
    $('ld-llm-detail').textContent = 'Ready';
    setDot(ui.dotLLM, ui.lblLLM, ui.psLLM, 'ready', 'LLM ✓');
  } catch (error) {
    $('ld-llm-detail').textContent = `Failed: ${error.message}`;
    setDot(ui.dotLLM, ui.lblLLM, ui.psLLM, 'error', 'LLM ✗');
    toast(`LLM load failed: ${error.message}`, '#ff5c5c');
  }

  try {
    setDot(ui.dotTTS, ui.lblTTS, ui.psTTS, 'loading', 'TTS loading');
    $('ld-tts-detail').textContent = 'Downloading SpeechT5...';
    ui.skipLoadBtn.style.display = 'block';
    state.ttsPipe = await pipeline('text-to-speech', 'Xenova/speecht5_tts', {
      quantized: false,
      progress_callback: (p) => {
        if (p.status === 'progress' && p.progress) {
          $('ld-tts-bar').style.width = `${Math.round(p.progress)}%`;
          $('ld-tts-detail').textContent = `${p.file || ''} ${Math.round(p.progress)}%`;
        }
      },
    });
    state.ttsAvailable = true;
    $('ld-tts-bar').style.width = '100%';
    $('ld-tts-detail').textContent = 'Ready';
    setDot(ui.dotTTS, ui.lblTTS, ui.psTTS, 'ready', 'TTS ✓');
  } catch (_) {
    state.ttsAvailable = false;
    $('ld-tts-detail').textContent = 'Using browser TTS fallback';
    setDot(ui.dotTTS, ui.lblTTS, ui.psTTS, 'ready', 'TTS (browser)');
  }

  ui.loaderOverlay.classList.remove('show');
  ui.skipLoadBtn.style.display = 'none';
  state.modelsLoading = false;
  state.modelsLoaded = Boolean(state.sttPipe && state.llmPipe);

  if (state.modelsLoaded) {
    ui.appInner.style.display = 'block';
    ui.loadModelsBtn.style.display = 'none';
    setBusy(false);
    setStatus(`Ready — speak or type in ${ui.targetLang.value}`);
    updateWelcome();
  } else {
    setStatus('Some models failed to load. Retry loading models.');
    setBusy(true);
  }
}

async function disposePipelines() {
  const pipes = [state.sttPipe, state.llmPipe, state.ttsPipe].filter(Boolean);
  await Promise.all(
    pipes.map((pipeRef) =>
      pipeRef.dispose().catch(() => {
        return undefined;
      }),
    ),
  );
}

function initReveal() {
  state.revealObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) entry.target.classList.add('in');
      }
    },
    { threshold: 0.1 },
  );
  document.querySelectorAll('.reveal').forEach((el) => state.revealObserver.observe(el));
}

function registerEvents() {
  ui.loadModelsBtn.addEventListener('click', () => {
    ui.loaderOverlay.classList.add('show');
    ui.appInner.style.display = 'block';
    loadAllModels().catch((e) => {
      ui.loaderOverlay.classList.remove('show');
      state.modelsLoading = false;
      toast(`Model load failed: ${e.message}`, '#ff5c5c');
    });
  });

  ui.skipLoadBtn.addEventListener('click', () => {
    state.ttsAvailable = false;
    setDot(ui.dotTTS, ui.lblTTS, ui.psTTS, 'ready', 'TTS (browser)');
    $('ld-tts-detail').textContent = 'Using browser voice';
    if (state.llmPipe) {
      ui.loaderOverlay.classList.remove('show');
      ui.skipLoadBtn.style.display = 'none';
      state.modelsLoaded = Boolean(state.sttPipe && state.llmPipe);
      setBusy(false);
      setStatus(`Ready — speak or type in ${ui.targetLang.value}`);
      updateWelcome();
    }
  });

  ui.micBtn.addEventListener('click', () => {
    if (state.isBusy || !state.modelsLoaded) return;
    if (!state.isRecording) startRecording().catch(() => undefined);
    else stopRecording();
  });

  ui.textSendBtn.addEventListener('click', async () => {
    if (state.isBusy || !state.modelsLoaded) return;
    const text = ui.textInput.value.trim();
    if (!text) return;
    ui.textInput.value = '';
    showTranscript(text);
    await runLLM(text);
  });

  ui.textInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      ui.textSendBtn.click();
    }
  });

  $('copy-transcript').addEventListener('click', () => {
    navigator.clipboard.writeText(ui.transcriptBody.innerText || '');
    toast('Copied');
  });

  $('copy-response').addEventListener('click', () => {
    navigator.clipboard.writeText(ui.responseBody.innerText || '');
    toast('Copied');
  });

  $('edit-transcript').addEventListener('click', () => {
    if (state.isBusy || !state.modelsLoaded) return;
    const current = ui.transcriptBody.innerText;
    const edited = prompt('Edit transcript:', current);
    if (!edited || !edited.trim()) return;
    const next = edited.trim();
    ui.transcriptBody.innerHTML = `<span style="color:var(--snow)">${escHtml(next)}</span>`;
    runLLM(next).catch(() => undefined);
  });

  ui.replayBtn.addEventListener('click', () => {
    const txt = ui.responseBody.innerText.replace(/▶.*/g, '').trim().slice(0, 300);
    if (state.lastAudioBuffer && state.ttsAvailable) {
      playFloat32Audio(state.lastAudioBuffer, state.lastAudioSampleRate);
      return;
    }
    fallbackTTS(txt);
  });

  document.querySelectorAll('.mode-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.mode-tab').forEach((item) => item.classList.remove('active'));
      tab.classList.add('active');
      state.practiceMode = tab.dataset.mode;
      if (state.modelsLoaded) updateWelcome();
    });
  });

  ui.targetLang.addEventListener('change', () => {
    if (state.modelsLoaded) updateWelcome();
  });
  ui.level.addEventListener('change', () => {
    if (state.modelsLoaded) updateWelcome();
  });

  window.addEventListener('beforeunload', () => {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
      state.mediaRecorder.stop();
    }
    cleanupRecordingResources();
    stopStreamTracks();
    speechSynthesis.cancel();
    closeContext(state.playbackCtx);
    state.revealObserver?.disconnect();
    disposePipelines().catch(() => undefined);
  });
}

setStatus('Click "Load AI Models" to start');
setBusy(true);
registerEvents();
initReveal();
