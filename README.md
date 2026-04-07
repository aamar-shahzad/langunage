# LinguaAI

Browser-based language practice app using local AI models for speech-to-text, response generation, and speech output.

## What This Project Does

- Runs fully in the browser (no backend required)
- Uses Whisper Tiny for STT
- Uses Qwen2.5-0.5B-Instruct for chat and corrections
- Uses SpeechT5 when available, with browser voice fallback
- Shows language scores (fluency, grammar, vocabulary) and tips

## Project Structure

- `index.html` - app markup and sections
- `styles.css` - all styles
- `app.js` - model loading, chat flow, audio handling, and UI state

## Runtime Safety Improvements Included

- Bounded in-memory chat history (prevents unbounded growth)
- Busy-state guard to prevent overlapping STT/LLM/TTS runs
- Cleanup hooks for media streams, animation frames, audio contexts, and observers
- Model disposal on page unload (`pipeline.dispose()`)
- Lightweight replay chips that avoid retaining per-message raw audio buffers

## Run Locally

Because this app uses ES modules from CDN and browser APIs, serve it over HTTP:

```bash
python3 -m http.server 8080
```

Then open:

- [http://localhost:8080](http://localhost:8080)

## GitHub Pages Deployment

1. Push this project to a public GitHub repository.
2. In the repository, open **Settings** -> **Pages**.
3. Under **Build and deployment**:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/ (root)`
4. Save and wait a few minutes for the first deployment.

The site URL will be:

- `https://<your-username>.github.io/<your-repo>/`

## Notes

- First model load can download several hundred MB; later loads come from browser cache.
- `index (1).html` is the original source snapshot. The active deployable app is `index.html`.
