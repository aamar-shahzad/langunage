# LinguaAI

Browser-based language practice app using local AI models for speech-to-text, response generation, and speech output.

## What This Project Does

- Runs fully in the browser (no backend required)
- Uses Whisper Tiny for STT
- Uses Qwen2.5-0.5B-Instruct for chat and corrections
- Uses SpeechT5 when available, with browser voice fallback
- Shows language scores (fluency, grammar, vocabulary) and tips
- Supports multiple use cases in one UI:
  - Language Coach
  - Interview Coach
  - Meeting Notes Assistant

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

## Model Loading Strategy

The app now supports profile-based loading so you do not need every model upfront:

- `Text only` - loads only the LLM
- `Voice input + browser voice` - loads LLM + STT
- `Full voice` - loads LLM + STT + SpeechT5 TTS

You can also load STT/TTS later with the optional model buttons after startup.

## Use Cases

Choose from the **Use case** selector in the app:

- `Language Coach`
  - Practice conversation, correction, or immersion mode
  - Receives coaching plus score/tip output
- `Interview Coach`
  - Simulates interviewer-style Q&A
  - Add role details in the context box (stack, level, round type)
- `Meeting Notes Assistant`
  - Turns spoken/typed input into structured notes
  - Produces summary, decisions, action items, and blockers

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
