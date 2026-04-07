# QwenDesk

Text-first local AI work assistant for high-value daily writing tasks.

## Why This Scope

The prior multi-model audio flow was heavier and less reliable on some machines.  
This version uses one practical local model:

- `onnx-community/Qwen2.5-0.5B-Instruct`

That keeps startup simpler while still solving common user pain.

## Features

- `Rewrite` for cleaner, clearer writing
- `Summarize` into summary, key points, and action items
- `Reply Generator` with short, balanced, and detailed options
- `Quick templates` for common workflows:
  - Sales follow-up reply
  - Customer support apology
  - Interview follow-up email
  - Meeting recap with actions

## Tech

- Static app: `index.html`, `styles.css`, `app.js`
- Transformers.js via CDN
- Qwen running locally in browser

## Run Locally

```bash
python3 -m http.server 8080
```

Open [http://localhost:8080](http://localhost:8080).

## GitHub Pages

Deploy settings:

- Branch: `main`
- Folder: `/ (root)`

Published URL format:

- `https://<username>.github.io/<repo>/`

## Maintainer

- GitHub: [@aamar-shahzad](https://github.com/aamar-shahzad)
- Repository: [aamar-shahzad/langunage](https://github.com/aamar-shahzad/langunage)

## Next Ideas

- persistent local history of outputs
- saved custom templates
- optional STT/TTS as separate opt-in modules
