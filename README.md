# speech-to-manual

## CLI quality controls

The CLI supports per-stage model overrides and generation presets.

### Profiles

- `--profile default` keeps current defaults from config.
- `--profile balanced` uses stronger local defaults for quality/speed balance.
- `--profile quality` uses maximum-quality local defaults (can be much slower).

### Per-stage model overrides

- `--model-clean`
- `--model-plan`
- `--model-draft`
- `--model-latex`
- `--model-latex-fix`

### Per-stage output length overrides

- `--num-predict-clean`
- `--num-predict-plan-prep`
- `--num-predict-plan`
- `--num-predict-draft`
- `--num-predict-latex`
- `--num-predict-latex-fix`

### Example

```bash
speech-to-manual audio ./audio/slova.mp3 \
  --profile balanced \
  --model-latex qwen2.5-coder:14b \
  --model-latex-fix qwen2.5-coder:14b \
  --num-predict-latex 3600 \
  --num-predict-latex-fix 3600 \
  --whisper-model medium \
  --beam-size 8
```

## OpenAI-compatible API mode

You can switch from local Ollama to any OpenAI-compatible endpoint.

### Required flags

- `--provider openai_compat`
- `--api-base-url https://your-provider/v1`
- `--api-key <token>` or `--api-key-env OPENAI_API_KEY`

### Example

```bash
export OPENAI_API_KEY="YOUR_TOKEN"

speech-to-manual text_file ./texts/slova.txt \
  --provider openai_compat \
  --api-base-url https://api.openai.com/v1 \
  --model-clean gpt-4.1-mini \
  --model-plan gpt-4.1 \
  --model-draft gpt-4.1 \
  --model-latex gpt-4.1 \
  --model-latex-fix gpt-4.1
```
