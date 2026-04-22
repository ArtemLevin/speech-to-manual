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
