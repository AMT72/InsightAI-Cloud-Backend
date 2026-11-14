# InsightAI Cloud Backend

Flask backend for the InsightAI mobile app (hosted on Render).

| Endpoint | Used by | Provider / model |
|---|---|---|
| `POST /analyze` | Read, Ask | Groq — `qwen/qwen3.8-27b` (override with `GROQ_MODEL`) |
| `POST /guide/reason` | Guide | OpenRouter — `anthropic/claude-sonnet-4` (OpenAI-compatible client, `chat.completions.create`) |
| `GET /health` | status check | reports `guide_available` when `OPENROUTER_API_KEY` is set |
| `WS /stream` | connection status / keep-alive | Read/Ask legacy protocol; Guide is not served here |

Guide perception (YOLOv8n + MiDaS Small + safety check) runs **on the phone**. The backend only receives the frame, the scene summary and the Guide memory, and returns one short spoken instruction.

## Environment variables

| Name | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | yes | Read / Ask |
| `OPENROUTER_API_KEY` | yes (for Guide) | Do not use `ANTHROPIC_API_KEY` |
| `GROQ_MODEL` | no | default `qwen/qwen3.8-27b` |
| `MAX_IMAGE_SIDE` | no | default `1024` |

Copy `.env.example` to `.env` for local runs. Never commit `.env`.

## Local test

```powershell
py -m pip install -r requirements-cloud.txt
$env:GROQ_API_KEY="..."; $env:OPENROUTER_API_KEY="..."
py cloud_app.py
```

Open http://localhost:8001/health.

## Render

- Build command: `pip install -r requirements-cloud.txt`
- Start command: `gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 --bind 0.0.0.0:$PORT cloud_app:app`
- Environment: `GROQ_API_KEY`, `OPENROUTER_API_KEY` (and optionally `GROQ_MODEL`)
- Health check path: `/health`

Render Free sleeps when idle, so the first request after a pause can be slow.

## Notes

- Guide prompt is the original glasses prompt (`guide_reasoner.py`), filled from the request's `scene` and `memory`.
- The Read/Ask prompts in `cloud_app.py` are not the original glasses prompts.
- Experimental: relative depth is not a measured distance. Do not rely on Guide for independent walking without further validation.
