# InsightAI Cloud — Read / Ask only

This is a separate minimal cloud backend based on the WebSocket protocol used by InsightAI Mobile Phase 3. It does **not** run Colab or download model weights. The original source code was inspected but is not modified. The original Read/Ask prompts were not ported verbatim: the cloud backend has new safety-conscious prompts, so compare output quality before replacing the original.

## Local test

```powershell
py -m pip install -r requirements-cloud.txt
$env:GROQ_API_KEY="YOUR_KEY"
py cloud_app.py
```

Open http://localhost:8001/health. WebSocket endpoint: `ws://localhost:8001/stream`.

## Render

Create a new **Python Web Service** from a GitHub repository containing these files.
- Build command: `pip install -r requirements-cloud.txt`
- Start command: `gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 --bind 0.0.0.0:$PORT cloud_app:app`
- Environment variable: `GROQ_API_KEY` (secret). Optional: `GROQ_MODEL`.
- Health endpoint: `/health`

Render Free sleeps when idle; cold starts and API quotas affect availability. This setup is not suitable for safety-critical navigation.

## Mobile

In the mobile project `app/index.tsx`, change the `SERVER_URL` constant to your actual Render `https://...onrender.com` address (without `/stream`). The mobile code appends `/stream` and converts HTTPS to WSS. Rebuild/restart Expo as appropriate. Phase 3 Guide is marked “Coming next” and not implemented.

## Important

- Never commit `.env`, keys, or the original archive containing `.env`.
- The original ZIP contains a `.env`; rotate any secrets if they have been exposed.
- Do not deploy the heavy original `modelV6.py` and `guide_pipeline.py` on Render Free; they depend on Torch/YOLO/MiDaS and exceed the intended lightweight scope.
- This backend deliberately does **not** claim Guide capability. Test Read/Ask accuracy and latency on representative Arabic/English images before production.
