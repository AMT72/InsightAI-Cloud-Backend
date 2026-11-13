
import os
import requests
from flask import Blueprint, request, jsonify

guide_bp = Blueprint("guide_reasoner", __name__)

MODEL = "qwen/qwen3.8-27b"


def proximity_word(value):
    if value >= 0.85:
        return "very close"
    if value >= 0.65:
        return "close"
    if value >= 0.40:
        return "nearby"
    return "far"


@guide_bp.post("/guide/reason")
def guide_reason():
    data = request.get_json(silent=True) or {}
    frame = data.get("frame")
    scene = data.get("scene") or {}
    memory = data.get("memory") or {}

    if not frame or not frame.startswith("data:image/jpeg;base64,"):
        return jsonify(error="Missing JPEG image"), 400

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return jsonify(error="OpenRouter key not configured"), 503

    objects = [
        (obj.get("label", "Generic Obstacle"),
         proximity_word(float(obj.get("proximity", 0))))
        for obj in scene.get("objects", [])
    ]

    free_space = float(scene.get("freeSpaceRatio", 0))
    path_state = (
        "clear" if free_space >= 0.70
        else "partly blocked" if free_space >= 0.40
        else "blocked"
    )

    prompt = f"""You are a real-time voice navigation assistant for a blind user.

Goal: {memory.get('user_goal', '')}
Previous guidance: {memory.get('previous_guidance')}
Recent context: {memory.get('recent_context', [])}
Important landmarks: {memory.get('key_landmarks', [])}
Detected objects: {objects}
Path state: {path_state}

OUTPUT RULES:
- Output ONE short instruction only.
- Prefer 3 to 10 words.
- Give only the immediate action the user should take now.
- Mention only an object that directly affects movement.
- Never describe the whole scene.
- Never explain your reasoning.
- Never mention percentages, scores, confidence, depth values, or internal measurements.
- Never use headings such as "analysis", "situation", or "current status".
- Never output multiple paragraphs.
- Use plain, direct English suitable for immediate TTS.

Good examples:
"Stop. Person ahead."
"Move slightly right."
"Continue straight."
"Door ahead. Keep left."

Return only the final spoken instruction."""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 60,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": frame}},
                    ],
                }],
            },
            timeout=25,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty model response")

        return jsonify(text=answer.strip(), model=MODEL)

    except Exception as exc:
        print("Guide Qwen error:", exc)
        return jsonify(error="Guide reasoning unavailable"), 502
