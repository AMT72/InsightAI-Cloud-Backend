
import os
import logging

from flask import Blueprint, request, jsonify
from openai import OpenAI

guide_bp = Blueprint("guide_bp", __name__)

# anthropic/claude-sonnet-4 was retired on OpenRouter (404). Override with the GUIDE_MODEL env var if needed.
MODEL = os.getenv("GUIDE_MODEL", "anthropic/claude-sonnet-4.6")


def proximity_word(value):
    if value >= 0.85:
        return "very close"
    if value >= 0.65:
        return "close"
    if value >= 0.40:
        return "nearby"
    return "far"


def build_guide_prompt(scene, memory):
    objects = [
        (
            obj.get("label", "Generic Obstacle"),
            proximity_word(float(obj.get("proximity") or 0))
        )
        for obj in scene.get("objects", [])
    ]

    free_space = float(scene.get("freeSpaceRatio") or 0)

    path_state = (
        "clear" if free_space >= 0.70
        else "partly blocked" if free_space >= 0.40
        else "blocked"
    )

    # Original InsightAI glasses prompt
    prompt = f"""You are a real-time voice navigation assistant for a blind user.

Goal: {memory.get('user_goal', '')}
Previous guidance: {memory.get('previous_guidance')}
Recent context: {memory.get('recent_context')}
Important landmarks: {memory.get('key_landmarks')}
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

    return prompt


@guide_bp.post("/guide/reason")
def guide_reason():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify(error="Expected JSON object"), 400

    frame = data.get("frame")
    scene = data.get("scene") or {}
    memory = data.get("memory") or {}

    if not isinstance(frame, str) or not frame.startswith(
        "data:image/jpeg;base64,"
    ):
        return jsonify(error="Missing JPEG image"), 400

    if not isinstance(scene, dict) or not isinstance(memory, dict):
        return jsonify(error="Invalid scene or memory"), 400

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return jsonify(error="OpenRouter key not configured"), 503

    try:
        prompt = build_guide_prompt(scene, memory)

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=55.0,
            max_retries=0,
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": frame,
                            },
                        },
                    ],
                }
            ],
            max_tokens=60,
        )

        answer = (
            response.choices[0].message.content or ""
        ).strip()

        if not answer:
            raise ValueError("Empty model response")

        print("CLAUDE GUIDE:", answer, flush=True)

        return jsonify(
            text=answer,
            model=MODEL,
        )

    except Exception as exc:
        logging.exception("Claude Guide reasoning failed")

        # Short diagnostic so the app log shows the real cause (invalid key, no credits, model error...).
        return jsonify(
            error="Claude Guide reasoning unavailable",
            detail=(
                f"{type(exc).__name__} "
                f"status={getattr(exc, 'status_code', None)}: "
                f"{str(exc)[:200]}"
            ),
        ), 502
