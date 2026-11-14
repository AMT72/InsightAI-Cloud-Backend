import os
import json
import base64
from flask import Blueprint, request, jsonify
from anthropic import Anthropic

guide_bp = Blueprint("guide_bp", __name__)

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "anthropic/claude-sonnet-4"

GUIDE_SYSTEM_PROMPT = """

You are a real-time voice navigation assistant for a blind user.

Goal: {mem['user_goal']}
Previous guidance: {mem['previous_guidance']}
Recent context: {mem['recent_context']}
Important landmarks: {mem['key_landmarks']}
Detected objects: {object_summary}
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

Return only the final spoken instruction.

"""

def build_guide_user_prompt(spatial_data, memory=None):
    memory = memory or {}

    prompt = f"""
Navigation state:
{json.dumps(spatial_data, ensure_ascii=False)}

Memory:
{json.dumps(memory, ensure_ascii=False)}

Return only one short navigation instruction.
"""
    return prompt.strip()


def call_claude_guide(image_b64, spatial_data, memory=None):
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is missing")

    user_prompt = build_guide_user_prompt(spatial_data, memory)

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64
            }
        },
        {
            "type": "text",
            "text": user_prompt
        }
    ]

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=60,
        temperature=0.2,
        system=GUIDE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    text = ""
    if response.content:
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text += block.text

    text = (text or "").strip()

    if not text:
        text = "Move carefully forward."

    return text


@guide_bp.route("/guide/reason", methods=["POST"])
def guide_reason():
    try:
        data = request.get_json(force=True)

        image_b64 = data.get("frame", "")
spatial_data = data.get("scene", {})
        memory = data.get("memory", {})

        if not image_b64:
            return jsonify({"error": "image_base64 is required"}), 400

        # إذا الصورة جاية بصيغة data URL
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]

        guide_text = call_claude_guide(
            image_b64=image_b64,
            spatial_data=spatial_data,
            memory=memory
        )

        print(f"CLAUDE GUIDE: {guide_text}")

        return jsonify({
            "ok": True,
            "text": guide_text
        })

    except Exception as e:
        print(f"CLAUDE GUIDE ERROR: {e}")
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500
