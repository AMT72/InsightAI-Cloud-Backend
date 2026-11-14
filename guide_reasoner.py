import os
import json
import base64
from flask import Blueprint, request, jsonify
from anthropic import Anthropic

guide_bp = Blueprint("guide_bp", __name__)

# استخدم مفتاح Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# مودل Claude
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4")

client = Anthropic(api_key=ANTHROPIC_API_KEY)

GUIDE_SYSTEM_PROMPT = """
You are the navigation reasoning engine for InsightAI, an assistive system for visually impaired users.

Your job:
- Give VERY SHORT spoken navigation guidance.
- Be direct, practical, and safe.
- Focus only on movement guidance.
- Do not describe unnecessary visual details.
- Do not explain your reasoning.
- Do not mention confidence scores.
- Do not mention object lists unless needed for navigation.
- Prefer commands like:
  - "Continue straight."
  - "Move slightly left."
  - "Move slightly right."
  - "Stop. Obstacle ahead."
  - "Turn slightly left."
  - "Turn slightly right."
  - "Path is clear."

Rules:
- Output ONE short sentence only.
- Maximum about 8 words if possible.
- If obstacle is very close, prioritize safety.
- If path is clear, say "Continue straight."
- If there is an obstacle on one side, guide around it.
- If the scene is unclear, say "Move carefully forward."
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

        image_b64 = data.get("image_base64", "")
        spatial_data = data.get("spatial_data", {})
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
