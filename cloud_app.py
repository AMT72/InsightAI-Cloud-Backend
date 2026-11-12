"""InsightAI cloud API for Read/Ask. Guide is intentionally unavailable."""
import base64
import io
import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request
from flask_sock import Sock
from groq import Groq
from PIL import Image

app = Flask(__name__)
sock = Sock(app)
logging.basicConfig(level=logging.INFO)
MODEL = os.getenv('GROQ_MODEL', 'qwen/qwen3.8-27b')
MAX_IMAGE_SIDE = int(os.getenv('MAX_IMAGE_SIDE', '1024'))
READ_PROMPT = ("You are InsightAI Read Mode, an assistive reading tool. "
               "Read all visible text accurately in its original language and order. "
               "Do not invent missing or unclear text. If there is no legible text, say NO_TEXT. "
               "Return only the reading, without introductory commentary.")
ASK_PROMPT = ("You are InsightAI Ask Mode, assisting a person with visual impairment. "
              "Answer the user's question based only on the provided image. "
              "Be concise, clear and specific. If something is unclear or not visible, say so. "
              "Do not invent details or assert distances you cannot measure.")

@app.get('/health')
def health():
    ready = bool(os.getenv('GROQ_API_KEY'))
    return jsonify(status='ok' if ready else 'not_configured', model=MODEL,
                   local_model={'ready':ready, 'loading':False,
                                'phase':'ready' if ready else 'unavailable',
                                'error':'' if ready else 'GROQ_API_KEY is missing'},
                   guide_available=False), 200 if ready else 503

def analyze(mode, frame, question=''):
    if mode not in ('read','ask'):
        raise ValueError('Unsupported mode')
    if not frame:
        raise ValueError('No image provided')
    raw = frame.split(',',1)[-1]
    image_bytes = base64.b64decode(raw, validate=True)
    if len(image_bytes) > 8 * 1024 * 1024:
        raise ValueError('Image exceeds 8MB')
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert('RGB')
        image.thumbnail((MAX_IMAGE_SIDE,MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer,format='JPEG',quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    prompt = READ_PROMPT if mode == 'read' else ASK_PROMPT + '\nUser question: ' + (question or 'Describe what is visible.')
    response = Groq(api_key=os.environ['GROQ_API_KEY'],timeout=55).chat.completions.create(
        model=MODEL,
        messages=[{'role':'user','content':[
            {'type':'text','text':prompt},
            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+encoded}},
        ]}], max_completion_tokens=500, temperature=0,
    )
    return (response.choices[0].message.content or '').strip()


@app.post('/analyze')
def analyze_http():
    """One-shot image analysis: independent of persistent WebSocket connections."""
    if not os.getenv('GROQ_API_KEY'):
        return jsonify(error='Server GROQ_API_KEY is not configured'), 503
    if request.content_length and request.content_length > 12 * 1024 * 1024:
        return jsonify(error='Request too large'), 413
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error='Expected JSON object'), 400
    mode = payload.get('mode')
    frame = payload.get('frame')
    question = payload.get('question', '')
    req_id = str(payload.get('request_id') or uuid.uuid4().hex)[:128]
    if mode not in ('read', 'ask') or not isinstance(frame, str) or not isinstance(question, str):
        return jsonify(error='Invalid mode, frame, or question', request_id=req_id), 400
    try:
        result = analyze(mode, frame, question)
        return jsonify(request_id=req_id, text=result or 'I could not identify the content clearly.', mode=mode)
    except (ValueError, base64.binascii.Error) as exc:
        return jsonify(error=str(exc), request_id=req_id), 400
    except Exception as exc:
        logging.exception('HTTP inference failed request_id=%s', req_id)
        return jsonify(error=f'Cloud analysis failed ({type(exc).__name__}). Check Render logs.', request_id=req_id), 502

@sock.route('/stream')
def stream(ws):
    session = uuid.uuid4().hex
    send_lock = threading.Lock()
    active = True
    busy = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1)
    latest = None
    last_text = ''
    def send(message):
        with send_lock:
            try:
                ws.send(json.dumps(message,ensure_ascii=False))
            except Exception:
                pass
    def job(mode, frame, question):
        nonlocal last_text
        try:
            text = analyze(mode,frame,question)
            if not text:
                text = 'I could not identify the content clearly.'
            last_text = text
            send({'type':'description','mode':mode,'text':text,'session_id':session})
        except Exception as exc:
            logging.exception('Inference failed')
            send({'type':'error','message':f'Analysis failed: {type(exc).__name__}: {exc}'})
        finally:
            busy.release()
    try:
        while active:
            raw = ws.receive()
            if raw is None:
                break
            if isinstance(raw,bytes):
                latest = 'data:image/jpeg;base64,'+base64.b64encode(raw).decode('ascii')
                continue
            try:
                msg = json.loads(raw)
            except (ValueError,TypeError):
                send({'type':'error','message':'Invalid JSON'})
                continue
            kind = msg.get('type')
            if kind == 'start':
                send({'type':'status','message':'Session started'})
                send({'type':'model_status','local_model':{'ready':bool(os.getenv('GROQ_API_KEY')),
                     'loading':False,'phase':'ready' if os.getenv('GROQ_API_KEY') else 'unavailable','error':''}})
            elif kind == 'frame':
                latest = msg.get('data') or latest
            elif kind in ('read','ask'):
                if not os.getenv('GROQ_API_KEY'):
                    send({'type':'error','message':'Server GROQ_API_KEY is not configured'})
                elif not busy.acquire(blocking=False):
                    send({'type':'error','message':'Previous request is still processing'})
                else:
                    executor.submit(job,kind,msg.get('frame') or latest,msg.get('question',''))
                    send({'type':'status','message':'Analyzing image…'})
            elif kind in ('guide','mode_change') and (kind == 'guide' or msg.get('mode') == 'guidance'):
                send({'type':'error','message':'Guide is not enabled in this experimental cloud backend. Do not rely on it for navigation.'})
            elif kind == 'repeat':
                if last_text:
                    send({'type':'description','text':last_text,'session_id':session})
            elif kind == 'stop':
                active = False
    finally:
        executor.shutdown(wait=False,cancel_futures=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','8001')))
