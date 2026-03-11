"""
OpenAI-compatible API server for rdna4-quant INT4 engine.

Exposes /v1/chat/completions and /v1/models endpoints.
Works with: Open WebUI, curl, openai Python SDK, any OpenAI-compatible client.

Usage:
  python api_server.py                          # default port 8000
  python api_server.py --port 8080              # custom port
  python api_server.py --host 0.0.0.0           # listen on all interfaces
  QUANT_DIR=./quantized_v4_gptq python api_server.py  # custom weights

Then connect with:
  curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "qwen3-14b-int4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
import json
import uuid
import threading
from pathlib import Path

# ---- Import engine (loads model into VRAM) ----
sys.path.insert(0, str(Path(__file__).parent))

# Import everything from the engine module
import int4_engine_v5 as engine

tokenizer = engine.tokenizer
device = engine.device

# Thread lock — model is single-GPU, one request at a time
_inference_lock = threading.Lock()


def _build_prompt(messages, enable_thinking=False):
    """Build Qwen3 ChatML prompt from messages list."""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # Skip thinking by default (faster, cleaner streaming)
    if enable_thinking:
        prompt_parts.append("<|im_start|>assistant\n")
    else:
        prompt_parts.append("<|im_start|>assistant\n<think>\n</think>\n")
    return "\n".join(prompt_parts)


def generate_response(messages, max_tokens=512, temperature=0.7, top_p=0.9,
                      stream=False, enable_thinking=False):
    """Generate a response from chat messages. Returns (text, usage_dict, timings)."""

    prompt = _build_prompt(messages, enable_thinking)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    S = input_ids.shape[1]

    kv = engine.KVCache(
        engine.num_layers, 1, S + max_tokens + 16,
        engine.num_kv_heads, engine.head_dim, device, dtype=torch.uint8
    )

    # Prefill
    t0 = time.time()
    with torch.no_grad():
        logits = engine.fast_prefill(input_ids, kv)
    prefill_time = time.time() - t0

    # Sample first token
    logits_last = logits[0, :].float()
    if temperature > 0:
        logits_last = logits_last / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = -float('inf')
            logits_last = torch.zeros_like(logits_last).scatter_(0, sorted_idx, sorted_logits)
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs.unsqueeze(0), 1)
    else:
        next_id = logits_last.argmax(-1, keepdim=True).unsqueeze(0)

    generated = [next_id.item()]
    pos = S

    # Decode loop
    t_decode = time.time()
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            hidden = F.embedding(next_id.view(1), engine.embed_w).view(1, 1, engine.hidden_size)
            logits = engine.decode_step_logits(hidden, pos, kv)
            logits_last = logits[0, 0, :].float()

            if temperature > 0:
                logits_last = logits_last / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[mask] = -float('inf')
                    logits_last = torch.zeros_like(logits_last).scatter_(0, sorted_idx, sorted_logits)
                probs = F.softmax(logits_last, dim=-1)
                next_id = torch.multinomial(probs.unsqueeze(0), 1)
            else:
                next_id = logits_last.argmax(-1, keepdim=True).unsqueeze(0)

            tok = next_id.item()
            generated.append(tok)
            pos += 1

            if tok == tokenizer.eos_token_id:
                break

            # Check for <|im_end|> in raw decode
            if len(generated) > 2:
                tail = tokenizer.decode(generated[-4:], skip_special_tokens=False)
                if '<|im_end|>' in tail:
                    break

    decode_time = time.time() - t_decode
    decode_tps = len(generated) / decode_time if decode_time > 0 else 0

    # Extract response text (strip think blocks)
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    text = raw
    if '</think>' in text:
        text = text[text.index('</think>') + len('</think>'):]
    text = text.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()

    usage = {
        "prompt_tokens": S,
        "completion_tokens": len(generated),
        "total_tokens": S + len(generated),
    }
    timings = {
        "prefill_ms": prefill_time * 1000,
        "decode_ms": decode_time * 1000,
        "decode_tps": decode_tps,
    }
    return text, usage, timings


def generate_stream(messages, max_tokens=512, temperature=0.7, top_p=0.9,
                    enable_thinking=False):
    """Generator that yields SSE chunks for streaming responses."""

    prompt = _build_prompt(messages, enable_thinking)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    S = input_ids.shape[1]

    kv = engine.KVCache(
        engine.num_layers, 1, S + max_tokens + 16,
        engine.num_kv_heads, engine.head_dim, device, dtype=torch.uint8
    )

    # Prefill
    with torch.no_grad():
        logits = engine.fast_prefill(input_ids, kv)

    # Sample first token
    logits_last = logits[0, :].float()
    if temperature > 0:
        logits_last = logits_last / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = -float('inf')
            logits_last = torch.zeros_like(logits_last).scatter_(0, sorted_idx, sorted_logits)
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs.unsqueeze(0), 1)
    else:
        next_id = logits_last.argmax(-1, keepdim=True).unsqueeze(0)

    generated = [next_id.item()]
    pos = S
    in_think = False
    think_done = False
    prev_text = ""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    with torch.no_grad():
        for _ in range(max_tokens - 1):
            hidden = F.embedding(next_id.view(1), engine.embed_w).view(1, 1, engine.hidden_size)
            logits = engine.decode_step_logits(hidden, pos, kv)
            logits_last = logits[0, 0, :].float()

            if temperature > 0:
                logits_last = logits_last / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[mask] = -float('inf')
                    logits_last = torch.zeros_like(logits_last).scatter_(0, sorted_idx, sorted_logits)
                probs = F.softmax(logits_last, dim=-1)
                next_id = torch.multinomial(probs.unsqueeze(0), 1)
            else:
                next_id = logits_last.argmax(-1, keepdim=True).unsqueeze(0)

            tok = next_id.item()
            generated.append(tok)
            pos += 1

            if tok == tokenizer.eos_token_id:
                break

            # Handle think blocks
            raw = tokenizer.decode(generated, skip_special_tokens=False)
            if '<|im_end|>' in raw:
                break

            if not think_done:
                if '<think>' in raw and '</think>' not in raw:
                    in_think = True
                    continue
                if in_think and '</think>' in raw:
                    in_think = False
                    think_done = True
                    idx = raw.index('</think>') + len('</think>')
                    clean = raw[idx:].replace('<|im_end|>', '').replace('<|im_start|>', '').strip()
                    if clean:
                        yield _sse_chunk(request_id, clean)
                    prev_text = clean
                    continue
                if in_think:
                    continue

            # Stream visible text
            clean = tokenizer.decode(generated, skip_special_tokens=True)
            if think_done:
                idx = raw.index('</think>') + len('</think>') if '</think>' in raw else 0
                clean = raw[idx:].replace('<|im_end|>', '').replace('<|im_start|>', '')

            delta = clean[len(prev_text):]
            if delta:
                yield _sse_chunk(request_id, delta)
            prev_text = clean

    # Final chunk
    yield _sse_done(request_id)


def _sse_chunk(request_id, content):
    """Format a single SSE data chunk."""
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3-14b-int4",
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(data)}\n\n"


def _sse_done(request_id):
    """Final SSE chunk with finish_reason=stop."""
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3-14b-int4",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"


# ---- FastAPI / Uvicorn server ----

def create_app():
    """Create FastAPI app with OpenAI-compatible endpoints."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="rdna4-quant API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "qwen3-14b-int4",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rdna4-quant",
                "permission": [],
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)
        enable_thinking = body.get("enable_thinking", False)

        if not messages:
            return JSONResponse({"error": {"message": "messages is required"}}, status_code=400)

        if stream:
            def event_stream():
                with _inference_lock:
                    for chunk in generate_stream(messages, max_tokens, temperature, top_p,
                                                 enable_thinking):
                        yield chunk
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Non-streaming
        with _inference_lock:
            text, usage, timings = generate_response(messages, max_tokens, temperature, top_p,
                                                     enable_thinking=enable_thinking)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen3-14b-int4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": usage,
            "timings": timings,  # extra: decode speed info
        }
        return JSONResponse(response)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": "qwen3-14b-int4",
            "vram_mb": int(torch.cuda.memory_allocated() / 1024**2),
            "gpu": torch.cuda.get_device_name(0),
        }

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI-compatible API server for rdna4-quant")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    args = parser.parse_args()

    app = create_app()

    print(f"\n{'='*60}")
    print(f"  rdna4-quant API server")
    print(f"  Model: qwen3-14b-int4")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print(f"  Endpoints:")
    print(f"    POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"    GET  http://{args.host}:{args.port}/v1/models")
    print(f"    GET  http://{args.host}:{args.port}/health")
    print(f"{'='*60}\n")

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
