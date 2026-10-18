import time
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

from _stage import load
import quantize
import tools  # Our AST math & Python interpreter

app = FastAPI(title="Production Agentic LLM API")

print("[agent] Loading 42.7M INT8 model...")
ckpt_path = "out_45m_run2/model_45m_chat_int8.pt"
model, cfg, _ = quantize.load_quantized(ckpt_path, device="cpu")
model.eval()

tok_mod = load("01_tokenizer")
gen_mod = load("06_generate")
tok = tok_mod.ByteLevelBPE.load("tokenizer_general/chat")
end_id = tok.token_to_id["<|im_end|>"]
specials = ["<|im_start|>", "<|im_end|>"]

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "my-agent-45m"
    messages: List[Message]
    temperature: Optional[float] = 0.6
    max_tokens: Optional[int] = 120

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_msg:
        raise HTTPException(status_code=400, detail="Missing user message")

    tool_used = None
    tool_output = None
    final_reply = None

    # --- AGENT ROUTER 1: Check for Exact Math ---
    math_q = tools.find_math_question(user_msg)
    if math_q and math_q.bare:
        tool_used = "calculator"
        tool_output = math_q.answer
        final_reply = f"[Agent Tool: Calculator] {math_q.answer}"

    # --- AGENT ROUTER 2: Check for Live Clock / Date ---
    elif any(k in user_msg.lower() for k in ["current date", "current time", "what time is it", "today's date", "date today"]):
        tool_used = "system_clock"
        now_str = tools.now().strftime("%A, %B %d, %Y at %H:%M:%S UTC")
        tool_output = now_str
        final_reply = f"[Agent Tool: System Clock] The current date and time is {now_str}."

    # --- AGENT ROUTER 3: Check for Word Counts & Python Strings ---
    else:
        string_q = tools.find_string_question(user_msg)
        if string_q and string_q.bare:
            tool_used = "python_interpreter"
            tool_output = string_q.answer
            final_reply = f"[Agent Tool: Python] {string_q.answer}"

    # --- AGENT ROUTER 4: General Knowledge Query (Route to Neural Network) ---
    if final_reply is None:
        prompt = (
            f"<|im_start|>system\nYou are a helpful, concise assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_ids = tok.encode_with_specials(prompt, specials)

        params = gen_mod.SamplingParams(
            temperature=req.temperature,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.25,
            seed=None
        )

        out_ids, _, _ = gen_mod.generate(
            model, prompt_ids, req.max_tokens, params,
            use_cache=True, eos_id=end_id, block_size=cfg.block_size
        )

        new_tokens = out_ids[len(prompt_ids):]
        if new_tokens and new_tokens[-1] == end_id:
            new_tokens = new_tokens[:-1]
        final_reply = tok.decode(new_tokens).replace("<|im_end|>", "").strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_reply},
            "finish_reason": "stop"
        }],
        "tool_used": tool_used,
        "tool_output": tool_output
    }