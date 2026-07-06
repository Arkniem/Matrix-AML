"""OpenAI-compatible LLM client (stdlib only) for the agent seams.

Targets the CCHMC LiteLLM gateway + nemotron-3-super by default. Override via env:
  AMLMM_LLM_BASE_URL, AMLMM_LLM_API_KEY, AMLMM_LLM_MODEL.

nemotron-3-super is a reasoning model: we send the Llama-Nemotron control phrase
'detailed thinking off' as the system message and a generous max_tokens, then read
the JSON from message.content. Network/parse failures raise LLMError so callers
(AgentHooks) fall back to the deterministic default — a gateway outage never breaks a run.
"""
from __future__ import annotations
import os
import json
import urllib.request
import urllib.error

# On-network gateway (reachable from cluster compute nodes). For off-network use an
# SSH tunnel and set AMLMM_LLM_BASE_URL=http://localhost:4000/v1.
DEFAULT_BASE_URL = "http://bmiclusterp2.chmcres.cchmc.org:4000/v1"
DEFAULT_API_KEY = "sk-56374ba744be776b96b77e674da80827f06568c8ba574455"
DEFAULT_MODEL = "nemotron-3-super"
THINKING_OFF = "detailed thinking off"
# Per-agent output cap. 500k effectively means "do not truncate" -- the gateway
# clamps to the model's real limit and the model stops at EOS well before this.
DEFAULT_MAX_TOKENS = int(os.environ.get("AMLMM_LLM_MAX_TOKENS", "500000"))


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=180):
        self.base_url = (base_url or os.environ.get("AMLMM_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("AMLMM_LLM_API_KEY", DEFAULT_API_KEY)
        self.model = model or os.environ.get("AMLMM_LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    def _post(self, payload):
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"gateway request failed: {e}")

    def chat(self, prompt, system=THINKING_OFF, max_tokens=None, temperature=0.0,
             json_mode=False):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        d = self._post(payload)
        try:
            content = d["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {str(d)[:200]}")
        if not content:
            fr = (d.get("choices") or [{}])[0].get("finish_reason")
            raise LLMError(f"empty content (finish_reason={fr}); raise max_tokens")
        return content

    def chat_json(self, prompt, required=(), max_tokens=None, retries=2) -> dict:
        last = None
        p = prompt
        for _ in range(retries + 1):
            txt = self.chat(p, max_tokens=max_tokens, json_mode=True)
            try:
                obj = json.loads(txt)
                missing = [k for k in required if k not in obj]
                if not missing:
                    return obj
                last = f"missing keys {missing}"
            except json.JSONDecodeError as e:
                last = f"JSON parse error: {e}; got {txt[:160]!r}"
            p = (prompt + f"\n\nYour previous reply was invalid ({last}). "
                 f"Return ONLY a JSON object with keys {list(required)}.")
        raise LLMError(last or "no valid response")
