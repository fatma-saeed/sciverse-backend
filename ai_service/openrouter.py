from ai_service.guardrails import redact
"""Optional OpenRouter transport. Secrets are read only from environment."""
import os
import httpx
from fastapi import HTTPException

def configuration():
    return bool(os.getenv('OPENROUTER_API_KEY') and os.getenv('OPENROUTER_MODEL'))

def chat(messages):
    if not configuration():
        raise HTTPException(503, 'Configure OPENROUTER_API_KEY and OPENROUTER_MODEL locally')
    try:
        with httpx.Client(timeout=45, trust_env=False) as client:
            r = client.post('https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']},
                json={'model': os.environ['OPENROUTER_MODEL'], 'messages': redact(messages),
                      'temperature': 0, 'max_tokens': 1500})
            if r.status_code == 402:
                raise HTTPException(402, 'OpenRouter credits insufficient; add credits or configure an available free model')
            if r.status_code in (400,404):
                raise HTTPException(502, 'OpenRouter rejected model or request settings; verify OPENROUTER_MODEL and model availability')
            if r.status_code in (401,403):
                raise HTTPException(503, 'OpenRouter authentication failed; check server configuration')
            if r.status_code == 429:
                raise HTTPException(429, 'OpenRouter rate limit; retry later')
            r.raise_for_status()
            text = r.json()['choices'][0]['message']['content']
            if not isinstance(text, str) or not text.strip() or len(text) > 16000:
                raise ValueError('Empty response')
            return text
    except httpx.ConnectError as exc:
        raise HTTPException(503, 'Cannot connect to OpenRouter; check server network and DNS') from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(504, 'OpenRouter timed out') from exc
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(502, 'Invalid or unsuccessful OpenRouter response') from exc
