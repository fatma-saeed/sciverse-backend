from ai_service.guardrails import validate_question, redact, GUARDRAILS_VERSION
from ai_service.evidence_answer import finalize
from ai_service.openrouter import chat, configuration
from .capabilities import CAPABILITIES
from threading import Lock
from fastapi import HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator
from ai_service.runtime import run_query

class AgentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    @field_validator('text')
    @classmethod
    def meaningful(cls, v):
        return validate_question(v)
    text: str = Field(min_length=3, max_length=2000)

def attach_agent(app):
    app.state.agent_lock = Lock()

    @app.get('/api/v1/agent/capabilities', tags=['Live mission agent'])
    def capabilities():
        return CAPABILITIES

    @app.get('/api/v1/agent/health', tags=['Live mission agent'])
    def health():
        return {'guardrailsVersion': GUARDRAILS_VERSION, 'status': 'ready', 'mode': 'in-process', 'llmProvider': 'openrouter', 'llmConfigured': configuration(), 'llmConnectivity': 'not checked'}

    @app.post('/api/v1/agent/chat', tags=['OpenRouter agent'])
    def llm_query(request: AgentRequest):
        if not configuration():
            raise HTTPException(503, 'Configure OPENROUTER_API_KEY and OPENROUTER_MODEL locally')
        if not app.state.agent_lock.acquire(blocking=False):
            raise HTTPException(429, 'Agent busy; retry shortly')
        try:
            result = app.state.agent.run_with_llm(request.text, chat, max_steps=8)
            result['mode'] = 'openrouter'
            result['schemaVersion'] = '1.0'
            result = finalize(result, request.text)
            return redact(result)
        finally:
            app.state.agent_lock.release()

    @app.post('/api/v1/agent/query', tags=['Live mission agent'])
    @app.post('/api/ai/agent/query', tags=['Live mission agent'])
    def query(request: AgentRequest):
        return redact(run_query(app.state.agent, app.state.agent_lock, request.text))
