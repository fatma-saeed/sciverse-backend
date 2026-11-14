"""Local service for the original SciVerse deterministic mission agent."""
from .runtime import run_query
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
lock = Lock()

def AgentLoader():
    from terratwin_agent import Agent
    return Agent()

@asynccontextmanager
async def lifespan(app):
    from terratwin_agent import Agent
    app.state.agent = Agent()
    yield

app = FastAPI(title='SciVerse mission agent', lifespan=lifespan, version='1.0.0-rc.1')
class Question(BaseModel):
    text: str = Field(min_length=3, max_length=2000)

@app.get('/health')
def health():
    return {'status': 'ready', 'mode': 'deterministic agent; no LLM'}

@app.post('/query')
def query(request: Question):
    return run_query(app.state.agent, lock, request.text)
