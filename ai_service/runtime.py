import re
from ai_service.guardrails import validate_question, redact
from fastapi import HTTPException

def run_query(agent, lock, text):
    try:
        text = validate_question(text)
        if re.search(r'[\u0600-\u06ff]', text):
            raise ValueError('Local search currently parses English only. Write the mission question in English or select AI research for Arabic. / استخدم الإنجليزية في البحث المحلي أو اختار AI research للعربي.')
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not lock.acquire(blocking=False):
        raise HTTPException(429, 'Agent busy; retry shortly', headers={'Retry-After': '5'})
    try:
        result = agent.run(text)
    finally:
        lock.release()
    for rank, pick in enumerate(result['picks'], 1):
        pick['sourceRank'] = pick['rank']
        pick['rank'] = rank
        pick['scoreInterval'] = None
        pick['confidenceMethod'] = 'uncalibrated heuristic'
    result['evidenceStatus'] = 'exploratory; requires source and site verification'
    result['capabilitiesUrl'] = '/api/v1/agent/capabilities'
    result['schemaVersion'] = '1.0'
    result['serviceVersion'] = '1.0.0-rc.1'
    result['limitations'] = 'Percentiles are relative rankings, not physical similarity percentages. Agent confidence is a heuristic label, not calibrated uncertainty. Purpose presets are exploratory weights, not validation of ice resources, biosignatures, or mission suitability. Some model inputs are imputed. Regional bounds are approximate. English request parsing only. Review audit flags before use.'
    return redact(result)
