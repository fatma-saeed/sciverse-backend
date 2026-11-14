import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .v1 import attach_v1
from .agent_api import attach_agent
from .ai_api import attach_ai
from .models import SearchRequest, ComparisonRequest
from .store import DataStore, LIMITATIONS, score_view


def create_app(outputs=None, provenance=None, ai_outputs=None):
    store = DataStore(outputs, provenance)

    @asynccontextmanager
    async def lifespan(app):
        try:
            store.reload()
        except FileNotFoundError:
            # Keep documentation and reload available before pipeline outputs arrive.
            pass
        try:
            ai_store.reload()
        except (OSError, ValueError, KeyError, TypeError):
            pass
        from ai_service.main import AgentLoader
        app.state.agent = AgentLoader()
        yield

    app = FastAPI(title='TerraTwin data API', version='1.0.0-rc.1', lifespan=lifespan,
                  description='SciVerse backend with an in-process mission agent and saved AI results.')
    attach_agent(app)
    attach_v1(app)
    app.state.store = store
    ai_store = attach_ai(app, ai_outputs)
    app.add_middleware(CORSMiddleware, allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173').split(','),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

    @app.middleware('http')
    async def require_data(request, call_next):
        if (request.url.path.startswith('/api/')
                and not request.url.path.startswith('/api/ai/')
                and not request.url.path.startswith('/api/v1/')
                and request.url.path not in ('/api/admin/reload', '/api/health')
                and request.method != 'OPTIONS'
                and store.snapshot is None):
            return JSONResponse(status_code=503, content={
                'status': 'missing',
                'detail': 'Pipeline outputs are not loaded. Provide rankings.json and frontend_catalog.json, then POST /api/admin/reload.',
                'outputsDir': str(store.outputs),
            })
        return await call_next(request)

    @app.get('/api/health')
    def health():
        return {'status': 'ready' if store.snapshot is not None else 'missing',
                'dataLoaded': store.snapshot is not None,
                'outputsDir': str(store.outputs)}

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse(status_code=400, content={'detail': 'Invalid request', 'errors': [{'field': list(e['loc']), 'message': e['msg']} for e in exc.errors()]})

    def context(data, request):
        if request.targetId not in data.references:
            raise HTTPException(400, f'Unknown targetId: {request.targetId}')
        if request.objective in data.unscored:
            return {'scored': False, 'needs': data.unscored[request.objective], 'limitations': LIMITATIONS}
        if (request.targetId, request.objective) not in data.by_group:
            raise HTTPException(400, f'Objective unavailable for this target: {request.objective}')
        return None

    @app.get('/api/references')
    def references():
        return list(store.snapshot.references.values())

    @app.get('/api/references/{id}')
    def reference(id: str):
        data = store.snapshot
        if id not in data.references:
            raise HTTPException(404, f'Unknown reference id: {id}')
        ref = data.references[id]
        rows = [r for r in data.terrain if r.get('site', r.get('target')) == ref['target']]
        return {**ref, 'terrainFeatures': rows or None, 'terrainFeaturesStatus': 'verified' if rows else 'missing', 'limitations': LIMITATIONS}

    @app.get('/api/objectives')
    def objectives():
        data = store.snapshot
        scored = []
        for (target_id, objective), group in data.by_group.items():
            weights = group.get('weights')
            if weights is None:
                weights = {name: next((r['features'][name]['weight'] for r in group['results'] if r.get('features', {}).get(name, {}).get('weight') is not None), None) for name in ('slope_mean_deg', 'slope_p90_deg', 'roughness_m', 'relief_m')}
            scored.append({'targetId': target_id, 'objective': objective, 'why': group.get('why'), 'weights': weights, 'status': 'missing' if any(v is None for v in weights.values()) else 'verified'})
        return {'scoredObjectives': scored, 'unscoredObjectives': data.unscored, 'limitations': LIMITATIONS}

    @app.get('/api/earth-sites')
    def sites(includeExcluded: bool = False):
        data = store.snapshot
        result = []
        for id in data.candidates:
            candidate = data.candidate(id, includeExcluded)
            analyses = data.candidates[id].get('analysis', [])
            if analyses and all(a.get('eligible') is False for a in analyses) and not includeExcluded:
                continue
            result.append({k: v for k, v in candidate.items() if k not in ('analysis', 'dimensions')})
        return result

    @app.get('/api/earth-sites/{id}')
    def site(id: str, includeExcluded: bool = False):
        data = store.snapshot
        if id not in data.candidates:
            raise HTTPException(404, f'Unknown earth-site id: {id}')
        analyses = data.candidates[id].get('analysis', [])
        if analyses and all(a.get('eligible') is False for a in analyses) and not includeExcluded:
            raise HTTPException(404, 'Excluded site; use includeExcluded=true')
        return data.candidate(id, includeExcluded)

    @app.post('/api/discovery/search')
    def search(request: SearchRequest):
        data = store.snapshot
        unscored = context(data, request)
        if unscored:
            return unscored
        results = []
        for (target, objective, id), row in data.rows.items():
            if (target, objective) != (request.targetId, request.objective):
                continue
            if row.get('eligible') is False and not request.includeExcluded:
                continue
            candidate = data.candidates[id]
            env = candidate.get('discovery', {}).get('environment', [])
            env = [env] if isinstance(env, str) else env or []
            if request.environments and not {e.casefold() for e in env}.intersection(e.casefold() for e in request.environments):
                continue
            haystack = ' '.join(str(candidate.get(k, '')) for k in ('title', 'region', 'description', 'tags')).casefold()
            if request.search and request.search.casefold() not in haystack:
                continue
            if request.minScore is not None and (row.get('score') is None or row['score'] < request.minScore):
                continue
            results.append({'id': id, 'title': candidate.get('title'), **score_view(row)})
        results.sort(key=lambda r: (r['score'] is not None, r['score'] if r['score'] is not None else float('-inf')), reverse=True)
        return {'scored': True, 'targetId': request.targetId, 'objective': request.objective, 'results': results, 'limitations': LIMITATIONS}

    @app.post('/api/comparisons')
    def compare(request: ComparisonRequest):
        data = store.snapshot
        unscored = context(data, request)
        for id in request.candidateIds:
            if id not in data.candidates:
                raise HTTPException(404, f'Unknown earth-site id: {id}')
        if unscored:
            return unscored
        results = []
        for id in dict.fromkeys(request.candidateIds):
            row = data.rows.get((request.targetId, request.objective, id), {})
            if row.get('eligible') is False and not request.includeExcluded:
                continue
            result = score_view(row)
            for name in ('similar', 'partial', 'different', 'cannot_reproduce', 'note', 'eligible'):
                result.setdefault(name, None)
            results.append({'id': id, **result, 'dimensions': data.dimensions(id, request.targetId)})
        return {'scored': True, 'results': results, 'limitations': LIMITATIONS}

    @app.get('/api/datasets')
    def datasets():
        data = store.snapshot
        return {'sources': data.provenance, 'status': 'missing' if data.provenance is None else 'verified', 'method': data.rankings.get('method'), 'limitations': LIMITATIONS}

    @app.get('/api/evaluation')
    def evaluation():
        value = store.snapshot.evaluation
        return value if value is not None else {'evaluation': None, 'status': 'missing'}

    @app.post('/api/admin/reload')
    def reload():
        try:
            data = store.reload()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(503, 'Reload failed; previous snapshot retained. Check pipeline files and schema.') from exc
        return {'reloaded': True, 'references': len(data.references), 'candidates': len(data.candidates)}

    return app

app = create_app()
