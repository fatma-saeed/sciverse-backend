"""Read-only adapter for SciVerse's precomputed outputs; never imports/trains AI."""
import csv
import os
from pathlib import Path
from threading import Lock
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from .store import read_json

LIMITATIONS = ('SciVerse precomputed multi-theme results; not the terrain-only mission scores. '
              'Percentiles are not probabilities or confidence. Score intervals and confidence '
              'are not supplied. Grid points describe survey regions, not verified sites. '
              'These outputs do not establish bare-ground eligibility or mission suitability.')

class AIQuery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    targetId: str
    limit: int = Field(default=10, ge=1, le=100)
    includeGrid: bool = False
    search: str | None = None

class AIStore:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('TERRATWIN_AI_OUTPUTS_DIR', 'nasa_ai/SciVerse/outputs')).resolve()
        self.snapshot = None
        self.lock = Lock()

    def reload(self):
        with self.lock:
            catalog = read_json(self.directory / 'frontend_scores.json')
            targets = {t['id']: t for t in catalog['targets']}
            points = {p['id']: p for p in catalog['points']}
            with (self.directory / 'rankings.csv').open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            with (self.directory / 'ai_scores.csv').open(encoding='utf-8-sig', newline='') as f:
                body_scores = {r['earth_id']: r for r in csv.DictReader(f)}
            for row in rows:
                if row['target_id'] not in targets or row['earth_id'] not in points:
                    raise ValueError('AI ranking references an unknown target or candidate')
                for field in ('rank', 'overall_pct', 'thermal_pct', 'dryness_pct', 'pressure_pct', 'terrain_pct', 'surface_pct', 'lat', 'lon'):
                    row[field] = float(row[field]) if row.get(field) else None
            snapshot = dict(targets=targets, points=points, rows=rows, body_scores=body_scores,
                            themes=catalog['themes'], evaluation=read_json(self.directory / 'ai_evaluation.json', optional=True))
            self.snapshot = snapshot
        return snapshot


def attach_ai(app, directory=None):
    store = AIStore(directory)
    app.state.ai_store = store
    router = APIRouter(prefix='/api/ai', tags=['SciVerse AI outputs'])

    def data():
        if store.snapshot is None:
            raise HTTPException(503, 'AI outputs not loaded; configure TERRATWIN_AI_OUTPUTS_DIR and POST /api/ai/reload')
        return store.snapshot

    def evidence(row):
        return {**row, 'score': row.get('overall_pct'), 'scoreScale': 'percentile 0–100',
                'scoreInterval': None, 'confidence': None, 'uncertaintyStatus': 'missing',
                'limitations': LIMITATIONS}

    @router.get('/status')
    def status():
        return {'loaded': store.snapshot is not None, 'outputsDir': str(store.directory),
                'mode': 'precomputed', 'limitations': LIMITATIONS}

    @router.get('/targets')
    def targets():
        return list(data()['targets'].values())

    @router.post('/recommendations')
    def recommendations(request: AIQuery):
        d = data()
        if request.targetId not in d['targets']:
            raise HTTPException(400, f'Unknown AI targetId: {request.targetId}')
        rows = [r for r in d['rows'] if r['target_id'] == request.targetId
                and (request.includeGrid or r['earth_kind'] != 'grid')
                and d['body_scores'].get(r['earth_id'], {}).get('label') != 'control'
                and (not request.search or request.search.casefold() in (r['earth_id']+' '+r.get('earth_name', '')).casefold())]
        rows.sort(key=lambda r: (r['overall_pct'] is not None, r['overall_pct'] if r['overall_pct'] is not None else float('-inf')), reverse=True)
        return {'targetId': request.targetId, 'method': 'precomputed target theme ranking',
                'results': [evidence(r) for r in rows[:request.limit]], 'limitations': LIMITATIONS}

    @router.get('/candidates/{id}')
    def candidate(id: str, targetId: str = Query(...)):
        d = data()
        if id not in d['points']:
            raise HTTPException(404, f'Unknown AI candidate id: {id}')
        if targetId not in d['targets']:
            raise HTTPException(400, f'Unknown AI targetId: {targetId}')
        point = d['points'][id]
        values = point.get('scores', {}).get(targetId)
        return {'id': id, 'name': point.get('name'), 'kind': point.get('kind'),
                'targetId': targetId, 'score': values[0] if values else None,
                'scoreScale': 'percentile 0–100', 'scoreInterval': None, 'confidence': None,
                'uncertaintyStatus': 'missing', 'themes': dict(zip(d['themes'], values)) if values else None,
                'status': 'available' if values else 'missing', 'limitations': LIMITATIONS}

    @router.get('/evaluation')
    def evaluation():
        value = data()['evaluation']
        return {'evaluation': value, 'status': 'available' if value is not None else 'missing', 'limitations': LIMITATIONS}

    @router.post('/reload')
    def reload():
        try:
            d = store.reload()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(503, 'AI reload failed; previous snapshot retained. Check output files.') from exc
        return {'loaded': True, 'targets': len(d['targets']), 'candidates': len(d['points'])}

    app.include_router(router)
    return store
