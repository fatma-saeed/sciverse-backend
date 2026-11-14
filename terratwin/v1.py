"""Supported frontend contract, backed by the existing model."""
from collections import Counter
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .capabilities import CAPABILITIES

class Discovery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    targetId: str
    objective: str = 'general'
    region: str | None = None
    includeGrid: bool = False
    excludePolar: bool = False
    limit: int = Field(default=10, ge=1, le=100)

class Comparison(Discovery):
    candidateIds: list[str] = Field(min_length=1, max_length=20)

class Result(BaseModel):
    schemaVersion: str = '1.0'
    data: Any
    limitations: list[str] = CAPABILITIES['limitations']

def attach_v1(app):
    router = APIRouter(prefix='/api/v1', tags=['Supported frontend API'])
    def agent():
        return app.state.agent
    def validate(req):
        import terratwin_ai as ai
        from terratwin_agent import REGIONS
        if req.targetId not in agent().d.tid:
            raise HTTPException(400, 'Unknown targetId')
        if req.objective not in ai.PURPOSES:
            raise HTTPException(400, 'Unsupported objective')
        if req.region is not None and req.region not in REGIONS:
            raise HTTPException(400, 'Unsupported region; consult objectives metadata')
    def detail(id, target, objective):
        a = agent()
        if id not in a.d.eid or a.d.earth[a.d.eid[id]].get('label') == 'control':
            raise HTTPException(404, 'Unknown or non-recommendable candidate')
        e = a.d.earth[a.d.eid[id]]
        return {'id': id, 'kind': e['kind'], 'classification': e.get('label') or 'survey-region',
                'coordinate': {'latitude': float(e['lat']), 'longitude': float(e['lon']), 'convention': 'degrees north / east; negative south / west'},
                'facts': a.site_facts(id, target, objective), 'audit': a.audit_site(id),
                'rawFeatures': {k: (v if v != '' else None) for k, v in e.items()},
                'scoreInterval': None, 'confidence': None, 'uncertaintyStatus': 'missing'}
    @router.get('/catalog', response_model=Result)
    def catalog():
        a = agent()
        def coordinate(row, longitude):
            return {'latitude': float(row['lat']), 'longitude': float(row[longitude]),
                    'convention': 'degrees north / east; negative south / west'}
        return Result(data={
            'features': [{'id': t['target_id'], 'body': t['body'], 'title': t['name'],
                'kind': 'Dataset reference', 'coordinate': coordinate(t, 'lon_east'),
                'description': t.get('site_status') or 'Reference from the backend dataset',
                'context': 'Dataset reference; source coverage and coordinate verification vary.',
                'sourceLabel': ' / '.join(filter(None, [t.get('climate_source'), t.get('terrain_source')])) or 'Dataset source metadata unavailable',
                'status': 'hypothesis'} for t in a.d.targets],
            'candidates': [{'id': e['id'], 'body': 'earth', 'title': e.get('name') or e['id'],
                'region': e.get('country') or 'Region not supplied', 'coordinate': coordinate(e, 'lon'),
                'description': e.get('notes') or 'Candidate from the backend dataset; evaluate against a selected planetary reference.',
                'notice': 'Exploratory candidate, not a validated planetary equivalent.',
                'tags': [], 'sourceLabel': 'Backend dataset; per-measurement provenance may be incomplete', 'status': 'hypothesis',
                'discovery': {'environment': 'Unknown', 'targetPlanets': [], 'missionObjectives': [],
                    'researchDimensions': [], 'suggestedApplications': [],
                    'limitations': 'Use live discovery and comparison for target-specific evidence.',
                    'coordinateTreatment': 'dataset-coordinate'}}
                for e in a.d.earth if e['kind'] != 'grid' and e.get('label') != 'control']})

    @router.get('/targets', response_model=Result)
    def targets():
        return Result(data=agent().list_targets())
    @router.get('/objectives', response_model=Result)
    def objectives():
        import terratwin_ai as ai
        from terratwin_agent import REGIONS
        return Result(data={'objectives': ai.PURPOSES, 'regions': REGIONS,
            'method': 'expert-set exploratory theme weights; ice and biology are not directly measured'})
    @router.post('/discovery/search', response_model=Result)
    def search(req: Discovery):
        validate(req)
        a = agent()
        rows = a.search_sites(req.targetId, req.objective, len(a.d.earth), req.region, req.excludePolar)
        rows = [r for r in rows if req.includeGrid or a.d.earth[a.d.eid[r['earth_id']]]['kind'] != 'grid'][:req.limit]
        return Result(data=[{**r, 'sourceRank': r['rank'], 'rank': i, 'scoreScale': 'dataset percentile',
            'scoreInterval': None, 'confidence': None, 'details': detail(r['earth_id'], req.targetId, req.objective)} for i, r in enumerate(rows, 1)])
    @router.get('/candidates/{id}', response_model=Result)
    def candidate(id: str, targetId: str, objective: str = 'general'):
        validate(Discovery(targetId=targetId, objective=objective))
        return Result(data=detail(id, targetId, objective))
    @router.post('/comparisons', response_model=Result)
    def comparisons(req: Comparison):
        validate(req)
        if req.region or req.excludePolar or req.includeGrid or req.limit != 10:
            raise HTTPException(400, 'Comparison accepts targetId, objective and candidateIds only')
        return Result(data=[detail(id, req.targetId, req.objective) for id in dict.fromkeys(req.candidateIds)])
    @router.get('/readiness', response_model=Result)
    def readiness():
        return Result(data={'agent': 'ready', 'savedOutputs': app.state.ai_store.snapshot is not None,
            'legacyTerrain': 'excluded from supported frontend contract', 'frontendIntegrated': True})
    @router.get('/provenance', response_model=Result)
    def provenance():
        a = agent()
        return Result(data={'counts': {'earth': len(a.d.earth), 'targets': len(a.d.targets),
            'kinds': dict(Counter(e['kind'] for e in a.d.earth)), 'labels': dict(Counter(e.get('label') or 'unlabeled' for e in a.d.earth))},
            'sourceFamilies': CAPABILITIES['sourceFamiliesDocumentedByDataset'],
            'verification': CAPABILITIES['provenanceStatus'],
            'acquisitionDates': None, 'perMeasurementSourceURLs': None,
            'processing': {'climate': 'Earth POWER area climatology; Mars MCD model; Moon Diviner product',
            'terrain': 'coarse sampled DEM relief/slope; not crater-scale measurement',
            'classifierMissing': 'Earth-mean environmental imputation; geology missing flag',
            'surface': 'albedo proxy', 'geology': 'Earth Macrostrat keywords, classifier only'}})
    app.include_router(router)
