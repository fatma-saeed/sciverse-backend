"""File-backed singleton. No scientific computation occurs in this module."""
import csv
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from threading import Lock

LIMITATIONS = 'Terrain-only scores. Temperature, geology, water/ice, and dust are not included. Earth does not reproduce lunar or Martian gravity, atmosphere, or radiation.'
FEATURES = ('slope_mean_deg', 'slope_p90_deg', 'roughness_m', 'relief_m')
DIMENSIONS = ('terrain', 'geology', 'temperature', 'water-ice', 'dust-wind', 'operational-relevance')

def key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())

def read_json(path, optional=False):
    if optional and not path.exists():
        return None
    with path.open(encoding='utf-8-sig') as f:
        return json.load(f, parse_constant=lambda v: None)

def score_view(row):
    result = deepcopy(row)
    result['scoreInterval'] = result.pop('score_interval_p10_p90', result.get('scoreInterval'))
    for field in ('score', 'scoreInterval', 'confidence', 'rank'):
        result.setdefault(field, None)
    if result.get('eligible') is False:
        result['rank'] = None
    result['status'] = 'missing' if result['score'] is None else 'verified'
    result['limitations'] = LIMITATIONS
    result['missingFields'] = [f for f in ('score', 'scoreInterval', 'confidence') if result[f] is None]
    features = result.get('features') or {}
    for name in FEATURES:
        feature = features.setdefault(name, {})
        for field in ('earth', 'target', 'similarity', 'weight'):
            feature.setdefault(field, None)
        feature.setdefault('status', 'missing' if any(feature[f] is None for f in ('earth', 'target', 'similarity', 'weight')) else 'verified')
    result['features'] = features
    return result

class Snapshot:
    def __init__(self, outputs, provenance):
        self.rankings = read_json(outputs / 'rankings.json')
        self.catalog = read_json(outputs / 'frontend_catalog.json')
        self.groups = self.rankings['rankings']
        self.candidates = {c['id']: c for c in self.catalog['candidates']}
        if len(self.candidates) != len(self.catalog['candidates']):
            raise ValueError('Duplicate candidate ids')
        self.unscored = self.catalog['unscoredObjectives']
        self.references = {}
        self.by_group = {}
        self.rows = {}
        for group in self.groups:
            target = group['target']
            ids = {a['targetId'] for c in self.candidates.values() for a in c.get('analysis', []) if a.get('target') == target}
            if len(ids) > 1:
                raise ValueError(f'Conflicting target ids: {target}')
            target_id = next(iter(ids), None)
            if target_id is None:
                target_id = {'jezerocrater': 'mars-jezero-context', 'moonsouthpole': 'moon-south-pole-context', 'lunarsouthpole': 'moon-south-pole-context', 'southpole': 'moon-south-pole-context'}.get(key(target))
            if target_id is None:
                raise ValueError(f'No frontend targetId for {target}')
            self.references[target_id] = {f: group.get(f) for f in ('target', 'body', 'pixel_m', 'tile_km')}
            self.references[target_id]['targetId'] = target_id
            pair = (target_id, group['objective'])
            if pair in self.by_group:
                raise ValueError(f'Duplicate ranking group: {pair}')
            self.by_group[pair] = group
            for row in group['results']:
                if row.get('role') == 'target':
                    continue
                matches = [c['id'] for c in self.candidates.values() if key(row['site']) in {key(c['id']), key(c['id'].removeprefix('earth-')), key(c['title'])}]
                if len(matches) != 1:
                    raise ValueError(f'Cannot uniquely join ranking site {row["site"]!r} to catalog; use matching title or id')
                row_key = (*pair, matches[0])
                if row_key in self.rows:
                    raise ValueError(f'Duplicate site result: {row_key}')
                self.rows[row_key] = row
        self.evaluation = read_json(outputs / 'evaluation.json', optional=True)
        self.provenance = read_json(provenance, optional=True)
        path = outputs / 'comparable_features.csv'
        self.terrain = []
        if path.exists():
            with path.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('role') == 'target':
                        for name in FEATURES:
                            value = row.get(name)
                            row[name] = None if value in (None, '', 'NaN', 'nan') else float(value)
                        self.terrain.append(row)

    def candidate(self, candidate_id, include_excluded=False):
        candidate = deepcopy(self.candidates[candidate_id])
        candidate['analysis'] = [score_view(a) for a in candidate.get('analysis', []) if include_excluded or a.get('eligible') is not False]
        candidate['limitations'] = LIMITATIONS
        return candidate

    def dimensions(self, candidate_id, target_id):
        target = self.references[target_id]['target']
        source = self.candidates[candidate_id].get('dimensions', {})
        rows = deepcopy(source.get(target, source.get(target_id, [])))
        if not rows:
            return [{'id': name, 'label': name, 'status': 'missing', 'statement': None, 'caveat': 'No dimension evidence supplied.'} for name in DIMENSIONS]
        return rows

class DataStore:
    def __init__(self, outputs=None, provenance=None):
        self.outputs = Path(outputs or os.environ.get('TERRATWIN_OUTPUTS_DIR', 'outputs')).resolve()
        self.provenance = Path(provenance or os.environ.get('TERRATWIN_PROVENANCE_PATH', str(self.outputs.parent / 'provenance.json'))).resolve()
        self._lock = Lock()
        self.snapshot = None

    def reload(self):
        # Readers retain the previous complete snapshot if any file fails to load.
        with self._lock:
            replacement = Snapshot(self.outputs, self.provenance)
            self.snapshot = replacement
        return replacement
