"""Explicit challenge scope for frontend consumers; no inferred scientific measurements."""
CAPABILITIES = {
    'schemaVersion': '1.0',
    'status': 'exploratory-prototype',
    'supports': ['Earth candidate discovery', 'Moon and Mars reference contexts',
                 'purpose-weighted ranking', 'feature comparisons', 'limitations and audit flags',
                 'coordinates for map visualization'],
    'dimensions': {
        'terrain': 'coarse slope and relief; not landing safety or trafficability',
        'temperature': 'mixed observational and model context; spatial scales differ',
        'dryness': 'precipitation proxy',
        'pressure': 'comparison with explicit physical gaps',
        'surface': 'albedo proxy, not mineralogical equivalence',
        'geology': 'Earth keyword flags in classifier; no paired target geology comparison',
        'waterIce': 'not validated for resource prospecting',
        'biology': 'not directly evaluated',
        'caves': 'no subsurface geometry or stability analysis',
        'dustWind': 'not directly included in five-theme similarity'
    },
    'sourceFamiliesDocumentedByDataset': ['NASA POWER', 'LRO Diviner', 'Lunar Prospector',
        'Mars Climate Database', 'MGS MOLA', 'Odyssey GRS', 'MSL REMS',
        'Open-Meteo / Copernicus DEM', 'OpenTopoData', 'Macrostrat'],
    'provenanceStatus': 'Dataset documentation inspected; upstream downloads and every individual measurement not independently verified.',
    'limitations': [
        'Candidates are hypotheses, not verified novel discoveries or approved landing/base sites.',
        'Percentiles refer to the evaluated candidate dataset, not all Earth locations.',
        'Purpose weights are expert-set starting points, not mission validation.',
        'Region filters use rectangular bounds rather than country boundary polygons.',
        'Some coordinates are unverified; coarse grid cells represent survey regions.',
        'Missing classifier inputs may be imputed to Earth means internally.',
        'Confidence labels are uncalibrated heuristics.',
        'Gravity and radiation comparisons include hard-coded literature values requiring source review.',
        'Map markers use dataset coordinates; imagery is visual context, not a measurement layer.'
    ]
}
