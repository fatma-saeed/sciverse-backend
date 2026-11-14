import unittest
from fastapi.testclient import TestClient
from terratwin.main import app

class FrontendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()
    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_catalog_matches_source_rows(self):
        response = self.client.get('/api/v1/catalog')
        self.assertEqual(response.status_code, 200)
        data = response.json()['data']
        self.assertEqual(len(data['features']), len(app.state.agent.d.targets))
        for candidate in data['candidates']:
            raw = app.state.agent.d.earth[app.state.agent.d.eid[candidate['id']]]
            self.assertNotEqual(raw.get('label'), 'control')
            self.assertNotEqual(raw['kind'], 'grid')
            self.assertEqual(candidate['coordinate']['latitude'], float(raw['lat']))
            self.assertNotEqual(candidate['status'], 'demonstration')

    def test_discovery_profile_comparison_use_same_records(self):
        query = {'targetId': 'mars_jezero', 'objective': 'rover_mobility', 'region': 'egypt', 'limit': 3}
        response = self.client.post('/api/v1/discovery/search', json=query)
        self.assertEqual(response.status_code, 200)
        rows = response.json()['data']
        self.assertGreater(len(rows), 0)
        catalog_ids = {r['id'] for r in self.client.get('/api/v1/catalog').json()['data']['candidates']}
        for row in rows:
            self.assertIn(row['earth_id'], catalog_ids)
            profile = self.client.get('/api/v1/candidates/' + row['earth_id'], params={'targetId': query['targetId'], 'objective': query['objective']})
            self.assertEqual(profile.json()['data'], row['details'])
        comparison = self.client.post('/api/v1/comparisons', json={
            'targetId': query['targetId'], 'objective': query['objective'], 'candidateIds': [r['earth_id'] for r in rows]})
        self.assertEqual(comparison.status_code, 200)
        self.assertEqual(comparison.json()['data'], [r['details'] for r in rows])

    def test_no_fake_fallback_for_empty_or_invalid_queries(self):
        response = self.client.post('/api/v1/discovery/search', json={
            'targetId': 'mars_jezero', 'region': 'antarctica', 'excludePolar': True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['data'], [])
        self.assertEqual(self.client.get('/api/v1/candidates/nonexistent', params={'targetId': 'mars_jezero'}).status_code, 404)
        self.assertEqual(self.client.post('/api/v1/discovery/search', json={'targetId': 'nonexistent'}).status_code, 400)

if __name__ == '__main__':
    unittest.main()
