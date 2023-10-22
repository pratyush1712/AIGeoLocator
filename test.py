import unittest
from app import app, cache


class GraftAppTests(unittest.TestCase):
    test_query = "rivers"
    test_thresh = 0.05

    def setUp(self):
        app.config["TESTING"] = True
        self.app = app.test_client()

    def test_app_health(self):
        resp = self.app.get("/health")
        self.assertEqual(resp.json, {"status": "OK"})
        self.assertEqual(resp.status_code, 200)

    def test_classified_point(self):
        resp = self.app.post(
            "/classified-points",
            query_string={"query": self.test_query, "thresh": self.test_thresh},
        )
        cached_data = cache.get(f"{self.test_query}_{self.test_thresh}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json, {"blue_coords": cached_data[0], "top_locs": cached_data[1]}
        )


if __name__ == "__main__":
    unittest.main()
