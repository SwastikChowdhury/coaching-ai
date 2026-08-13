"""Smoke tests for the operational endpoints (liveness + metrics exposure)."""

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():
    """/health returns the exact 200 + {"status": "ok"} contract probes rely on."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_metrics_exposed():
    """Prometheus scrape endpoint is wired up — '# HELP' confirms real exposition output."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "# HELP" in resp.text


def test_metrics_include_expected_series():
    """Phase 1 series: in-progress gauge, high-res HTTP histogram, agent buckets past 10s."""
    # Labeled histograms are not emitted until observed; materialize the bucket lines.
    from app.observability.metrics import agent_latency
    agent_latency.labels(agent="conversation", outcome="ok").observe(0)

    body = client.get("/metrics").text
    assert "http_requests_inprogress" in body
    assert "http_request_duration_highr_seconds" in body
    assert "muse_agent_latency_seconds_bucket" in body
    assert 'le="30.0"' in body
    assert 'le="60.0"' in body