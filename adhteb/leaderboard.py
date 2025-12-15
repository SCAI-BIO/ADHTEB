import logging

import requests
from pydantic import BaseModel

from adhteb.results import BenchmarkResult

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Metadata for a model."""

    name: str
    url: str


class LeaderboardEntry(BaseModel):
    """Leaderboard entry for a model."""

    model: ModelMetadata
    aggregate_score: float
    cohort_benchmarks: list[BenchmarkResult]


def publish_entry(entry: LeaderboardEntry):
    """
    Send the leaderboard entry to the leaderboard website API.
    """

    LEADERBOARD_API_URL = "https://api.adhteb.scai.fraunhofer.de/leaderboard/"
    PUBLISH_TOKEN = "adhteb-leaderboard"
    headers = {"Content-Type": "application/json", "X-Adhteb-Publish-Token": PUBLISH_TOKEN}

    response = requests.post(LEADERBOARD_API_URL, json=entry.model_dump(), headers=headers)

    if response.status_code == 200:
        logger.info("Leaderboard entry published successfully.")
    else:
        response.raise_for_status()
