import logging
import requests

from pydantic import BaseModel
from typing import List

from adhteb.results import BenchmarkResult

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Metadata for a model."""
    name: str
    url: str


class LeaderboardEntry(BaseModel):
    """Leaderboard entry for a model."""
    model: ModelMetadata
    cohort_benchmarks: List[BenchmarkResult]


def publish_entry(entry: LeaderboardEntry):
    """
    Send the leaderboard entry to the leaderboard website API.
    """

    LEADERBOARD_API_URL = "http://api.hadteb.k8s.bio.scai.fraunhofer.de/leaderboard/"
    headers = {"Content-Type": "application/json"}

    response = requests.post(LEADERBOARD_API_URL, json=entry.model_dump_json(), headers=headers)

    if response.status_code == 201:
        logger.info("Leaderboard entry published successfully.")
    else:
        response.raise_for_status()
