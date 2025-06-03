from dataclasses import dataclass

# from loguru import logger
import requests
from pydantic import BaseModel


@dataclass
class MoviesAPIConfig:
    token: str
    url: str = (
        "https://api.themoviedb.org/3/discover/movie"
        "?sort_by=release_date.desc&vote_count.gte=10&page={}"
    )
    pages: int = 400  # number of pages to read
    timeout: int = 10  # seconds
    HTTP_OKAY: int = 200

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "accept": "application/json",
        }


class MoviesAPIClient:
    def _init_(self, config: MoviesAPIConfig):
        self.config = config

    def fetch_movies(self) -> list:
        movies: list[MovieData] = list()
        for page in range(1, self.config.pages + 1):
            response = requests.get(
                self.config.url.format(page),
                headers=self.config.headers,
                timeout=self.config.timeout,
            )
            if response.status_code != self.config.HTTP_OKAY:
                continue
                # error for current page, skip it
            data = response.json().get("results", [])
            if not data:
                continue
            movies.extend(map(lambda x: MovieData(**x), data))
        return movies


class MovieData(BaseModel):
    id: int
    adult: bool
    original_language: str
    original_title: str
    overview: str

    class Config:
        extra = "ignore"
