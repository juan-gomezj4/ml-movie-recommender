from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import requests
from loguru import logger
from pandas import DataFrame
from pydantic import BaseModel

from src.utils.feature_store_interface import FeatureStoreInterface


@dataclass
class MoviesAPIConfig:
    token: str
    feature_group: str
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


class MoviesAPIData(BaseModel):
    id: int
    adult: bool
    original_language: str
    original_title: str
    overview: str
    popularity: float
    vote_average: float
    vote_count: int
    release_date: datetime | None = None

    class Config:
        extra = "ignore"
        json_encoders: ClassVar = {datetime: lambda v: v.strftime("%Y-%m-%d")}  # format date Y-m-d


class MoviesAPIClient:
    ERR_NO_MOVIES: ClassVar[str] = "No movies were fetched from the API"
    ERR_NOT_INITIALIZED: ClassVar[str] = "Movies DataFrame not initialized"
    ERR_FETCH_DETAILS: ClassVar[str] = "Error fetching details for movie {}: {}"
    ERR_FETCH_POPULAR: ClassVar[str] = "Error fetching popular movies page {}: {}"

    def __init__(self, config: MoviesAPIConfig, existing_ids: set | None = None):
        self.config = config
        self.existing_ids: set | None = existing_ids
        self.movies: DataFrame | None = None
        self.__build_movie_base()

    def __build_movie_base(self) -> "MoviesAPIClient":
        logger.info("Starting base movie fetch...")

        movies: list[MoviesAPIData] = list()
        skipped_count, total_processed = 0, 0
        for page in range(1, self.config.pages + 1):
            response = requests.get(
                self.config.url.format(page),
                headers=self.config.headers,
                timeout=self.config.timeout,
            )
            if response.status_code != self.config.HTTP_OKAY:
                continue
                # error for current page, skip it
            data: list[dict] = response.json().get("results", [])
            if not data:
                continue
            total_processed += len(data)
            for movie in data:
                if self.existing_ids and movie["id"] in self.existing_ids:
                    skipped_count += 1
                    continue
                movies.append(MoviesAPIData(**movie))

        movies_dict: list[dict] = [movie.model_dump() for movie in movies]
        self.movies = DataFrame(movies_dict).drop_duplicates(subset="id")

        logger.info(
            f"\nMovie fetch summary:\n"
            f"- Total movies processed: {total_processed}\n"
            f"- Skipped (existing): {skipped_count}\n"
            f"- New movies added: {len(self.movies)}"
        )

        if self.movies.empty:
            raise ValueError(self.ERR_NO_MOVIES)

        return self

    def fetch_popular_movie_ids(self) -> "MoviesAPIClient":
        """Fetch IDs of currently popular movies and mark them in DataFrame.

        Args:
            pages: Number of pages to fetch from popular movies list

        Returns:
            MoviesAPIClient: Self reference for method chaining
        """
        logger.info("Starting popular movies fetch...")

        URL: str = "https://api.themoviedb.org/3/movie/popular?language=es-ES&page={}"
        popular_ids: set = set()

        for page in range(1, self.config.pages + 1):
            try:
                response = requests.get(
                    URL.format(page),
                    headers=self.config.headers,
                    timeout=self.config.timeout,
                )

                if response.status_code == self.config.HTTP_OKAY:
                    data = response.json().get("results", [])
                    popular_ids.update(movie["id"] for movie in data)
            except Exception as e:
                logger.error(f"Error fetching popular movies page {page}: {e!s}")
                continue

        # Mark popular movies in DataFrame
        if self.movies is not None:
            self.movies["is_popular"] = self.movies["id"].isin(popular_ids)

        return self

    def fetch_movie_extended_info(self) -> "MoviesAPIClient":
        logger.info("Starting extended movie info fetch...")

        assert self.movies is not None, self.ERR_NOT_INITIALIZED
        extended_info: list[dict] = list()
        for movie_id in self.movies["id"]:
            movie_info: dict = self.__fetch_detailed_movie_metadata(movie_id)
            extended_info.append(movie_info)

        extended_info_df: DataFrame = DataFrame(extended_info)
        self.movies = self.movies.merge(extended_info_df, on="id", how="left")
        return self

    def __fetch_detailed_movie_metadata(self, movie_id: int) -> dict:
        """Fetch detailed metadata for a specific movie from TMDb API.

        Args:
            movie_id: TMDb movie identifier

        Returns:
            dict: Extended movie metadata including runtime, budget, etc.
        """
        URL: str = f"https://api.themoviedb.org/3/movie/{movie_id}?language=es-ES"
        try:
            response = requests.get(URL, headers=self.config.headers, timeout=self.config.timeout)

            if response.status_code == self.config.HTTP_OKAY:
                data: dict = response.json()
                return {
                    "id": movie_id,
                    "runtime": data.get("runtime"),
                    "budget": data.get("budget"),
                    "revenue": data.get("revenue"),
                    "status": data.get("status"),
                    "tagline": data.get("tagline"),
                    "genres": [g["name"] for g in data.get("genres", [])],
                    "spoken_languages": [lang["name"] for lang in data.get("spoken_languages", [])],
                }

        except Exception as e:
            logger.error(self.ERR_FETCH_DETAILS.format(movie_id, str(e)))

        # Return default values if request fails
        return {
            "id": movie_id,
            "runtime": None,
            "budget": None,
            "revenue": None,
            "status": None,
            "tagline": None,
            "genres": [],
            "spoken_languages": [],
        }

    def add_extraction_date(self) -> "MoviesAPIClient":
        if self.movies is None:
            raise ValueError(self.ERR_NOT_INITIALIZED)

        extraction_date: str = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Adding extraction date: {extraction_date}")

        self.movies["extraction_date"] = extraction_date
        return self

    def store_movies_features(self, feature_store: FeatureStoreInterface) -> "MoviesAPIClient":
        if self.movies is None:
            raise ValueError(self.ERR_NOT_INITIALIZED)

        if "extraction_date" not in self.movies.columns:
            self.add_extraction_date()
        feature_store.insert(self.config.feature_group, self.movies)
        return self
