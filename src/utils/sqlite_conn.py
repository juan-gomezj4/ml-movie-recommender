from json import dumps
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Any, ClassVar, Optional

from loguru import logger
from pandas import DataFrame, Index

from src.utils.feature_store_interface import FeatureStoreInterface


class SQLiteConn(FeatureStoreInterface):
    ERR_NO_DB_PATH: ClassVar[str] = "Database path must be provided to initialize connection"
    ERR_DB_NOT_EXISTS: ClassVar[str] = "Database file {} does not exist"
    ERR_CONN_NOT_INITIALIZED: ClassVar[str] = "Connection not initialized"
    ERR_MISSING_FEATURE_GROUP: ClassVar[str] = "Feature group name must be provided"
    ERR_EMPTY_FEATURES: ClassVar[str] = "Feature group name and features must be provided"

    _instance: Optional["SQLiteConn"] = None
    _conn: Connection | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "SQLiteConn":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str | None = None):
        if not self._conn and not db_path:
            raise ValueError(self.ERR_NO_DB_PATH)

        if db_path:
            assert Path(db_path).exists(), self.ERR_DB_NOT_EXISTS.format(db_path)

        if not self._conn and db_path:
            self._conn = connect(db_path)

    def __prepare_for_storage(self, features: DataFrame) -> DataFrame:
        df = features.copy()
        df = df.convert_dtypes()
        list_cols: Index[str] = df.select_dtypes(include=["object"]).columns
        for col in list_cols:
            df[col] = df[col].apply(lambda x: dumps(x) if isinstance(x, list) else x)
        return df

    def insert(self, feature_group: str, features: DataFrame) -> None:
        if not feature_group or features.empty:
            raise ValueError(self.ERR_EMPTY_FEATURES)

        logger.info(f"Storing {len(features)} records in {feature_group} feature group")
        features_to_store: DataFrame = self.__prepare_for_storage(features)

        features_to_store.to_sql(
            name=feature_group,
            con=self._conn,
            if_exists="append",
            index=False,
        )

    def fetch_existing_movie_ids(self, feature_group: str) -> set[int]:
        if not self._conn:
            raise ValueError(self.ERR_CONN_NOT_INITIALIZED)

        if not feature_group:
            raise ValueError(self.ERR_MISSING_FEATURE_GROUP)

        logger.info(f"Fetching existing movie IDs from {feature_group}")
        query: str = "SELECT id FROM ?"
        idx: DataFrame = DataFrame(
            self._conn.execute(query, (feature_group,)).fetchall(), columns=["id"]
        )
        return set(idx["id"].tolist())
