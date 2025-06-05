from json import dumps, loads
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

    DESEARIALIZE_COLS: ClassVar[list[str]] = [
        "genres",
        "spoken_languages",
    ]  # this is tech debt

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

    def __deserialize_list_columns(self, features: DataFrame) -> DataFrame:
        """Deserialize JSON strings back to lists with proper UTF-8 encoding."""
        df: DataFrame = features.copy()
        for col in self.DESEARIALIZE_COLS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: [
                        bytes(item, "utf-8").decode("utf-8") if isinstance(item, str) else item
                        for item in loads(x)
                    ]
                    if isinstance(x, str)
                    else x
                )
        return df

    def insert(
        self,
        feature_group: str,
        features: DataFrame,
        mode: Any = "append",  # TODO: Change
    ) -> None:
        if not feature_group or features.empty:
            raise ValueError(self.ERR_EMPTY_FEATURES)

        logger.info(f"Storing {len(features)} records in {feature_group} feature group")
        features_to_store: DataFrame = self.__prepare_for_storage(features)

        features_to_store.to_sql(
            name=feature_group,
            con=self._conn,
            if_exists=mode,
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

    def query_features(self, feature_group: str, columns: list[str] | None = None) -> DataFrame:
        if not self._conn:
            raise ValueError(self.ERR_CONN_NOT_INITIALIZED)

        if not feature_group:
            raise ValueError(self.ERR_MISSING_FEATURE_GROUP)

        logger.info(f"Querying features from {feature_group} with columns {columns}")
        query: str = f"SELECT {', '.join(columns) if columns else '*'} FROM {feature_group}"  # noqa: S608
        features: DataFrame = DataFrame(self._conn.execute(query).fetchall(), columns=columns)
        deserialized_features: DataFrame = self.__deserialize_list_columns(features)
        return deserialized_features
