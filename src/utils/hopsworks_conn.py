from typing import Optional

import hopsworks
from hsfs.connection import Connection


class HopsworksConn:
    _instance = None
    _conn: Optional[Connection] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, api_key: str = None):
        if not self._conn and api_key is None:
            raise ValueError("API key must be provided to initialize connection")

        if not self._conn:
            self._conn = hopsworks.login(
                api_key_value=api_key,
                project="feature_store_project",
            )

    def conn(self):
        assert self._conn is not None, "Connection not initialized"
        return self._conn
