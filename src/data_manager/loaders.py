from __future__ import annotations
import os
import pandas as pd
from typing import Any, Protocol, Callable
from sqlalchemy.engine import Engine
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import fsspec
import mlflow
import json
import yaml
from fsspec.implementations.local import LocalFileSystem
from mlflow.entities import Experiment, Run
import shutil
from s3fs.core import S3FileSystem
import pickle


class Loader(Protocol):
    _configure: Callable[..., None]
    load: Callable[..., Any]


class DBMSLoader(Loader, Protocol):
    _connectable: Engine

    def load(self, sql, **kwargs) -> pd.DataFrame:
        with self._connectable.connect() as con:
            df = pd.read_sql(sql, con=con, **kwargs)

        return df


class SnowflakeLoader(DBMSLoader):
    _account: str | None = os.environ.get("SNOWFLAKE_ACCOUNT")
    _user: str | None = os.environ.get("SNOWFLAKE_USER")
    _password: str | None = os.environ.get("SNOWFLAKE_PASSWORD")
    _database: str | None = os.environ.get("SNOWFLAKE_DATABASE")
    _schema: str | None = os.environ.get("SNOWFLAKE_SCHEMA")
    _warehouse: str | None = os.environ.get("SNOWFLAKE_WAREHOUSE")
    _role: str | None = os.environ.get("SNOWFLAKE_ROLE")

    def __init__(
        self,
        account: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        schema: str | None = None,
        warehouse: str | None = None,
        role: str | None = None,
    ):

        self._configure(
            account=account or self._account,
            user=user or self._user,
            password=password or self._password,
            database=database or self._database,
            schema=schema or self._schema,
            warehouse=warehouse or self._warehouse,
            role=role or self._role,
        )

    def _configure(
        self,
        account: str | None,
        user: str | None,
        password: str | None,
        database: str | None = None,
        schema: str | None = None,
        warehouse: str | None = None,
        role: str | None = None,
    ):
        if not user or not account or not password:
            raise ValueError(
                "Missing account, user or password for snowflake."
                "You can provide them as arguments or set the SNOWFLAKE_ACCOUNT, SNOWFLAKE_PASSWORD and SNOWFLAKE_USER environment variables."
            )

        self._connectable = create_engine(
            URL(
                account=account,
                user=user,
                password=password,
                database=database,
                schema=schema,
                warehouse=warehouse,
                role=role,
            )
        )

    def load(self, sql, **kwargs) -> pd.DataFrame:
        return super().load(sql, **kwargs)


class FileLoader(Loader, Protocol):
    _filesystem: LocalFileSystem | S3FileSystem

    @staticmethod
    def _compute_filepath(file_key: str, base_directory: str | None = None) -> str: ...

    def load(self, file_key: str, base_directory: str | None = None, **kwargs):
        filepath = self._compute_filepath(file_key, base_directory)

        if not self._filesystem.exists(filepath):
            raise FileNotFoundError(f"The computed filepath {filepath} doesn't exist")

        if filepath.endswith(".parquet"):
            with self._filesystem.open(filepath, "rb") as f:
                readed_df_or_file = pd.read_parquet(f, **kwargs)

        elif filepath.endswith(".csv"):
            with self._filesystem.open(filepath, "rb") as f:
                readed_df_or_file = pd.read_csv(f, **kwargs)

        elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            with self._filesystem.open(filepath, "rb") as f:
                readed_df_or_file = pd.read_excel(f, **kwargs)

        elif filepath.endswith(".json"):
            with self._filesystem.open(filepath, "r", encoding="utf-8") as f:
                readed_df_or_file = json.load(f)

        elif filepath.endswith(".yaml") or filepath.endswith("yml"):
            with self._filesystem.open(filepath, "r", encoding="utf-8") as f:
                readed_df_or_file = yaml.safe_load(f)

        elif filepath.endswith(".pickle") or filepath.endswith(".pkl"):
            with self._filesystem.open(filepath, "rb") as f:
                readed_df_or_file = pickle.load(f, **kwargs)

        elif filepath.endswith(".html") or filepath.endswith(".log"):
            with self._filesystem.open(filepath, "r", encoding="utf-8") as f:
                readed_df_or_file = f.read()

        return readed_df_or_file


class LocalFileLoader(FileLoader):
    _filesystem: LocalFileSystem

    def __init__(self, **filesystem_args):
        self._configure(**filesystem_args)

    def _configure(self, **filesystem_args):
        self._filesystem = fsspec.filesystem("file", **filesystem_args)

    @staticmethod
    def _compute_filepath(file_key: str, base_directory: str | None = None) -> str:
        return os.path.join(base_directory or "", file_key)

    def load(self, file_key: str, base_directory: str | None = None, **kwargs) -> Any:
        return super().load(file_key=file_key, base_directory=base_directory, **kwargs)


class S3FileLoader(FileLoader):
    _filesystem: S3FileSystem
    _aws_access_key_id: str | None = os.environ.get("AWS_ACCESS_KEY_ID")
    _aws_secret_access_key: str | None = os.environ.get("AWS_SECRET_ACCESS_KEY")

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        **filesystem_args,
    ):

        self._configure(
            aws_access_key_id=aws_access_key_id or self._aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key or self._aws_secret_access_key,
            **filesystem_args,
        )

    def _configure(
        self,
        aws_access_key_id: str | None,
        aws_secret_access_key: str | None,
        **filesystem_args,
    ):
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "Missing aws_secret_access_key or aws_access_key_id for s3."
                "You can provide them as arguments or set the AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY environment variables."
            )
        self._filesystem = fsspec.filesystem(
            "s3", key=aws_access_key_id, secret=aws_secret_access_key, **filesystem_args
        )

    @staticmethod
    def _compute_filepath(file_key: str, base_directory: str | None = None) -> str:
        return os.path.join("s3://", base_directory or "", file_key)

    def load(self, file_key: str, base_directory: str | None = None, **kwargs) -> Any:
        return super().load(file_key=file_key, base_directory=base_directory, **kwargs)


class MlflowFileLoader(LocalFileLoader):
    _download_dir: str = "tmp/mlflow"
    _tracking_uri: str | None = os.environ.get("MLFLOW_TRACKING_URI")
    _experiment: Experiment
    _run: Run

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        download_dir: str | None = None,
        tracking_uri: str | None = None,
        **filesystem_args,
    ):

        super().__init__(**filesystem_args)

        self._configure_mlflow(
            download_dir=download_dir or self._download_dir,
            tracking_uri=tracking_uri or self._tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
        )

    def _configure_mlflow(
        self,
        experiment_name: str,
        run_name: str,
        tracking_uri: str | None,
        download_dir: str | None = None,
    ):
        if not tracking_uri:
            raise ValueError(
                "Missing mlflow tracking uri."
                "You can provide it as an argument or set the MLFLOW_TRACKING_URI environment variable."
            )

        self._download_dir = download_dir or self._download_dir
        self._tracking_uri = tracking_uri
        self._experiment = mlflow.get_experiment_by_name(
            experiment_name
        ) or mlflow.get_experiment(mlflow.create_experiment(experiment_name))
        self._create_or_update_run(run_name)

    def _create_or_update_run(self, run_name: str):
        runs = mlflow.search_runs(
            experiment_ids=[self._experiment.experiment_id],
            filter_string=f'tags.mlflow.runName = "{run_name}"',
            output_format="list",
        )

        run_id = runs[0].info.run_id if runs else None

        with (
            mlflow.start_run(
                experiment_id=self._experiment.experiment_id, run_id=run_id
            )
            if run_id
            else mlflow.start_run(
                experiment_id=self._experiment.experiment_id,
                run_name=run_name,
            )
        ) as run:
            self._run = run

    def load(self, file_key: str, base_directory: str | None = None, **kwargs) -> Any:
        artifact_key = super()._compute_filepath(file_key, base_directory)
        downloaded_key = super()._compute_filepath(artifact_key, self._download_dir)

        mlflow.artifacts.download_artifacts(
            run_id=self._run.info.run_id,
            artifact_path=artifact_key,
            dst_path=self._download_dir,
        )

        try:
            return super().load(file_key=downloaded_key, **kwargs)
        finally:
            if os.path.exists(self._download_dir):
                shutil.rmtree(self._download_dir)
