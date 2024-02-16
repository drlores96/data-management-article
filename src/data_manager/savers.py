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
from matplotlib.figure import FigureBase
from mlflow.entities import Experiment, Run
import shutil
from s3fs.core import S3FileSystem
import pickle


class Saver(Protocol):
    _configure: Callable[..., Any]
    save: Callable[..., Any]


class DBMSSaver(Saver, Protocol):
    _connectable: Engine

    def save(
        self, data: pd.DataFrame, table: str, schema: str | None = None, **kwargs
    ) -> None:
        with self._connectable.connect() as con:
            data.to_sql(
                name=table,
                con=con,
                schema=schema,
                index=False,
                **kwargs,
            )


class SnowflakeSaver(DBMSSaver):
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

    def save(
        self, data: pd.DataFrame, table: str, schema: str | None = None, **kwargs
    ) -> None:
        super().save(data, table=table, schema=schema, **kwargs)


class FileSaver(Saver, Protocol):
    _filesystem: LocalFileSystem | S3FileSystem

    @staticmethod
    def _compute_filepath(file_key: str, base_directory: str | None = None) -> str: ...

    def save(self, data, file_key: str, base_directory: str | None = None, **kwargs) -> None:
        filepath = self._compute_filepath(file_key, base_directory)

        if filepath.endswith(".parquet") and isinstance(data, pd.DataFrame):
            with self._filesystem.open(filepath, "wb") as f:
                data.to_parquet(f, index=False, **kwargs)

        elif filepath.endswith(".csv") and isinstance(data, pd.DataFrame):
            with self._filesystem.open(filepath, "wb") as f:
                data.to_csv(f, index=False, **kwargs)

        elif (filepath.endswith(".xlsx") or filepath.endswith(".xls")) and isinstance(
            data, pd.DataFrame
        ):
            with self._filesystem.open(filepath, "wb") as f:
                data.to_excel(f, index=False, **kwargs)

        elif (filepath.endswith(".pickle") or filepath.endswith(".pkl")) and isinstance(
            data, pd.DataFrame
        ):
            with self._filesystem.open(filepath, "wb") as f:
                data.to_pickle(f, **kwargs)

        elif filepath.endswith(".json") and isinstance(data, dict):
            with self._filesystem.open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, **kwargs)

        elif (filepath.endswith(".yaml") or filepath.endswith("yml")) and isinstance(
            data, dict
        ):
            with self._filesystem.open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, **kwargs)

        elif filepath.endswith(".pickle") or filepath.endswith(".pkl"):
            with self._filesystem.open(filepath, "wb") as f:
                pickle.dump(data, f, **kwargs)

        elif filepath.endswith(".html") or filepath.endswith(".log"):
            with self._filesystem.open(filepath, "w", encoding="utf-8") as f:
                f.write(data, **kwargs)

        elif (
            filepath.endswith(".png")
            or filepath.endswith(".jpeg")
            or filepath.endswith(".svg")
        ) and issubclass(type(data), FigureBase):
            with self._filesystem.open(filepath, "wb") as f:
                data.savefig(f, **kwargs)


class LocalFileSaver(FileSaver):
    _filesystem: LocalFileSystem

    def __init__(self, **filesystem_args):
        self._configure(**filesystem_args)

    def _configure(self, **filesystem_args):
        self._filesystem = fsspec.filesystem("file", **filesystem_args)

    @staticmethod
    def _compute_filepath(file_key: str, base_directory: str | None = None) -> str:
        return os.path.join(base_directory or "", file_key)

    def save(self, data, file_key: str, base_directory: str | None = None, **kwargs) -> None:
        filepath = self._compute_filepath(file_key, base_directory)
        if (
            "auto_mkdir" not in self._filesystem.__dict__.keys()
            and not self._filesystem.exists(filepath)
        ):
            raise ValueError("Specified filepath doesn't exist.")

        super().save(data, file_key=file_key, base_directory=base_directory, **kwargs)


class S3FileSaver(FileSaver):
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

    def save(self, data, file_key: str, base_directory: str | None = None, **kwargs) -> None:
        return super().save(
            data, file_key=file_key, base_directory=base_directory, **kwargs
        )


class MlflowFileSaver(LocalFileSaver):
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

        super().__init__(**{**filesystem_args, **{"auto_mkdir": True}})

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
        tracking_uri: str | None = None,
        download_dir: str | None = None,
    ):
        if not tracking_uri:
            raise ValueError(
                "Missing mlflow tracking uri."
                "You can provide it as an argument or set the MLFLOW_TRACKING_URI environment variable."
            )

        self._download_dir = download_dir or self._download_dir
        self._tracking_uri = tracking_uri or self._tracking_uri
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

    def save(self, data, file_key: str, base_directory: str | None = None, **kwargs) -> None:
        artifact_key = super()._compute_filepath(file_key, base_directory)
        artifact_path = os.path.dirname(artifact_key)
        downloaded_key = super()._compute_filepath(artifact_key, self._download_dir)

        super().save(data, file_key=downloaded_key, **kwargs)

        try:
            with mlflow.start_run(self._run.info.run_id):
                mlflow.log_artifact(downloaded_key, artifact_path=artifact_path)
        finally:
            if os.path.exists(self._download_dir):
                shutil.rmtree(self._download_dir)

        self._create_or_update_run(self._run.info.run_name)
