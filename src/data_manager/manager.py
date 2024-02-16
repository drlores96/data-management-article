from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from typing import Type, Any, Dict
from data_manager.loaders import (
    Loader,
    SnowflakeLoader,
    LocalFileLoader,
    S3FileLoader,
    MlflowFileLoader
)
from data_manager.savers import (
    Saver,
    SnowflakeSaver,
    LocalFileSaver,
    S3FileSaver,
    MlflowFileSaver
)
from data_manager.validator import Schema

ImplementedLoaders: Dict[str, Type[Loader]]= {
    "snowflake": SnowflakeLoader,
    "local": LocalFileLoader,
    "S3": S3FileLoader,
    "mlflow": MlflowFileLoader
}

ImplementedSavers: Dict[str, Type[Saver]] = {
    "snowflake": SnowflakeSaver,
    "local": LocalFileSaver,
    "S3": S3FileSaver,
    "mlflow": MlflowFileSaver
}

class ValidationError(Exception):
    pass


def get_data_manager(
        loader_type: str = "local",
        loader_args: Dict[Any, Any] = {},
        saver_type: str = "local",
        saver_args: Dict[Any, Any] = {},
    )-> DataManager:

        return DataManager(
            ImplementedLoaders.get(loader_type)(**loader_args), 
            ImplementedSavers.get(saver_type)(**saver_args)
        )


@dataclass
class DataManager:
    loader: Loader
    saver: Saver

    def _validate(
        self,
        data: pd.DataFrame,
        validator: Type[Schema] | None = None,
        validation_type: str = "none",
    ) -> bool:
        if validation_type == "none" and not validator:
            return True
        elif not validator:
            raise ValueError(
                f"Missing validator for validation_type: {validation_type}"
            )
        elif validation_type == "all" and validator:
            return validator.check(data)
        elif (
            validation_type.isdigit()
            and int(validation_type) in range(100)
            and validator
        ):
            return validator.check(data.sample(int(validation_type)))
        else:
            raise ValueError(
                f"Unrecognized validation type: {validation_type}. Accepted values are 'none', 'all' or integer representing the percentage of data to validate, i.e. '30'"
            )

    def load(
        self, validator: Type[Schema] | None = None, validation_type: str = "none", **kwargs
    ) -> Any:
        data = self.loader.load(**kwargs)
        if not self._validate(
            data, validator=validator, validation_type=validation_type
        ):
            raise ValidationError("Data doesn't match given schema")
        else:
            return data

    def save(
        self, data, validator: Type[Schema] | None = None, validation_type: str = "none", **kwargs
    ) -> None:
        if not self._validate(
            data, validator=validator, validation_type=validation_type
        ):
            raise ValidationError("Data doesn't match given schema")
        else:
            self.saver.save(data, **kwargs)
