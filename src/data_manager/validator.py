from typing import Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def check(cls, data: pd.DataFrame) -> bool:
        data_dict = data.to_dict(orient="records")
        try:
            [cls.model_validate(dictionary) for dictionary in data_dict]
            return True
        except:
            return False


class Iris(Schema):
    sepal_length: Optional[float] = Field(gt=0)
    sepal_width: Optional[float] = Field(gt=0)
    petal_length: Optional[float] = Field(gt=0)
    petal_width: Optional[float] = Field(gt=0)
    species: Optional[str] = Field(max_length=10)