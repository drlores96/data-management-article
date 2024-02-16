import pandas as pd
from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def check(cls, data: pd.DataFrame) -> bool:
        data_dict = data.to_dict(orient="records")
        try:
            [cls.model_validate(**dictionary) for dictionary in data_dict]
            return True
        except:
            return False
