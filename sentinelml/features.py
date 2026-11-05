"""The shared training/serving contract. Inputs are hourly observations."""
from datetime import datetime, timedelta
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

FEATURE_VERSION = 'hourly-window-v1'
WINDOW = 6
HORIZON = 6
SENSORS = ('temperature', 'vibration', 'pressure', 'load')
FEATURE_NAMES = [f'{s}_{stat}' for s in SENSORS for stat in ('last', 'mean', 'std', 'slope')]

class Reading(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    timestamp: datetime
    temperature: float = Field(ge=-40, le=200)
    vibration: float = Field(ge=0, le=50)
    pressure: float = Field(ge=0, le=300)
    load: float = Field(ge=0, le=100)

    @field_validator('timestamp')
    @classmethod
    def timezone_required(cls, value):
        if value.tzinfo is None:
            raise ValueError('timestamp must include a timezone')
        return value

def build_features(readings: list[Reading]) -> np.ndarray:
    if len(readings) != WINDOW:
        raise ValueError(f'Exactly {WINDOW} hourly readings are required')
    for previous, current in zip(readings, readings[1:]):
        if current.timestamp - previous.timestamp != timedelta(hours=1):
            raise ValueError('Readings must be ordered at contiguous one-hour intervals')
    features = []
    for sensor in SENSORS:
        values = np.array([getattr(row, sensor) for row in readings])
        features.extend([values[-1], values.mean(), values.std(), np.polyfit(np.arange(WINDOW), values, 1)[0]])
    return np.asarray(features)
