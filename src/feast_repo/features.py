from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

# Offline source: parquet created by src/data_prep.py
# NOTE: Path is relative to the Feast repo directory (src/feast_repo)
source = FileSource(
    path="../../data/athletes.parquet",
    timestamp_field="event_timestamp",
)

athlete = Entity(
    name="athlete",
    join_keys=["athlete_entity_id"],
    value_type=ValueType.INT64,
)

# Baseline features
fv_athletes_v1 = FeatureView(
    name="fv_athletes_v1",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="weight", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="howlong", dtype=Float32),
        Field(name="gender", dtype=String),
        Field(name="region", dtype=String),
        Field(name="eat", dtype=String),
        Field(name="background", dtype=String),
        Field(name="experience", dtype=String),
        Field(name="schedule", dtype=String),
        Field(name="total_lift", dtype=Float32),
    ],
    source=source,
    online=True,
)

# Engineered features
fv_athletes_v2 = FeatureView(
    name="fv_athletes_v2",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="weight", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="howlong", dtype=Float32),
        Field(name="gender", dtype=String),
        Field(name="region", dtype=String),
        Field(name="eat", dtype=String),
        Field(name="background", dtype=String),
        Field(name="experience", dtype=String),
        Field(name="schedule", dtype=String),
        Field(name="bmi", dtype=Float32),
        Field(name="weight_height_ratio", dtype=Float32),
        Field(name="bmi_age", dtype=Float32),
        Field(name="total_lift", dtype=Float32),
    ],
    source=source,
    online=True,
)
