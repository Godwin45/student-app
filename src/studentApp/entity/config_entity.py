from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class FeatureConfig:
    root_dir: Path
    student_df: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_df: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    training_data: Path
    trained_model_path: Path