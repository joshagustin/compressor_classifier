from dataclasses import dataclass
from typing import Callable

@dataclass
class PreprocessingConfig:
    normalize: bool
    remove_stopwords: bool
    stem: bool

@dataclass
class ExperimentConfig:
    compressor_name: str
    agg_func: Callable
    dis_func: Callable
    k: int
    candidate_count: int
    para: bool
    cos_sim: bool
    ncd: bool
    oracle: bool