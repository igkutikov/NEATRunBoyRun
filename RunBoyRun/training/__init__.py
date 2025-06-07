
import os

JSON_STATISTICS_FPATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_stats.json')
CSV_STATISTICS_FPATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_stats.csv')
INI_NEAT_CONFIG_FPATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'neat-config.ini')
JSON_STEP_TRACE_FPATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'steps_trace_stats.json')

from RunBoyRun.training import trainers
from RunBoyRun.training import statistics

__all__ = [
    'trainers',
    'statistics'
]