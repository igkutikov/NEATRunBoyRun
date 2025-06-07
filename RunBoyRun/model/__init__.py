import os

PICKLE_GRADUATE_FPATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'graduate.pkl')

from RunBoyRun.model import interfaces
from RunBoyRun.model import maps
from RunBoyRun.model import boards
from RunBoyRun.model import characters
from RunBoyRun.model import evaluators

__all__ = [
    'interfaces',
    'maps',
    'boards',
    'characters',
    'evaluators'
]