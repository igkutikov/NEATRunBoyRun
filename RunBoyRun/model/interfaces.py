import typing
import abc
import enum
import numpy


CELL_TYPE = typing.Literal[' ', '#', '=', '*']
CELL_MID_AIR: int = ord(' ')  # 32
CELL_PLATFORM: int = ord('=') # 61
CELL_FENCE: int = ord('#')    # 35
CELL_SAW: int = ord('*')      # 42


CHARRACHTER_KIND = typing.Literal['Player', 'AI', 'MonteCarlo']

PROGRESSION_DIRECTION = typing.Literal['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']


class E_GameOverResult(enum.IntEnum):
    WIN = 1
    LOSE = 2
    DRAW = 3


class E_DecisionKind(enum.IntEnum):
        NONE = 0x00
        CLICK = 0x01
        NO_CLICK = 0x02
        MISS_CLICK = 0x03
        NO_MISS_CLICK = 0x04


DECISION_THRESHOLD: float = 0.5


class IMap:
    def __init__(self, nCols: int, nRows: int):
       self._nRows: int = nRows
       self._nCols: int = nCols
        
    @property
    def nRows(self) -> int:
        return self._nRows
    
    @property
    def nCols(self) -> int:
        return self._nCols


    @abc.abstractmethod
    def slice(self, rng: typing.Tuple[int, int]) -> numpy.ndarray: ...


    @abc.abstractmethod
    def is_coord_valid(self, x: int, y: int) -> bool: ...


    @abc.abstractmethod
    def get_cell(self, x: int, y: int) -> typing.Optional[int]: ...


    @abc.abstractmethod
    def print_map(self, rng: typing.Tuple[int, int], pos: typing.Optional[typing.Tuple[int, int]] = None) -> None: ...


    @abc.abstractmethod
    def clone(self) -> 'IMap': ...


    @abc.abstractmethod
    def append(self, row: typing.Collection) -> None: ...


    @abc.abstractmethod
    def extend(self, cols: typing.Collection[typing.Collection]) -> None: ...


class IBoard:
    OnProgressEventCallback = typing.Callable[[], None]
    def __init__(self):
        self._subscribtions: typing.List[IBoard.OnProgressEventCallback] = []

    @property
    @abc.abstractmethod
    def nRows(self) -> int: ...
    

    @property
    @abc.abstractmethod
    def nCols(self) -> int: ...
    

    @property
    @abc.abstractmethod
    def nOverallRows(self) -> int: ...
    

    @property
    @abc.abstractmethod
    def nOverallCols(self) -> int: ...


    @property
    @abc.abstractmethod
    def Progress(self) -> int: ...


    @property
    @abc.abstractmethod
    def Map(self) -> IMap: ...


    @property
    @abc.abstractmethod
    def display(self) -> numpy.ndarray: ...


    @abc.abstractmethod
    def is_coord_valid(self, x: int, y: int) -> bool: ...


    @abc.abstractmethod
    def get_cell(self, x: int, y: int) -> typing.Optional[int]: ...


    @abc.abstractmethod
    def print_map(self, pos: typing.Optional[typing.Tuple[int, int]] = None) -> None: ...


    def subscribe(self, callback: OnProgressEventCallback) -> None:
        self._subscribtions.append(callback)


    def unsubscribe(self, callback: OnProgressEventCallback) -> None:
        while callback in self._subscribtions:
            self._subscribtions.remove[callback]


    def _notify(self) -> None:
        for callback in self._subscribtions:
            callback()

    
    @abc.abstractmethod
    def clone() -> 'IBoard': ...


class ICharacter:
    def __init__(self, kind: CHARRACHTER_KIND, idx: int, board: IBoard, posY: int, direction: PROGRESSION_DIRECTION):
        self.__kind: CHARRACHTER_KIND = kind
        self.__idx: int = idx
        self._board: IBoard = board
        self._posY: int = posY
        self._prev_posY: int = posY
        self._posX: int = 0
        self._prev_posX: int = posY
        self._initial_direction: PROGRESSION_DIRECTION = direction
        self._direction: PROGRESSION_DIRECTION = direction
        self._prev_direction: PROGRESSION_DIRECTION = direction
        self._nSteps: int = 0
        self._cell_kind: int = self._board.get_cell(self._prev_posY, self._posY)
        self._prev_cell_kind: int = self._cell_kind
        self._dead: bool = False
        self._game_over_result: typing.Optional[E_GameOverResult] = None # None = pending
        
#region Properties

    @property
    def kind(self) -> CHARRACHTER_KIND:
        return self.__kind
    

    @property
    def Id(self) -> int:
        return self.__idx


    @property
    def PosY(self) -> int:
        return self._posY


    @property
    def PrevPosY(self) -> int:
        return self._prev_posY


    @property
    def PosX(self) -> int:
        return self._posX


    @property
    def Direction(self) -> PROGRESSION_DIRECTION:
        return self._direction


    @property
    def PrevDirection(self) -> PROGRESSION_DIRECTION:
        return self._prev_direction


    @property
    def CellKind(self) -> int:
        return self._cell_kind
    

    @property
    def PrevCellKind(self) -> int:
        return self._prev_cell_kind
    

    @property
    def Dead(self) -> bool:
        return self._dead


    @property
    def nSteps(self) -> int:
        return self._nSteps

#endregion Properties

    @abc.abstractmethod
    def do_step(self, action: typing.Optional[bool], verbose: bool = False) -> typing.Tuple[int, int]: ...


class NEATDataProvider:
    def __init__(self, board: IBoard):
        self.__board: IBoard = board
        self.__data: numpy.ndarray = numpy.full((board.nRows, board.nCols), float(CELL_FENCE), dtype=numpy.float32).flatten()
        self.__board.subscribe(self.__on_progress)
        self.__on_progress()

    def __on_progress(self) -> None:
        self.__data = self.__board.display.astype(numpy.float32).flatten()

    @property
    def board(self) -> IBoard:
        return self.__board

    def generate_neat_input(self, position_x: int, position_y: int, direction: PROGRESSION_DIRECTION) -> typing.List[float]:
        data: typing.List[float] = [
            float(position_y), # 1
            1.0 if direction in ('S', 'SE') else -1.0 if direction in ('N', 'NE') else 0.0, # 1
            *self.__data, # 81
            float(self.__board.nOverallCols - self.__board.Progress) # 1
        ]
        return data


    def generate_neat_input2(self, position_x: int, position_y: int, direction: PROGRESSION_DIRECTION) -> typing.List[float]:
        data: typing.List[float] = [
            self.__board.nRows,
            self.__board.nCols,
            *self.__data,
            float(position_x),
            float(position_y),
            1.0 if direction in ('S', 'SE') else -1.0 if direction in ('N', 'NE') else 0.0,
            float(self.__board.get_cell(0, position_y))
        ]
        return data


class INEATEvaluator:
    def __init__(self, character: ICharacter, board: IBoard):
        self._character: ICharacter = character
        self._board: IBoard = board
        self._fitness: float = 0.0
        self._prev_fitness: float = 0.0
        self._decision: typing.Optional[typing.List[float]] = None
        self._prev_decision: typing.Optional[typing.List[float]] = None
        self._decision_kind: E_DecisionKind = E_DecisionKind.NONE
        self._prev_decision_kind: E_DecisionKind = E_DecisionKind.NONE
        self._nClick: int = 0
        self._nPrevClick: int = 0
        self._nNoClick: int = 0
        self._nPrevNoClick: int = 0
        self._nMissClick: int = 0
        self._nPrevMissClick: int = 0
        self._nNoMissClick: int = 0
        self._nPrevNoMissClick: int = 0
        self._last_platform_step: int = 0
        self._dead: bool = False
        
#region Properties

    @property
    def Fitness(self) -> float:
        return self._fitness
    

    @property
    def PrevFitness(self) -> float:
        return self._prev_fitness
    

    @property
    def Decision(self) -> typing.Collection[float]:
        return self._decision
    

    @property
    def PrevDecision(self) -> typing.Collection[float]:
        return self._prev_decision


    @property
    def DecisionActive(self) -> bool:
        return self._decision is not None and len(self._decision) > 0 and self._decision[0] > DECISION_THRESHOLD
    

    @property
    def PrevDecisionActive(self) -> bool:
        return self._prev_decision is not None and len(self._prev_decision) > 0 and  self._prev_decision[0] > DECISION_THRESHOLD


    @property
    def DecisionKind(self) -> E_DecisionKind:
        return self._decision_kind
    

    @property
    def PrevDecisionKind(self) -> E_DecisionKind:
        return self._prev_decision_kind
     

    @property
    def nClick(self) -> int:
        return self._nClick
    

    @property
    def nPrevClick(self) -> int:
        return self._nPrevClick
    

    @property
    def nNoClick(self) -> int:
        return self._nNoClick
    

    @property
    def nPrevNoClick(self) -> int:
        return self._nPrevNoClick
    

    @property
    def nMissClick(self) -> int:
        return self._nMissClick
    

    @property
    def nPrevMissClick(self) -> int:
        return self._nPrevMissClick
    

    @property
    def nNoMissClick(self) -> int:
        return self._nNoMissClick   
    
    @property
    def nPrevNoMissClick(self) -> int:
        return self._nPrevNoMissClick
    

    @property
    def nLastPlatformStep(self) -> int:
        return self._last_platform_step
    

    @property
    def Dead(self) -> bool:
        return self._dead
        
#endregion Properties

    @abc.abstractmethod
    def evaluate(self, decision: typing.Optional[typing.List[float]], verbose: bool) -> float: ...
