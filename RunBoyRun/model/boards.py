import typing
import numpy
from RunBoyRun.model import interfaces
from RunBoyRun.model import maps


class Board(interfaces.IBoard):
    def __init__(self, map: interfaces.IMap) -> None:
        interfaces.IBoard.__init__(self)
        self.__col_index: int = 0
        if isinstance(map, Board):
            self.__map: interfaces.IMap = typing.cast(Board, map).__map.clone()
        else:
            self.__map: interfaces.IMap = map
            ending_cols: typing.List[int] = [interfaces.CELL_PLATFORM] * self.nCols
            ending_cols[0] = interfaces.CELL_FENCE
            ending_cols[-1] = interfaces.CELL_FENCE
            ending_cols: typing.List[typing.List[int]] = [ending_cols] * (self.nCols - 1)
            self.__map.extend(ending_cols)
        

    @property
    def nRows(self) -> int:
        return self.__map.nRows
    
    @property
    def nCols(self) -> int:
        return self.__map.nRows
    

    @property
    def nOverallRows(self) -> int:
        return self.__map.nRows
    
    @property
    def nOverallCols(self) -> int:
        return self.__map.nCols


    @property
    def Progress(self) -> int:
        return self.__col_index


    @property
    def Map(self) -> interfaces.IMap:
        return self.__map


    @property
    def display(self) -> numpy.ndarray:
        return self.__map.slice((self.__col_index, self.__col_index+self.nCols))
    

    def is_coord_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.nRows and 0 <= y < self.nCols


    def get_cell(self, x: int, y: int) -> typing.Optional[int]:
        if not self.is_coord_valid(x, y):
            return None
        return self.__map.get_cell(self.__col_index + x, y)


    def print_map(self, pos: typing.Optional[typing.Tuple[int, int]] = None) -> None:
        if pos is not None and (pos[0] < 0 or pos[0] >= self.nCols or pos[1] < 0 or pos[1] >= self.nRows):
            pos = None
        if pos is not None:
            pos = (self.__col_index + pos[0], pos[1])
        self.__map.print_map((self.__col_index, self.__col_index + self.nCols), pos=pos)


    def proceed(self) -> bool:
        if self.__col_index + self.nCols >= self.__map.nCols:
            return False
        self.__col_index += 1
        self._notify()
        return True


    def clone(self) -> interfaces.IBoard:
        return Board(self)


if __name__ == "__main__":
    board: Board = Board(maps.MapFactory.create_map('random'))
    board.print_map((0, 3))
    board.proceed()
    board.print_map((0, 3))
    board.proceed()
    board.print_map((0, 3))
    board.proceed()
    board.print_map((0, 3))