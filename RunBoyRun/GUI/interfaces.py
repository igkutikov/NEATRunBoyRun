import typing
import math
import abc
import colorsys
import pygame
from RunBoyRun import model


class FRect(typing.Sized, typing.Iterable[float]):
    @typing.overload
    def __init__(self, left: float, top: float, width: float, height: float) -> None: ...
    @typing.overload
    def __init__(self, pos: typing.Union[pygame.math.Vector2, typing.Tuple[float, float]], size: typing.Union[pygame.math.Vector2, typing.Tuple[float, float]]) -> None: ...

    def __init__(self, *args) -> None:
        if len(args) == 4:
            self.__coords: typing.List[float] = list(args)
        else:
            self.__coords: typing.List[float] = [*args[0], *args[1]]

    @property
    def x(self) -> float:
        return self.__coords[0]

    @x.setter
    def x(self, x: float) -> None:
        self.__coords[0] = x

    
    @property
    def y(self) -> float:
        return self.__coords[1]
    
    @y.setter
    def y(self, y: float) -> None:
        self.__coords[1] = y


    @property
    def w(self) -> float:
        return self.__coords[2]

    @w.setter
    def w(self, w: float) -> None:
        self.__coords[2] = w

    
    @property
    def h(self) -> float:
        return self.__coords[3]

    @h.setter
    def h(self, h: float) -> None:
        self.__coords[3] = h
    

    @property
    def centerx(self) -> float:
        return self.x + (self.w / 2)

    @property
    def centery(self) -> float:
        return self.y + (self.h / 2)


    @property
    def left(self) -> float:
        return self.x
    
    @left.setter
    def left(self, left: float) -> None:
        self.x = left

    
    @property
    def top(self) -> float:
        return self.y
    
    @top.setter
    def top(self, top: float) -> None:
        self.y = top


    @property
    def width(self) -> float:
        return self.w
    
    @width.setter
    def width(self, width: float) -> None:
        self.w = width

    
    @property
    def height(self) -> float:
        return self.h
    
    @height.setter
    def height(self, height: float) -> None:
        self.h = height


    @property
    def right(self) -> float:
        return self.left + self.width

    
    @property
    def bottom(self) -> float:
        return self.top + self.height
    

    @property
    def topleft(self) -> typing.Tuple[float, float]:
        return self.left, self.top
    
    
    @property
    def bottomleft(self) -> typing.Tuple[float, float]:
        return self.left, self.bottom


    @property
    def topright(self) -> typing.Tuple[float, float]:
        return self.right , self.top


    @property
    def bottomright(self) -> typing.Tuple[float, float]:
        return self.right, self.bottom


    @property
    def midtop(self) -> typing.Tuple[float, float]:
        return self.centerx, self.top


    @property
    def midleft(self) -> typing.Tuple[float, float]:
        return self.left, self.centery


    @property
    def midbottom(self) -> typing.Tuple[float, float]:
        return self.centerx, self.bottom


    @property
    def midright(self) -> typing.Tuple[float, float]:
            self.right, self.centery


    @property
    def center(self) -> typing.Tuple[float, float]:
        return self.centerx, self.centery
    

    def __len__(self) -> typing.Literal[4]:
        return len(self.__coords)
    

    def __iter__(self) -> typing.Iterator[float]:
        return self.__coords.__iter__()
    
    def __getitem__(self, index: typing.Union[int, slice]) -> typing.Union[float, typing.List[float]]:
        raise self.__coords[index]
    

RGB = typing.Tuple[int, int, int]


def generate_colors_palette(nPlayers: int) -> typing.List[RGB]:
    colors: typing.List[RGB] = []
    min_v = 0.7  # Avoid dark colors (value/brightness)
    min_s = 0.7  # Avoid grayish colors (saturation)
    for i in range(nPlayers):
        # Evenly distribute hues, full saturation and value
        hue = i / nPlayers
        r, g, b = colorsys.hsv_to_rgb(hue, min_s, min_v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


class IBoard:
    def __init__(self, screen_size: typing.Tuple[int, int], board: model.boards.Board):
        self._board: model.boards.Board = board
        self.__scree_width: int = screen_size[0]
        self.__scree_height: int = screen_size[1]


    @property
    def Model(self) -> model.interfaces.IBoard:
        return self._board


    @property
    def ScreeWidth(self) -> int:
        return self.__scree_width
    

    @property
    def ScreeHeight(self) -> int:
        return self.__scree_height
    

    @property
    def NHTiles(self) -> int:
         return self._board.nRows 
    

    @property
    def NVTiles(self) -> int:
        return self._board.nCols
    
    
    @property
    def TileWidth(self) -> float:
        return self.__scree_width / self.NHTiles
    

    @property
    def TileHeight(self) -> float:
        return self.__scree_height / self.NVTiles
    

    def tile2pixel(self, tileX: int, tileY: int) -> pygame.Vector2:
        return pygame.Vector2(tileX*self.TileWidth, tileY*self.TileHeight)
    

    @abc.abstractmethod
    def render(self, screen: pygame.Surface) -> None: ...


    @abc.abstractmethod
    def clone(self) -> 'IBoard': ...
      

class ICharacter:
    def __init__(self, board: IBoard, character: model.interfaces.ICharacter, color: RGB):
        self._board: IBoard = board
        self._character: model.interfaces.ICharacter = character
        self._color: RGB = color
        
        dx: int = 1 if self._character.Direction in ('SE', 'S', 'NE', 'N') else 0
        dy: int = 1 if self._character.Direction in ('SE', 'S') else -1 if self._character.Direction in ('NE', 'N') else 0
        
        self._theta: float = math.atan2(-dy * self._board.TileHeight, dx * self._board.TileWidth)
        self._theta = math.degrees(self._theta)
        self._moved: bool = 0 != dx or 0 != dy
        self._dead: int = 0

#region Properties

    @property
    def Id(self) -> int:
        return self._character.Id


    @property
    def TileX(self) -> int:
        return self._character.PosX
    

    @property
    def TileY(self) -> int:
        return self._character.PosY
    

    @property
    def Dead(self) -> bool:
        return self._character.Dead
    

    @property
    def Theta(self) -> float:
        return self._theta


    @property
    def Moved(self) -> bool:
        return self._moved

#endregion Properties

    def do_step(self, verbose: bool = False) -> FRect:
        pos: pygame.Vector2 = pygame.Vector2(*self._board.tile2pixel(self.TileX, self.TileY))
        rect: FRect = FRect(pos.x, pos.y, self._board.TileWidth, self._board.TileHeight)

        dx: int
        dy: int
        dx, dy = self._do_step(False)
        
        tileX: int = self.TileX
        tileY: int = self.TileY

        tileX %= self._board.NHTiles
        tileY %= self._board.NVTiles

        self._moved = 0 != dx or 0 != dy
        self._theta = math.atan2(-dy * self._board.TileHeight, dx * self._board.TileWidth)
        self._theta = math.degrees(self._theta)
        
        return rect
    

    @abc.abstractmethod
    def _do_step(self, verbose: bool = False) -> typing.Tuple[int, int]: ...
        

    def render(self, screen: pygame.Surface) -> None:
        if self._character.Dead:
            self._dead += 1
            if self._dead > 2:
                self._dead = 2

        if self._dead < 2:
            self._render(screen)
