import typing
import pygame
from RunBoyRun import model
from RunBoyRun.GUI import interfaces

class Board(interfaces.IBoard):
    def __init__(self, screen_size: typing.Tuple[int, int], board: model.interfaces.IBoard):
        interfaces.IBoard.__init__(self, screen_size, board)
        
        def generate_fence(tile_width: float, tile_height: float) -> pygame.Surface:
            fence: pygame.Surface = pygame.Surface((tile_width - 2, tile_height - 2))

            size: typing.Tuple[int, int] = fence.get_size()

            start_x = 0
            start_y = 0
            end_x = size[0] / 3
            end_y = size[1]
            
            pygame.draw.line(fence, (0, 255, 0), (start_x,start_y), (end_x, end_y))

            start_x = end_x
            end_x = (size[0] / 3) * 2
            pygame.draw.line(fence, (0, 255, 0), (start_x,start_y), (end_x, end_y))

            start_x = end_x
            end_x = size[0]
            pygame.draw.line(fence, (0, 255, 0), (start_x,start_y), (end_x, end_y))

            return fence


        def generate_saw(tile_width: float, tile_height: float) -> pygame.Surface:
            saw: pygame.Surface = pygame.Surface((tile_width * 0.8, tile_height * 0.8))
            size: typing.Tuple[int, int] = saw.get_size()

            points: typing.Tuple[pygame.Vector2, ...] = (
                pygame.Vector2(size[0] * (0/6), size[1] * (3/6)),
                pygame.Vector2(size[0] * (1/6), size[1] * (2/6)),
                pygame.Vector2(size[0] * (1/6), size[1] * (1/6)),
                pygame.Vector2(size[0] * (2/6), size[1] * (1/6)),
                pygame.Vector2(size[0] * (3/6), size[1] * (0/6)),
                pygame.Vector2(size[0] * (4/6), size[1] * (1/6)),
                pygame.Vector2(size[0] * (5/6), size[1] * (1/6)),
                pygame.Vector2(size[0] * (5/6), size[1] * (2/6)),
                pygame.Vector2(size[0] * (6/6), size[1] * (3/6)),
                pygame.Vector2(size[0] * (5/6), size[1] * (4/6)),
                pygame.Vector2(size[0] * (5/6), size[1] * (5/6)),
                pygame.Vector2(size[0] * (4/6), size[1] * (5/6)),
                pygame.Vector2(size[0] * (3/6), size[1] * (6/6)),
                pygame.Vector2(size[0] * (2/6), size[1] * (5/6)),
                pygame.Vector2(size[0] * (1/6), size[1] * (5/6)),
                pygame.Vector2(size[0] * (1/6), size[1] * (4/6)),
                pygame.Vector2(size[0] * (0/6), size[1] * (3/6)),
            )

            pygame.draw.polygon(saw, (0, 255, 0), points, width=1)
            return saw
            

        def generate_platform(tile_width: float, tile_height: float) -> pygame.Surface:
            platform: pygame.Surface = pygame.Surface((tile_width * 0.8, tile_height * 0.8))
            size: typing.Tuple[int, int] = platform.get_size()

            pygame.draw.ellipse(platform, (0, 255, 0), pygame.Rect(0.0, 0.0, size[0], size[1]), width=1)
            return platform
        

        self.__fence: pygame.Surface = generate_fence(self.TileWidth, self.TileHeight)
        self.__saw: pygame.Surface = generate_saw(self.TileWidth, self.TileHeight)
        self.__platform: pygame.Surface = generate_platform(self.TileWidth, self.TileHeight)


    def __render_mesh(self, screen: pygame.Surface) -> None:
        for x in range(0, self.NHTiles + 1):
            x = min(x * self.TileWidth, self.ScreeWidth - 1)
            pygame.draw.line(screen, (0, 0, 255), (x,0), (x, self.ScreeHeight))
        for y in range(0, self.NVTiles + 1):
            y = min(y * self.TileHeight, self.ScreeHeight - 1)
            pygame.draw.line(screen, (0, 0, 255), (0,y), (self.ScreeWidth, y))


    def __draw_fence(self, screen: pygame.Surface, col: int, row: int) -> None:
        x = (col * self.TileWidth) + 1
        y = (row * self.TileHeight) + 1
        
        screen.blit(self.__fence, (x, y))


    def __draw_saw(self, screen: pygame.Surface, col: int, row: int) -> None:
        x = (col * self.TileWidth) + self.TileWidth * 0.1
        y = (row * self.TileHeight) + self.TileHeight * 0.1
        
        screen.blit(self.__saw, (x, y))


    def __draw_platform(self, screen: pygame.Surface, col: int, row: int) -> None:
        x = (col * self.TileWidth) + self.TileWidth * 0.1
        y = (row * self.TileHeight) + self.TileHeight * 0.1
        
        screen.blit(self.__platform, (x, y))


    def __render_elements(self, screen: pygame.Surface) -> None:
        row: int = 0
        col: int = 0

        while row < self.NVTiles:
            while col < self.NHTiles:
                cell: int = self._board.get_cell(col, row)
                if model.interfaces.CELL_FENCE == cell:
                    self.__draw_fence(screen, col, row)
                elif model.interfaces.CELL_SAW == cell:
                    self.__draw_saw(screen, col, row)
                elif model.interfaces.CELL_PLATFORM == cell:
                    self.__draw_platform(screen, col, row)
                col += 1
            col = 0
            row += 1


    def render(self, screen: pygame.Surface, *characters: typing.Iterable[interfaces.ICharacter]) -> None:
        screen.fill((0,0,0))
        self.__render_elements(screen)
        for character in characters:
            character.render(screen)
        self.__render_mesh(screen)


    def proceed(self) -> bool:
        return self._board.proceed()
    

    def clone(self) -> interfaces.IBoard:
        return Board((self.ScreeWidth, self.ScreeHeight), self._board.clone())

        
