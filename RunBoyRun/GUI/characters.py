import typing
import pygame
from RunBoyRun import model
from RunBoyRun.GUI import interfaces


class APoorGUICharacter(interfaces.ICharacter):
    def __init__(self, board: interfaces.IBoard, character: model.interfaces.ICharacter, color: interfaces.RGB):
        interfaces.ICharacter.__init__(self, board, character, color)
        self._arrow: pygame.Surface = pygame.Surface((self._board.TileWidth * 0.33333, self._board.TileHeight * 0.33333)).convert_alpha()
        self._still: pygame.Surface = pygame.Surface((self._board.TileWidth * 0.33333, self._board.TileHeight * 0.33333)).convert_alpha()

        size: typing.Tuple[int, int] = self._arrow.get_size()
        points: typing.Tuple[pygame.Vector2, ...] = (
            pygame.Vector2(0, size[1] / 3),
            pygame.Vector2(size[0] / 2, size[1] / 3),
            pygame.Vector2(size[0] / 2, 0),

            pygame.Vector2(size[0], size[1] / 2),

            pygame.Vector2(size[0] / 2, size[1]),
            pygame.Vector2(size[0] / 2, (size[1] / 3) * 2),
            pygame.Vector2(0, (size[1] / 3) * 2),
        )

        pygame.draw.polygon(self._arrow, self._color, points, width=2)
        pygame.draw.rect(self._still, self._color, pygame.rect.Rect(0, 0, size[0], size[1]), width=2)


    def _render(self, screen: pygame.Surface) -> None:
        if not self._moved or self.Dead:
            image: pygame.Surface = self._still
            size: typing.Tuple[int, int] = self._still.get_size()
        else:    
            image: pygame.Surface = pygame.transform.rotate(self._arrow, self._theta)
            size: typing.Tuple[int, int] = image.get_size()
        
        position: typing.Tuple[float, float] = self._board.tile2pixel(self.TileX, self.TileY)
        position = (
            position[0] + (self._board.TileWidth / 2),
            position[1] + (self._board.TileHeight / 2)
        )
        position = (
            position[0] - (size[0] / 2),
            position[1] - (size[1] / 2)
        )
        screen.blit(image, position)


class HumanCharacter(APoorGUICharacter):
    def __init__(self, board: interfaces.IBoard, character: model.characters.HumanCharacter, color: interfaces.RGB):
        APoorGUICharacter.__init__(self, board, character, color)


    def _do_step(self, verbose: bool = False) -> typing.Tuple[int, int]:
        scan_codes: pygame.key.ScancodeWrapper = pygame.key.get_pressed()
        if scan_codes[pygame.K_SPACE] or scan_codes[pygame.KSCAN_SPACE]:
            move: typing.Tuple[int, int] = self._character.do_step(True, verbose=False)
        else:
            move: typing.Tuple[int, int] = self._character.do_step(False, verbose=False)
        return move


class NEATCharacter(APoorGUICharacter):
    def __init__(self, board: interfaces.IBoard, character: model.characters.NEATCharacter, color: interfaces.RGB):
        APoorGUICharacter.__init__(self, board, character, color)


    def _do_step(self, verbose: bool = False) -> typing.Tuple[int, int]:
        decision: typing.Optional[bool] = typing.cast(model.characters.NEATCharacter, self._character).react()
        move: typing.Tuple[int, int] = self._character.do_step(decision, verbose)
        return move


class NEATTraineeCharacter(APoorGUICharacter):
    def __init__(self, board: interfaces.IBoard, character: model.characters.INEATTraineeCharacter, color: interfaces.RGB):
        APoorGUICharacter.__init__(self, board, character, color)


    def _do_step(self, verbose: bool = False) -> typing.Tuple[int, int]:
        decision: typing.Optional[bool] = typing.cast(model.characters.INEATTraineeCharacter, self._character).react()
        move: typing.Tuple[int, int] = self._character.do_step(decision, verbose)
        return move


class CharactersFactory:
    @staticmethod
    def create_character(idx: int, board: interfaces.IBoard, character: model.interfaces.ICharacter, color: interfaces.RGB) -> interfaces.ICharacter:
        if isinstance(character, model.characters.HumanCharacter):
            return HumanCharacter(board, character, color)
        elif isinstance(character, model.characters.NEATCharacter):
            return NEATCharacter(board, character, color)
        elif isinstance(character, model.characters.NEATFFTraineeCharacter):
            return NEATTraineeCharacter(board, character, color)
        elif isinstance(character, model.characters.NEATRecurrentTraineeCharacter):
            return NEATTraineeCharacter(board, character, color)
        elif isinstance(character, model.characters.NEATIZNNTraineeCharacter):
            return NEATTraineeCharacter(board, character, color)
        return None





