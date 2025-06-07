import typing
import pygame
from RunBoyRun import model
from RunBoyRun.GUI import interfaces
from RunBoyRun.GUI import boards
from RunBoyRun.GUI import characters

class RunBoyRun:
    def __init__(self, fps: float, board: model.boards.Board, *chars: typing.Tuple[model.interfaces.ICharacter]):
        self._board: model.boards.Board = board

        pygame.init()

        if fps < 0.1:
            fps = 0.1
        if fps > 60.0:
            fps = 60.0
        
        self.__fps: float = fps
        self.__pause: bool = False

        self.__key_down_reentrancy: typing.List[bool] = [False, False, False, False, False]
        self.__clock: pygame.time.Clock = pygame.time.Clock()
        self.__fps_font: pygame.font.Font = pygame.font.SysFont('Ariel', 28, bold=True)
        self.__pause_surface: pygame.Surface = pygame.font.SysFont('Ariel', 48, bold=True).render('PAUSE', True, pygame.Color('RED'))

        self.__gui_board: boards.Board = boards.Board((800, 600), self._board)
        pygame.display.set_caption('Run Boy Run!!!')

        self.__screen: pygame.SurfaceType = pygame.display.set_mode((self.__gui_board.ScreeWidth, self.__gui_board.ScreeHeight))
        self.__gui_characters: typing.List[interfaces.ICharacter] = []

        colors: typing.List[interfaces.RGB] = interfaces.generate_colors_palette(len(chars))
        for idx, character in enumerate(chars):
            self.__gui_characters.append(characters.CharactersFactory.create_character(idx, self.__gui_board, character, colors[idx]))

        self.__gui_board.render(self.__screen, *self.__gui_characters)
        pygame.display.flip()


    def __del__(self) -> None:
        pygame.quit()


    def __handle_keys(self) -> None:
        scan_codes: pygame.key.ScancodeWrapper = pygame.key.get_pressed()
        if scan_codes[pygame.K_UP]:
            if not self.__key_down_reentrancy[0]:
                self.__key_down_reentrancy[0] = True
                self.__fps += 0.1
                if self.__fps > 60:
                    self.__fps = 60
                print(f'set timer: {self.__fps}')
        else:
            self.__key_down_reentrancy[0]
        
        if scan_codes[pygame.K_DOWN]:
            if not self.__key_down_reentrancy[1]:
                self.__key_down_reentrancy[1] = True
                self.__fps -= 0.1
                if self.__fps <= 0.2:
                    self.__fps = 0.2
                print(f'set timer: {self.__fps}')
        else:
            self.__key_down_reentrancy[1] = False
        
        if scan_codes[pygame.K_PAGEUP]:
            if not self.__key_down_reentrancy[2]:
                self.__key_down_reentrancy[2] = True
                self.__fps += 5.0
                if self.__fps > 60:
                    self.__fps = 60
                print(f'set timer: {self.__fps}')
        else:
            self.__key_down_reentrancy[2] = False

        if scan_codes[pygame.K_PAGEDOWN]:
            if not self.__key_down_reentrancy[3]:
                self.__key_down_reentrancy[3] = True
                self.__fps -= 5.0
                if self.__fps <= 0.2:
                    self.__fps = 0.2
                print(f'set timer: {self.__fps}')
        else:
            self.__key_down_reentrancy[3] = False

        
        if scan_codes[pygame.K_p]:
            if not self.__key_down_reentrancy[4]:
                self.__key_down_reentrancy[4] = True
                self.__pause = not self.__pause
        else:
            self.__key_down_reentrancy[4] = False


    def run(self) -> None:
        if len(self.__gui_characters) <= 0:
            return
        
        running: int = 4
        while running > 0:
            self.__clock.tick_busy_loop(self.__fps)
            events: typing.List[pygame.event.Event] = pygame.event.get()
            self.__handle_keys()

            if not self.__pause:       
                for gui_char in self.__gui_characters:
                    gui_char.do_step()

                if not self.__gui_board.proceed():
                    break

            self.__gui_board.render(self.__screen, *self.__gui_characters)
            self.__screen.blit(self.__fps_font.render(f'{self.__clock.get_fps():.02f}', True, pygame.Color('RED')), (self.__screen.get_width() - 64, 4))
            
            if not self.__pause:
                for gui_char in self.__gui_characters:
                    if not gui_char.Dead:
                        break
                else:
                    break
            else:
                self.__screen.blit(self.__pause_surface, 
                    (
                        (self.__screen.get_width() / 2) - (self.__pause_surface.get_width() / 2),
                        (self.__screen.get_height() / 2) - (self.__pause_surface.get_height() / 2)
                    )
                )
                                
            pygame.display.flip()


        gameover_surface: pygame.Surface = pygame.font.SysFont('Ariel', 48, bold=True).render('GAME OVER', True, pygame.Color('RED'))

        while running > 0:
            self.__gui_board.render(self.__screen, *self.__gui_characters)
            self.__screen.blit(gameover_surface, 
                (
                    (self.__screen.get_width() / 2) - (gameover_surface.get_width() / 2),
                    (self.__screen.get_height() / 2) - (gameover_surface.get_height() / 2)
                )
            )
            pygame.display.flip()
            self.__clock.tick(self.__fps)
            running -= 1
