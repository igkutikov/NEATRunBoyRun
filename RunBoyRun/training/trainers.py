import typing
import abc
import time
import pygame
import pickle
import neat.config
import neat.population
import neat.genome
import model
import GUI
import training
from . import statistics


class TrainingGenerationStats:
    GENERATION: int = 0
    def __init__(self, generations: int, species: int):
        self.TotalTurns: int = 0
        self.nClick: int = 0
        self.nNoClick: int = 0
        self.nMissClick: int = 0
        self.nNoMissClick: int = 0
        self.BestFitness: float = 0.0
        self.Survivors: int = species
        self.Casualties: int = 0

        self.nTurnBasedClick: int = 0
        self.nTurnBasedNoClick: int = 0
        self.nTurnBasedMissClick: int = 0
        self.nTurnBasedNoMissClick: int = 0
        self.TurnBasedBestFitness: float = 0.0
        self.TurnBasedSurvivors: int = species
        self.TurnBasedCasualties: int = 0

        self.__generations: int = generations
        TrainingGenerationStats.GENERATION += 1
        self.__generation: int = TrainingGenerationStats.GENERATION
        self.__species: int = species
        self.__start_time = time.time()  # Track the start time of the fitness evaluation


    @property
    def nGenerations(self) -> int:
        return self.__generations

    @property
    def Generation(self) -> int:
        return self.__generation


    def collect_data(self, characters: typing.Collection[model.characters.NEATTraineeCharacter]) -> None:
        for character in characters:
            if not character.Dead:
                self.TotalTurns += 1
                break

        self.nTurnBasedClick = 0
        self.nTurnBasedNoClick = 0
        self.nTurnBasedMissClick = 0
        self.nTurnBasedNoMissClick = 0
        self.TurnBasedBestFitness = 0.0
        
        casualties: int = 0
        survivors: int = 0

        for character in characters:
            if character.Dead:
                casualties += 1
                continue
            survivors += 1

            if self.TurnBasedBestFitness < character.Fitness:
                self.TurnBasedBestFitness = character.Fitness
        
            if model.interfaces.E_DecisionKind.CLICK == character.DecisionKind:
                self.nTurnBasedClick += character.nClick - character.nPrevClick
            elif model.interfaces.E_DecisionKind.NO_CLICK == character.Evaluator.DecisionKind:
                self.nTurnBasedNoClick += character.nNoClick - character.nPrevNoClick
            elif model.interfaces.E_DecisionKind.MISS_CLICK == character.Evaluator.DecisionKind:
                self.nTurnBasedMissClick += character.nMissClick - character.nPrevMissClick
            elif model.interfaces.E_DecisionKind.NO_MISS_CLICK == character.Evaluator.DecisionKind:
                self.nTurnBasedNoMissClick += character.nNoMissClick - character.nPrevNoMissClick

        self.nClick += self.nTurnBasedClick
        self.nNoClick += self.nTurnBasedNoClick
        self.nMissClick += self.nTurnBasedMissClick
        self.nNoMissClick += self.nTurnBasedNoMissClick

        if self.BestFitness < self.TurnBasedBestFitness:
            self.BestFitness = self.TurnBasedBestFitness
        
        self.TurnBasedSurvivors = self.Survivors - survivors
        self.TurnBasedCasualties = casualties - self.Casualties
        self.Survivors = survivors
        self.Casualties = casualties


    def print_stats(self) -> None:
        elapsed_time = time.time() - self.__start_time
        print(f'\n=== Generation Stats {self.__generation} / {self.__generations} ===')
        print(f'Elapsed Time: {elapsed_time:.4f} seconds')
        print(f'Turn: {self.TotalTurns}')
        print(f'Fitness: {self.TurnBasedBestFitness:.6f} / {self.BestFitness:.6f}')
        print(f'# Clicks: {self.nTurnBasedClick} / {self.nClick} ')
        print(f'# No Clicks: {self.nTurnBasedNoClick} / {self.nNoClick}')
        print(f'# Miss-Clicks: {self.nTurnBasedMissClick} / {self.nMissClick}')
        print(f'# No Miss-Clicks: {self.nTurnBasedNoMissClick} / {self.nNoMissClick}')
        print(f'# Casualties: {self.TurnBasedCasualties} / {self.Casualties} / {self.__species}')
        print(f'# Survivors: {self.TurnBasedSurvivors} / {self.Survivors} / {self.__species}')
        print('========================\n')


class IRunBoyRunTrainer:
    def __init__(self, map_type: model.maps.MapName, evaluator: typing.Type[model.interfaces.INEATEvaluator]):
        self._generations: int = 0
        self._generation: int = 0
        self._board: model.boards.Board = model.boards.Board(model.maps.MapFactory.create_map(map_type))
        self.__config: neat.config.Config = IRunBoyRunTrainer.__load_config(training.INI_NEAT_CONFIG_FPATH)
        self.__population: neat.population.Population = neat.population.Population(self.__config)
        self.__evaluator: typing.Type[model.interfaces.INEATEvaluator] = evaluator
        self._tracer: typing.Optional[statistics.TrainingTrunStatsTracer] = statistics.TrainingTrunStatsTracer(training.JSON_STEP_TRACE_FPATH)
        # Add reporters to track progress
        self.__population.add_reporter(neat.StdOutReporter(True))
        self.__population.add_reporter(statistics.StatisticsReporter())
        # self.__population.add_reporter(checkpointer.CheckPointerReporter(population_path))


    @staticmethod
    def __load_config(config_path: str) -> neat.config.Config:
        return neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
    )


    @property
    def _Population(self) -> neat.population.Population:
        return self.__population
    

    @property
    def _Evaluator(self) -> typing.Type[model.interfaces.INEATEvaluator]:
        return self.__evaluator


    def train(self, generations: int) -> None:
        self._generations = generations
        self._generation: int = -1
        winner: neat.genome.DefaultGenome = self._Population.run(self.__evaluate_genomes, self._generations)
       
        stats_reporter: statistics.StatisticsReporter = None 
        for reporter in self._Population.reporters.reporters:
            if isinstance(reporter, statistics.StatisticsReporter):
                stats_reporter = reporter
                break

        if stats_reporter is not None:
            self.save_statistics(stats_reporter)

        graduate: model.characters.NEATGraduate = model.characters.NEATGraduate(winner, self.__config)
        self.save_graduate(graduate)
        print(f'winner genome {winner.key} => fittens: {winner.fitness}')


    def save_statistics(self, reporter: statistics.StatisticsReporter) -> None:
        stats: typing.List[statistics.TrainingStatisticsEntry] = statistics.generate_statistics(reporter)
        statistics.save_statistics_csv(training.CSV_STATISTICS_FPATH, stats)
        statistics.save_statistics_json(training.JSON_STATISTICS_FPATH, stats)


    def save_graduate(self, graduate: model.characters.NEATGraduate) -> None:
        with open(model.PICKLE_GRADUATE_FPATH, "wb") as pcklf:
            pickle.dump(graduate, pcklf)


    def update_statistics(self, stats: TrainingGenerationStats, characters: typing.Collection[model.characters.NEATTraineeCharacter]) -> None:
        stats.collect_data(characters)

        stats_reporter: statistics.StatisticsReporter = None 
        for reporter in self._Population.reporters.reporters:
            if isinstance(reporter, statistics.StatisticsReporter):
                stats_reporter = reporter
                break

        if stats_reporter is not None:
            stats_reporter.post_step(self.__config, self.__population, dict(((char.GenomeId, char.nSteps) for char in characters)))


    def _trun_start(self, turn: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, characters: typing.Collection[model.characters.NEATTraineeCharacter]) -> None:
        if self._tracer is not None:
            self._tracer.start_turn(turn, genomes, config, characters)


    def _trun_end(self, turn: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, characters: typing.Collection[model.characters.NEATTraineeCharacter]) -> None:
        if self._tracer is not None:
            self._tracer.end_turn(turn, genomes, config, characters)


    def __evaluate_genomes(self, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config) -> None:
        if len(genomes) <= 0:
            return
        self._generation += 1

        if self._tracer is not None:
            self._tracer.start_generation(self._generation, genomes, config)

        conext: typing.Any = self._prepare_generation(self._generation, genomes, config)

        self._evaluate_generation(self._generation, genomes, config, conext)

        self._finalize_generation(self._generation, genomes, config, conext)

        if self._tracer is not None:
            self._tracer.end_generation(self._generation, genomes, config)
    

    @abc.abstractmethod
    def _prepare_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config) -> typing.Any: ...

    @abc.abstractmethod
    def _evaluate_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, context: typing.Any) -> None: ...

    @abc.abstractmethod
    def _finalize_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, context: typing.Any) -> None: ...

 
class GUIRunBoyRunTrainer(IRunBoyRunTrainer):
    class __EvaluationContext(typing.NamedTuple):
        generation_stats: TrainingGenerationStats
        gui_board: GUI.boards.Board
        model_chars: typing.List[model.interfaces.ICharacter]
        gui_chars: typing.List[GUI.interfaces.ICharacter]
        
    def __init__(self, map_type: model.maps.MapName, evaluator: typing.Type[model.interfaces.INEATEvaluator], fps: float):
        IRunBoyRunTrainer.__init__(self, map_type, evaluator)
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
        
        self.__gui_board: GUI.boards.Board = GUI.boards.Board((800, 600), self._board)
        
        pygame.display.set_caption('Train Boy Train!!!')   
        self.__screen: pygame.SurfaceType = pygame.display.set_mode((self.__gui_board.ScreeWidth, self.__gui_board.ScreeHeight), flags=pygame.SCALED)
        

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


    def _prepare_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config) -> None:
        generation_stats: TrainingGenerationStats = TrainingGenerationStats(self._generations, len(genomes))
        colors: typing.List[GUI.interfaces.RGB] = GUI.interfaces.generate_colors_palette(len(genomes))
        gui_board: GUI.boards.Board = self.__gui_board.clone()
        model_chars: typing.List[model.interfaces.ICharacter] = []
        gui_chars: typing.List[GUI.interfaces.ICharacter] = []

        for idx, (genome_id, genome) in enumerate(genomes):
            model_chars.append(model.characters.NEATTraineeCharacter(idx, gui_board.Model, 1, 'SE', genome_id, genome, config, self._Evaluator))
        
        for idx, character in enumerate(model_chars):
            gui_chars.append(GUI.characters.CharactersFactory.create_character(idx, gui_board, character, colors[idx]))

        return GUIRunBoyRunTrainer.__EvaluationContext(generation_stats, gui_board, model_chars, gui_chars)


    def _evaluate_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, context: __EvaluationContext) -> None:
        trainee_chars: typing.List[model.characters.NEATTraineeCharacter] = [model_char for model_char in context.model_chars if isinstance(model_char, model.characters.NEATTraineeCharacter)]
        keep_playing: bool = True
        turns: int = -1

        context.gui_board.render(self.__screen, *context.gui_chars)
        pygame.display.flip()
        
        while keep_playing:
            self.__clock.tick_busy_loop(self.__fps)
            events: typing.List[pygame.event.Event] = pygame.event.get()
            self.__handle_keys()
            
            if not self.__pause:
                turns += 1
                self._trun_start(turns, genomes, config, trainee_chars)
                for gui_char in context.gui_chars:
                    gui_char.do_step()

                self.update_statistics(context.generation_stats, trainee_chars)
                context.generation_stats.print_stats()
                self._trun_end(turns, genomes, config, trainee_chars)

                for gui_char in context.gui_chars:
                    if not gui_char.Dead:
                        break
                else:
                    keep_playing = False

                if not context.gui_board.proceed():
                    keep_playing = False

            context.gui_board.render(self.__screen, *context.gui_chars)
            self.__screen.blit(self.__fps_font.render(f'{self.__clock.get_fps():.02f}', True, pygame.Color('RED')), (self.__screen.get_width() - 64, 4))

            if self.__pause:
                self.__screen.blit(self.__pause_surface, 
                    (
                        (self.__screen.get_width() / 2) - (self.__pause_surface.get_width() / 2),
                        (self.__screen.get_height() / 2) - (self.__pause_surface.get_height() / 2)
                    )
                )
            pygame.display.flip()


    def _finalize_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, context: __EvaluationContext) -> None:
        running: int = 4
        while running > 0:
            context.gui_board.render(self.__screen, *context.gui_chars)
            pygame.display.flip()
            running -= 1


class RunBoyRunTrainer(IRunBoyRunTrainer):
    class __EvaluationContext(typing.NamedTuple):
        generation_stats: TrainingGenerationStats
        board: model.boards.Board
        model_chars: typing.List[model.interfaces.ICharacter]

    def __init__(self, map_type: model.maps.MapName, evaluator: typing.Type[model.interfaces.INEATEvaluator], print_maps: bool = False):
        IRunBoyRunTrainer.__init__(self, map_type, evaluator)
        self.__print_map: bool = print_maps


    def __do_step(self, character: model.interfaces.ICharacter, verbose: bool = False) -> typing.Tuple[int, int]:
        if isinstance(character, model.characters.NEATTraineeCharacter):
            decision: typing.Optional[bool] = typing.cast(model.characters.NEATTraineeCharacter, character).react()
            move: typing.Tuple[int, int] = character.do_step(decision, verbose)
        if isinstance(character, model.characters.HumanCharacter):
            # TODO Get Keybaord Decision
            move: typing.Tuple[int, int] = character.do_step(False, verbose)
        else:
            move: typing.Tuple[int, int] = (0, 0)
        return move


    def _prepare_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config) -> None:
        generation_stats: TrainingGenerationStats = TrainingGenerationStats(self._generations, len(genomes))
        board: model.boards.Board = self._board.clone()
        model_chars: typing.List[model.interfaces.ICharacter] = []

        for idx, (genome_id, genome) in enumerate(genomes):
            model_chars.append(model.characters.NEATTraineeCharacter(idx, board, 1, 'SE', genome_id, genome, config, self._Evaluator))
        
        return GUIRunBoyRunTrainer.__EvaluationContext(generation_stats, board, model_chars)


    def _evaluate_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, context: __EvaluationContext) -> None: 
        trainee_chars: typing.List[model.characters.NEATTraineeCharacter] = [model_char for model_char in context.model_chars if isinstance(model_char, model.characters.NEATTraineeCharacter)]
        keep_playing = True
        turns: int = -1

        if self.__print_map:
            context.board.print_map()

        while keep_playing:
            turns += 1
            self._trun_start(turns, genomes, config, trainee_chars)
            for model_char in context.model_chars:
                self.__do_step(model_char)
            
            self.update_statistics(context.generation_stats, trainee_chars)
            context.generation_stats.print_stats()
            self._trun_end(turns, genomes, config, trainee_chars)
            
            for model_char in context.model_chars:
                if not model_char.Dead:
                    break
            else:
                keep_playing = False

            if not context.board.proceed():
                keep_playing = False

            if keep_playing and self.__print_map:
                context.board.print_map()


    def _finalize_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, context: __EvaluationContext) -> None:
        if self.__print_map:
            while running > 0:
                context.board.print_map()
                running -= 1