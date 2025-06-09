import typing
import pickle
import abc
import model
import neat.config
import neat.genome
from . import interfaces



class HumanCharacter(interfaces.ICharacter):
    def __init__(self, idx: int, board: interfaces.IBoard, posY: int, direction: interfaces.PROGRESSION_DIRECTION):
        interfaces.ICharacter.__init__(self, 'Player', idx, board, posY, direction)
        self._change_direction: bool = False        


    def do_step(self, action: typing.Optional[bool], verbose: bool = False) -> typing.Tuple[int, int]:
        if self.Dead:
            return (0, 0)
        
        self._prev_direction = self._direction
        self._prev_posY = self._posY
        self._prev_posX = self._posX
        self._prev_cell_kind = self._cell_kind
        
        move: typing.Tuple[int, int]
        cell_kind: int = self._board.get_cell(self._prev_posX, self._prev_posY)
        if action is not None and action and interfaces.CELL_PLATFORM == cell_kind:
            if self._initial_direction == 'SE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            else: # self._initial_direction == 'NE'
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif interfaces.CELL_PLATFORM == cell_kind and interfaces.CELL_PLATFORM == self._board.get_cell(self._prev_posX + 1, self._prev_posY):
            # do not update initial direction => self._initial_direction
            # it is a restore point once of the series of consecutive platforms
            self._direction = 'E'
            move = (1, 0)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif cell_kind in (interfaces.CELL_MID_AIR, interfaces.CELL_PLATFORM):
            if self._initial_direction == 'SE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            else: # self._initial_direction == 'NE'
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        else:
            self._dead = True
            move = (0, 0)

        return move
    

    def game_over(self, result: interfaces.E_GameOverResult) -> None:
        return
        # ICharacter.game_over(self, result)
        # if result == game_common.E_GameOverResult.WIN:
        #     self.__genome.fitness += 100.0
        # elif result == game_common.E_GameOverResult.LOSE:
        #     self.__genome.fitness -= 50.0
        # elif result == game_common.E_GameOverResult.DRAW:
        #     self.__genome.fitness += 10.0


class NEATGraduate(typing.NamedTuple):
    genome: neat.genome.DefaultGenome
    config: neat.config.Config


class NEATCharacter(interfaces.ICharacter):
    def __init__(self, idx: int, board: interfaces.IBoard, posY: int, direction: interfaces.PROGRESSION_DIRECTION, evaluator: typing.Type[interfaces.INEATEvaluator]):
        interfaces.ICharacter.__init__(self, 'AI', idx, board, posY, direction)
        self.__graduate: NEATGraduate = self.__load_graduate()
        self.__data_provider: interfaces.NEATDataProvider = interfaces.NEATDataProvider(self._board)
        self.__genome_id: int = self.__graduate.genome.key
        self.__network: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(self.__graduate.genome, self.__graduate.config)
        self.__graduate.genome.fitness = 0.0
        self.__evaluator: interfaces.INEATEvaluator = evaluator(self, self._board)
        self.__decision: typing.Optional[typing.List[float]] = None


    def __load_graduate(self) -> NEATGraduate:
        with open(model.PICKLE_GRADUATE_FPATH, "rb") as pklf:
            graduate: NEATGraduate = pickle.load(pklf)
        return graduate

#region properties

    @property
    def GenomeId(self) -> int:
        return self.__genome_id
    
    @property
    def Evaluator(self) -> interfaces.INEATEvaluator:
        return self.__evaluator
    
    @property
    def EvaluatedFitness(self) -> float:
        return self.__evaluator.Fitness
      

    @property
    def Fitness(self) -> float:
        return self.__graduate.genome.fitness
   
    
    @property
    def Decision(self) -> float:
        return self.__evaluator.Decision


    @property
    def DecisionActive(self) -> bool:
        return self.__evaluator.DecisionActive
    
    
    @property
    def DecisionKind(self) -> interfaces.E_DecisionKind:
        return self.__evaluator.DecisionKind


    @property
    def nClick(self) -> int:
        return self.__evaluator.nClick
    

    @property
    def nPrevClick(self) -> int:
        return self.__evaluator.nPrevClick
    

    @property
    def nNoClick(self) -> int:
        return self.__evaluator.nNoClick
    

    @property
    def nPrevNoClick(self) -> int:
        return self.__evaluator.nPrevNoClick
    

    @property
    def nMissClick(self) -> int:
        return self.__evaluator.nMissClick
    

    @property
    def nPrevMissClick(self) -> int:
        return self.__evaluator.nPrevMissClick
    

    @property
    def nNoMissClick(self) -> int:
        return self.__evaluator.nNoMissClick
    
    @property
    def nPrevNoMissClick(self) -> int:
        return self.__evaluator.nPrevNoMissClick
    
    
    @property
    def nSteps(self) -> int:
        return self._nSteps

#endregion properties
    
    def react(self) -> typing.Optional[bool]:
        if self.Dead:
            self._decision = None
            return None

        if self.__decision is not None and len(self.__decision) > 0:
            return self.__decision[0] > interfaces.DECISION_THRESHOLD

        if (self.__decision is None or len(self.__decision) == 0) and interfaces.CELL_PLATFORM == self._cell_kind:
            neat_input: typing.List[float] = self.__data_provider.generate_neat_input(self._posX, self._posY, self._direction)
            self.__decision = self.__network.activate(neat_input)
            return self.__decision is not None and len(self.__decision) > 0 and self.__decision[0] > interfaces.DECISION_THRESHOLD

        return None
            

    def do_step(self, action: typing.Optional[bool], verbose: bool = False) -> typing.Tuple[int, int]:
        if self.Dead:
            return (0, 0)
        
        self._prev_direction = self._direction
        self._prev_posY = self._posY
        self._prev_posX = self._posX
        self._prev_cell_kind = self._cell_kind
        
        move: typing.Tuple[int, int]
        cell_kind: int = self._board.get_cell(self._prev_posX, self._prev_posY)
        if action is not None and action and interfaces.CELL_PLATFORM == cell_kind:
            if self._initial_direction == 'SE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            else: # self._initial_direction == 'NE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif interfaces.CELL_PLATFORM == cell_kind and interfaces.CELL_PLATFORM == self._board.get_cell(self._prev_posX + 1, self._prev_posY):
            # do not update initial direction => self._initial_direction
            # it is a restore point once of the series of consecutive platforms
            self._direction = 'E'
            move = (1, 0)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif cell_kind in (interfaces.CELL_MID_AIR, interfaces.CELL_PLATFORM):
            if self._initial_direction == 'SE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            else: # self._initial_direction == 'NE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        else:
            self._dead = True
            move = (0, 0)

        self.__graduate.genome.fitness += self.__evaluator.evaluate(self.__decision, verbose)
        self.__decision = None
        return move
        

    def game_over(self, result: interfaces.E_GameOverResult) -> None:
        return
        # ICharacter.game_over(self, result)
        # if result == game_common.E_GameOverResult.WIN:
        #     self.__genome.fitness += 100.0
        # elif result == game_common.E_GameOverResult.LOSE:
        #     self.__genome.fitness -= 50.0
        # elif result == game_common.E_GameOverResult.DRAW:
        #     self.__genome.fitness += 10.0


class INEATTraineeCharacter(interfaces.ICharacter):
    def __init__(self, idx: int, board: interfaces.IBoard, posY: int, direction: interfaces.PROGRESSION_DIRECTION, evaluator: typing.Type[interfaces.INEATEvaluator], genome_id: int):
        interfaces.ICharacter.__init__(self, 'AI', idx, board, posY, direction)
        self._genome_id: int = genome_id
        self._data_provider: interfaces.NEATDataProvider = interfaces.NEATDataProvider(board)
        self._evaluator: interfaces.INEATEvaluator = evaluator(self, self._board)
        self._decision: typing.Optional[typing.List[float]] = None

#region Properties

    @property
    def GenomeId(self) -> int:
        return self._genome_id


    @property
    def Evaluator(self) -> interfaces.INEATEvaluator:
        return self._evaluator


    @property
    def EvaluatedFitness(self) -> float:
        return self._evaluator.Fitness


    @property
    def Decision(self) -> float:
        return self._evaluator.Decision


    @property
    def DecisionActive(self) -> bool:
        return self._evaluator.DecisionActive


    @property
    def DecisionKind(self) -> interfaces.E_DecisionKind:
        return self._evaluator.DecisionKind


    @property
    def nClick(self) -> int:
        return self._evaluator.nClick


    @property
    def nPrevClick(self) -> int:
        return self._evaluator.nPrevClick


    @property
    def nNoClick(self) -> int:
        return self._evaluator.nNoClick


    @property
    def nPrevNoClick(self) -> int:
        return self._evaluator.nPrevNoClick


    @property
    def nMissClick(self) -> int:
        return self._evaluator.nMissClick


    @property
    def nPrevMissClick(self) -> int:
        return self._evaluator.nPrevMissClick


    @property
    def nNoMissClick(self) -> int:
        return self._evaluator.nNoMissClick


    @property
    def nPrevNoMissClick(self) -> int:
        return self._evaluator.nPrevNoMissClick


    @property
    @abc.abstractmethod
    def Fitness(self) -> float: ... 

#endregion Properties


    def game_over(self, result: interfaces.E_GameOverResult) -> None:
        return
        # ICharacter.game_over(self, result)
        # if result == game_common.E_GameOverResult.WIN:
        #     self.__genome.fitness += 100.0
        # elif result == game_common.E_GameOverResult.LOSE:
        #     self.__genome.fitness -= 50.0
        # elif result == game_common.E_GameOverResult.DRAW:
        #     self.__genome.fitness += 10.0


class ANEATFFTraineeCharacter(INEATTraineeCharacter):
    def __init__(self, idx: int, board: interfaces.IBoard, posY: int, direction: interfaces.PROGRESSION_DIRECTION, genome_id: int, genome: neat.genome.DefaultGenome, config: neat.config.Config, evaluator: typing.Type[interfaces.INEATEvaluator]):
        INEATTraineeCharacter.__init__(self, idx, board, posY, direction, evaluator, genome_id)
        self._genome: neat.genome.DefaultGenome = genome
        self._genome.fitness = 0.0

#region Properties

    @property
    def Fitness(self) -> float:
        return self._genome.fitness


#endregion Properties

    def do_step(self, action: typing.Optional[bool], verbose: bool = False) -> typing.Tuple[int, int]:
        if self.Dead:
            return (0, 0)

        self._prev_direction = self._direction
        self._prev_posY = self._posY
        self._prev_posX = self._posX
        self._prev_cell_kind = self._cell_kind
        
        move: typing.Tuple[int, int]
        cell_kind: int = self._board.get_cell(self._prev_posX, self._prev_posY)
        if action is not None and action and interfaces.CELL_PLATFORM == cell_kind:
            if self._initial_direction == 'SE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            else: # self._initial_direction == 'NE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif interfaces.CELL_PLATFORM == cell_kind and interfaces.CELL_PLATFORM == self._board.get_cell(self._prev_posX + 1, self._prev_posY):
            # do not update initial direction => self._initial_direction
            # it is a restore point once of the series of consecutive platforms
            self._direction = 'E'
            move = (1, 0)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif cell_kind in (interfaces.CELL_MID_AIR, interfaces.CELL_PLATFORM):
            if self._initial_direction == 'SE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            else: # self._initial_direction == 'NE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        else:
            self._dead = True
            move = (0, 0)

        self._genome.fitness = self._evaluator.evaluate(self._decision, verbose)
        self._decision = None
        return move


class NEATFFTraineeCharacter(ANEATFFTraineeCharacter):
    def __init__(self, idx: int, board: interfaces.IBoard, posY: int, direction: interfaces.PROGRESSION_DIRECTION, genome_id: int, genome: neat.genome.DefaultGenome, config: neat.config.Config, evaluator: typing.Type[interfaces.INEATEvaluator]):
        ANEATFFTraineeCharacter.__init__(self, idx, board, posY, direction, genome_id, genome, config, evaluator)
        self.__network: neat.nn.FeedForwardNetwork = neat.nn.FeedForwardNetwork.create(genome, config)


    def react(self) -> typing.Optional[bool]:
        if self.Dead:
            self._decision = None
            return None

        if self._decision is not None and len(self._decision) > 0:
            return self._decision[0] > interfaces.DECISION_THRESHOLD

        if (self._decision is None or len(self._decision) == 0) and interfaces.CELL_PLATFORM == self._cell_kind:
            neat_input: typing.List[float] = self._data_provider.generate_neat_input(self._posX, self._posY, self._direction)
            self._decision = self.__network.activate(neat_input)
            return self._decision is not None and len(self._decision) > 0 and self._decision[0] > interfaces.DECISION_THRESHOLD

        return None
 

class NEATRecurrentTraineeCharacter(ANEATFFTraineeCharacter):
    def __init__(self, idx: int, board: interfaces.IBoard, posY: int, direction: interfaces.PROGRESSION_DIRECTION, genome_id: int, genome: neat.genome.DefaultGenome, config: neat.config.Config, evaluator: typing.Type[interfaces.INEATEvaluator]):
        ANEATFFTraineeCharacter.__init__(self, idx, board, posY, direction, genome_id, genome, config, evaluator)
        self.__network: neat.nn.RecurrentNetwork = neat.nn.RecurrentNetwork.create(genome, config)


    def react(self) -> typing.Optional[bool]:
        if self.Dead:
            self._decision = None
            return None

        if self._decision is not None and len(self._decision) > 0:
            return self._decision[0] > interfaces.DECISION_THRESHOLD

        if (self._decision is None or len(self._decision) == 0) and interfaces.CELL_PLATFORM == self._cell_kind:
            neat_input: typing.List[float] = self._data_provider.generate_neat_input(self._posX, self._posY, self._direction)
            self._decision = self.__network.activate(neat_input)
            return self._decision is not None and len(self._decision) > 0 and self._decision[0] > interfaces.DECISION_THRESHOLD

        return None


class NEATIZNNTraineeCharacter(INEATTraineeCharacter):
    def __init__(self, idx: int, board: interfaces.IBoard, posY: int, direction: interfaces.PROGRESSION_DIRECTION, genome_id: int, genome: neat.iznn.IZGenome, config: neat.config.Config, evaluator: typing.Type[interfaces.INEATEvaluator]):
        INEATTraineeCharacter.__init__(self, idx, board, posY, direction, evaluator, genome_id)
        self.__genome: neat.iznn.IZGenome = genome
        self.__network: neat.iznn.IZNN = neat.iznn.IZNN.create(genome, config)

#region properties

    @property
    def Fitness(self) -> float:
        return self.__genome.fitness

#endregion properties



    def react(self) -> typing.Optional[bool]:
        if self.Dead:
            self._decision = None
            return None

        if self._decision is not None and len(self._decision) > 0:
            return self._decision[0] > interfaces.DECISION_THRESHOLD

        if (self._decision is None or len(self._decision) == 0) and interfaces.CELL_PLATFORM == self._cell_kind:
            neat_input: typing.List[float] = self._data_provider.generate_neat_input(self._posX, self._posY, self._direction)
            self.__network.set_inputs(neat_input)
            self._decision = self.__network.advance(self.nSteps)
            return self._decision is not None and len(self._decision) > 0 and self._decision[0] > interfaces.DECISION_THRESHOLD

        return None


    def do_step(self, action: typing.Optional[bool], verbose: bool = False) -> typing.Tuple[int, int]:
        if self.Dead:
            return (0, 0)

        self._prev_direction = self._direction
        self._prev_posY = self._posY
        self._prev_posX = self._posX
        self._prev_cell_kind = self._cell_kind
        
        move: typing.Tuple[int, int]
        cell_kind: int = self._board.get_cell(self._prev_posX, self._prev_posY)
        if action is not None and action and interfaces.CELL_PLATFORM == cell_kind:
            if self._initial_direction == 'SE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            else: # self._initial_direction == 'NE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif interfaces.CELL_PLATFORM == cell_kind and interfaces.CELL_PLATFORM == self._board.get_cell(self._prev_posX + 1, self._prev_posY):
            # do not update initial direction => self._initial_direction
            # it is a restore point once of the series of consecutive platforms
            self._direction = 'E'
            move = (1, 0)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif cell_kind in (interfaces.CELL_MID_AIR, interfaces.CELL_PLATFORM):
            if self._initial_direction == 'SE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            else: # self._initial_direction == 'NE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        else:
            self._dead = True
            move = (0, 0)

        self.__genome.fitness = self._evaluator.evaluate(self._decision, verbose)
        self._decision = None
        return move


class NEATCRTNNTraineeCharacter(INEATTraineeCharacter):
    def __init__(self, idx: int, board: interfaces.IBoard, posY: int, direction: interfaces.PROGRESSION_DIRECTION, genome_id: int, genome: neat.genome.DefaultGenome, config: neat.config.Config, evaluator: typing.Type[interfaces.INEATEvaluator]):
        INEATTraineeCharacter.__init__(self, idx, board, posY, direction, evaluator, genome_id)
        self.__genome: neat.genome.DefaultGenome = genome
        self.__network: neat.ctrnn.CTRNN = neat.ctrnn.CTRNN.create(genome, config)

#region properties

    @property
    def Fitness(self) -> float:
        return self.__genome.fitness

#endregion properties

    def react(self) -> typing.Optional[bool]:
        if self.Dead:
            self._decision = None
            return None

        if self._decision is not None and len(self._decision) > 0:
            return self._decision[0] > interfaces.DECISION_THRESHOLD

        if (self._decision is None or len(self._decision) == 0) and interfaces.CELL_PLATFORM == self._cell_kind:
            neat_input: typing.List[float] = self._data_provider.generate_neat_input(self._posX, self._posY, self._direction)
            self._decision = self.__network.activate(neat_input)
            return self._decision is not None and len(self._decision) > 0 and self._decision[0] > interfaces.DECISION_THRESHOLD

        return None


    def do_step(self, action: typing.Optional[bool], verbose: bool = False) -> typing.Tuple[int, int]:
        if self.Dead:
            return (0, 0)
        
        self._prev_direction = self._direction
        self._prev_posY = self._posY
        self._prev_posX = self._posX
        self._prev_cell_kind = self._cell_kind

        move: typing.Tuple[int, int]
        cell_kind: int = self._board.get_cell(self._prev_posX, self._prev_posY)
        if action is not None and action and interfaces.CELL_PLATFORM == cell_kind:
            if self._initial_direction == 'SE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            else: # self._initial_direction == 'NE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif interfaces.CELL_PLATFORM == cell_kind and interfaces.CELL_PLATFORM == self._board.get_cell(self._prev_posX + 1, self._prev_posY):
            # do not update initial direction => self._initial_direction
            # it is a restore point once of the series of consecutive platforms
            self._direction = 'E'
            move = (1, 0)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        elif cell_kind in (interfaces.CELL_MID_AIR, interfaces.CELL_PLATFORM):
            if self._initial_direction == 'SE':
                self._direction = 'SE'
                self._initial_direction = self._direction
                self._posY += 1
                move = (1, 1)
            else: # self._initial_direction == 'NE':
                self._direction = 'NE'
                self._initial_direction = self._direction
                self._posY -= 1
                move = (1, -1)
            self._cell_kind = self._board.get_cell(1, self._posY)
            if self._cell_kind in (interfaces.CELL_FENCE, interfaces.CELL_SAW):
                self._dead = True
            else:
                self._nSteps += 1
        else:
            self._dead = True
            move = (0, 0)

        self.__genome.fitness = self._evaluator.evaluate(self._decision, verbose)
        self._decision = None
        return move


# def __topological_search(self, row_idx: int, col_idx: int, direction: game_common.PROGRESSION_DIRECTION) -> int:
    #     steps: int = 0
    #     if self._direction == 'S':
    #         while row_idx < self._board.nRows and col_idx < self._board.nCols:
    #             cell_kind: int = self._board.get_cell(col_idx, row_idx)
    #             if cell_kind == game_common.CELL_SAW or cell_kind == game_common.CELL_FENCE:
    #                 return steps if steps > 0 else -1
    #             elif cell_kind == game_common.CELL_PLATFORM:
    #                 if col_idx + 1 < self._board.nCols and self._board.get_cell(col_idx + 1, row_idx) == game_common.CELL_PLATFORM:
    #                     steps_same_dir: int = self.__topological_search(row_idx, col_idx + 1, direction)
    #                 else:
    #                     steps_same_dir: int = self.__topological_search(row_idx+1, col_idx + 1, direction)
                    
    #                 steps_other_dir: int = self.__topological_search(row_idx - 1, col_idx + 1, 'N')
    #                 return  steps + max(steps_same_dir, steps_other_dir) + 1
                
    #             #else: cell_kind == game_common.CELL_MID_AIR:
    #             steps += 1
    #             row_idx += 1
    #             col_idx += 1
    #     else: # self._direction == 'N':
    #         while row_idx >= 0 and col_idx < self._board.nCols:
    #             cell_kind: int = self._board.get_cell(col_idx, row_idx)
    #             if cell_kind == game_common.CELL_SAW or cell_kind == game_common.CELL_FENCE:
    #                 return steps if steps > 0 else -1
    #             elif cell_kind == game_common.CELL_PLATFORM:
    #                 if col_idx + 1 < self._board.nCols and self._board.get_cell(col_idx + 1, row_idx) == game_common.CELL_PLATFORM:
    #                     steps_same_dir: int = self.__topological_search(row_idx, col_idx + 1, direction)
    #                 else:
    #                     steps_same_dir: int = self.__topological_search(row_idx - 1, col_idx + 1, direction)
                    
    #                 steps_other_dir: int = self.__topological_search(row_idx - 1, col_idx + 1, 'S')
    #                 return  steps + max(steps_same_dir, steps_other_dir) + 1
    #             #else: cell_kind == game_common.CELL_MID_AIR:
    #             steps -= 1
    #             row_idx -= 1
    #             col_idx += 1

    #     return steps


    # def __calculate_fitness(self, display: boards.BoardSanp) -> float:
    #     cell_kind: int = display.get_cell(0, self._position_y)

    #     self.__topological_search_fitness = 0.0
    #     if cell_kind == game_common.CELL_SAW or cell_kind == game_common.CELL_FENCE:
    #         self.__topological_search_fitness = -6.0
    #     elif display.is_platform(0, self._position_y):
    #         self.__topological_search_fitness = self.__topological_search(self.__pre_position_y, 0, self._direction)
    #     else:
    #         self.__topological_search_fitness = self.__topological_search(self.__pre_position_y, 0, self._direction)
        
    #     self.__evaluated_fitness += self.__topological_search_fitness
    #     return self.__genome.fitness + self.__evaluated_fitness  

    
    # def __calculate_fitness2(self, display: boards.BoardSanp) -> float:
    #     self.__topological_search_fitness = 0.0
    #     self.__evaluated_fitness = 0.0
    #     if self._dead:
    #         self.__evaluated_fitness = 0.1
    #     return self.__genome.fitness + self.__evaluated_fitness

