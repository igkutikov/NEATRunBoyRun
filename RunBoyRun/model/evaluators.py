import typing
from RunBoyRun.model import interfaces
from RunBoyRun.model import characters

class NEATStepCounerEvaluator(interfaces.INEATEvaluator):
    def __init__(self, character: characters.INEATTraineeCharacter, board: interfaces.IBoard):
        interfaces.INEATEvaluator.__init__(self, character, board)
              

    def evaluate(self, decision: typing.Optional[typing.List[float]], verbose: bool = False) -> float:
        if self._character.Dead:
            if not self._dead:
                self._dead = True
            return 0.0
        
        self._prev_decision = self._decision
        self._decision = decision
        self._prev_decision_kind == self._decision_kind
        self._nPrevClick = self._nClick
        self._nPrevNoClick = self._nNoClick
        self._nPrevMissClick = self._nMissClick
        self._nPrevNoMissClick = self._nNoMissClick

        if interfaces.CELL_PLATFORM == self._character.PrevCellKind and self._decision is not None and len(self._decision) > 0:
            if self._decision[0] > interfaces.DECISION_THRESHOLD:
                self._decision_kind = interfaces.E_DecisionKind.CLICK
                self._nClick += 1
            else:
                self._decision_kind = interfaces.E_DecisionKind.NO_CLICK
                self._nNoClick += 1
        elif self._decision is not None and len(self._decision) > 0:
            if self._decision[0] > interfaces.DECISION_THRESHOLD:
                self._decision_kind = interfaces.E_DecisionKind.MISS_CLICK
                self._nMissClick += 1
            else:
                self._decision_kind = interfaces.E_DecisionKind.NO_MISS_CLICK
                self._nNoMissClick += 1
        else:
            self._decision_kind = interfaces.E_DecisionKind.NONE
        
        if self._character.CellKind == interfaces.CELL_PLATFORM:
            self._last_platform_step = self._character.nSteps
        self._fitness += 1

        return self._fitness
    

class NEATLastPlatformPenaltyEvaluator(interfaces.INEATEvaluator):
    def __init__(self, character: characters.INEATTraineeCharacter, board: interfaces.IBoard):
        interfaces.INEATEvaluator.__init__(self, character, board)
              

    def evaluate(self, decision: typing.Optional[typing.List[float]], verbose: bool = False) -> float:
        if self._character.Dead:
            if not self._dead:
                self._dead = True
                self._fitness -= (self._character.nSteps - self._last_platform_step)
                return self._fitness
            return 0.0
        
        self._prev_decision = self._decision
        self._decision = decision
        self._prev_decision_kind == self._decision_kind
        self._nPrevClick = self._nClick
        self._nPrevNoClick = self._nNoClick
        self._nPrevMissClick = self._nMissClick
        self._nPrevNoMissClick = self._nNoMissClick

        if interfaces.CELL_PLATFORM == self._character.PrevCellKind and self._decision is not None and len(self._decision) > 0:
            if self._decision[0] > interfaces.DECISION_THRESHOLD:
                self._decision_kind = interfaces.E_DecisionKind.CLICK
                self._nClick += 1
            else:
                self._decision_kind = interfaces.E_DecisionKind.NO_CLICK
                self._nNoClick += 1
        elif self._decision is not None and len(self._decision) > 0:
            if self._decision[0] > interfaces.DECISION_THRESHOLD:
                self._decision_kind = interfaces.E_DecisionKind.MISS_CLICK
                self._nMissClick += 1
            else:
                self._decision_kind = interfaces.E_DecisionKind.NO_MISS_CLICK
                self._nNoMissClick += 1
        else:
            self._decision_kind = interfaces.E_DecisionKind.NONE
        
        if self._character.CellKind == interfaces.CELL_PLATFORM:
            self._last_platform_step = self._character.nSteps

        self._fitness += 1
        return self._fitness


# class NEATEvaluator2(INEATEvaluator):
#     def __init__(self, board: boards.IBoard, posY: int, direction: definitions.PROGRESSION_DIRECTION):
#         INEATEvaluator.__init__(self, board, posY, direction)
      

#     def evaluate(self, dataprovider: NEATDataProvider, decision: typing.Optional[typing.List[float]]) -> float:
#         self._prev_decision = decision
#         self._prev_decision_kind == self._decision_kind
#         self._prev_direction = self._direction
#         self._prev_fitness = self._fitness
#         self._prev_posY = self._posY
#         self._prev_cell_kind = self._cell_kind
#         self._fitness = 0.0

#         if self._board.get_cell(0, self._posY) == definitions.CELL_PLATFORM and decision is not None:
#             if self.activation(decision):
#                 self._decision_kind = E_DecisionKind.CLICK
#                 self._nClick += 1
#                 if self._direction == 'S':
#                     self._direction = 'N'
#                     self._posY -= 1
#                 else: # self._direction == 'N':
#                     self._direction = 'S'
#                     self._posY += 1
#                 self._cell_kind = dataprovider.board.get_cell(0, self._posY)
#                 if self._cell_kind == definitions.CELL_FENCE or self._cell_kind == definitions.CELL_SAW:
#                     self._dead = True
#                 else:
#                     self._fitness += 1
#             else:
#                 self._decision_kind = E_DecisionKind.NO_CLICK
#                 self._nNoClick += 1
                
#                 self._cell_kind = dataprovider.board.get_cell(0, self._posY)
#                 if self._prev_cell_kind != definitions.CELL_PLATFORM or self._cell_kind != definitions.CELL_PLATFORM:
#                     if self._direction == 'N':
#                         self._posY += 1
#                     else: # self._direction == 'N'
#                         self._posY -= 1
#                     self._cell_kind = dataprovider.board.get_cell(0, self._posY)

#                 if self._cell_kind == definitions.CELL_FENCE or self._cell_kind == definitions.CELL_SAW:
#                     self._dead = True
#                 else:
#                     self._fitness += 1
#         elif decision is not None:
#             if self.DecisionActive:
#                 self._decision_kind = E_DecisionKind.MISS_CLICK
#                 self._nMissClick += 1
#             else:
#                 self._decision_kind = E_DecisionKind.NO_MISS_CLICK
#                 self._nNoMissClick += 1
            
#             self._cell_kind = dataprovider.board.get_cell(0, self._posY)
#             if self._prev_cell_kind != definitions.CELL_PLATFORM or self._cell_kind != definitions.CELL_PLATFORM:
#                 if self._direction == 'S':
#                     self._posY += 1
#                 else:
#                     self._posY -= 1
#                 self._cell_kind = dataprovider.board.get_cell(0, self._posY)

#             if self._cell_kind == definitions.CELL_FENCE or self._cell_kind == definitions.CELL_SAW:
#                 self._dead = True
#         else:
#             self._decision_kind = E_DecisionKind.NONE

#             self._cell_kind = dataprovider.board.get_cell(0, self._posY)
#             if self._prev_cell_kind != definitions.CELL_PLATFORM or self._cell_kind != definitions.CELL_PLATFORM:
#                 if self._direction == 'S':
#                     self._posY += 1
#                 else:
#                     self._posY -= 1
#                 self._cell_kind = dataprovider.board.get_cell(0, self._posY)

#             if self._cell_kind == definitions.CELL_FENCE or self._cell_kind == definitions.CELL_SAW:
#                 self._dead = True

#         if not self._dead:
#             if self._cell_kind == definitions.CELL_PLATFORM:
#                 self._last_platfom_step = self._nSteps
#             self._nSteps += 1
#         else: 
#             self._fitness -= self._nSteps - self._last_platfom_step
            
#         return self._fitness