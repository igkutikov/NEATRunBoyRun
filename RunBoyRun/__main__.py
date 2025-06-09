import typing
import os
import sys
import argparse
import matplotlib.pyplot
import keyboard

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model
import GUI
import training

parser = argparse.ArgumentParser(description='NEAT Intergrated Run Boy Run')
parser.add_argument('--mode', help='train, plot or play', required=False, default=None, choices=('play', 'train', 'plot'), dest='mode')

def play() -> None:
    board: model.boards.Board = model.boards.Board(model.maps.MapFactory.create_map('static'))
    characters: typing.List[model.interfaces.ICharacter] = [
        model.characters.NEATCharacter(0, board, 1, 'SE', model.evaluators.NEATStepCounerEvaluator),
    ]

    game: GUI.runboyrun.RunBoyRun = GUI.runboyrun.RunBoyRun(20, board, *characters)
    game.run()

    print("Press Any to exit...")
    keyboard.wait() # Waits for any key press


def train() -> None:
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerFF('random', model.evaluators.NEATStepCounerEvaluator)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerFF('static_with_obstacles', model.evaluators.NEATStepCounerEvaluator)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerFF('static', model.evaluators.NEATStepCounerEvaluator)
    trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerFF('random', model.evaluators.NEATLastPlatformPenaltyEvaluator, 60)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerFF('static_with_obstacles', model.evaluators.NEATLastPlatformPenaltyEvaluator, 40)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerFF('static', model.evaluators.NEATLastPlatformPenaltyEvaluator, 40)
    
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerRecurrent('random', model.evaluators.NEATStepCounerEvaluator)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerRecurrent('static_with_obstacles', model.evaluators.NEATStepCounerEvaluator)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerRecurrent('static', model.evaluators.NEATStepCounerEvaluator)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerRecurrent('random', model.evaluators.NEATStepCounerEvaluator, 40)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerRecurrent('static_with_obstacles', model.evaluators.NEATLastPlatformPenaltyEvaluator, 40)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerRecurrent('static', model.evaluators.NEATLastPlatformPenaltyEvaluator, 40)

    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerIZNN('random', model.evaluators.NEATStepCounerEvaluator)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerIZNN('static_with_obstacles', model.evaluators.NEATStepCounerEvaluator)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.RunBoyRunTrainerIZNN('static', model.evaluators.NEATStepCounerEvaluator)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerIZNN('random', model.evaluators.NEATStepCounerEvaluator, 40)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerIZNN('static_with_obstacles', model.evaluators.NEATStepCounerEvaluator, 40)
    #trainer: training.trainers.IRunBoyRunTrainer = training.trainers.GUIRunBoyRunTrainerIZNN('static', model.evaluators.NEATStepCounerEvaluator, 40)


    trainer.train(9000)
    del trainer

    stats: typing.List[training.statistics.TrainingStatisticsEntry] = training.statistics.load_statistics_json(training.JSON_STATISTICS_FPATH)
    training.statistics.plot_best_fitness(stats)
    training.statistics.plot_average_fitness(stats)
    matplotlib.pyplot.show()


def plot() -> None:
    stats: typing.List[training.statistics.TrainingStatisticsEntry] = training.statistics.load_statistics_json(training.JSON_STATISTICS_FPATH)
    training.statistics.plot_best_fitness(stats)
    training.statistics.plot_average_fitness(stats)
    matplotlib.pyplot.show()


args = parser.parse_args()
if not hasattr(args, 'mode') or args.mode is None or args.mode == 'play':
    play()
elif args.mode == 'train':
    train()
elif args.mode == 'plot':
    plot()