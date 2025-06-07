import typing
import os
import io
import csv
import abc
import model.interfaces
import neat.config
import neat.population
import neat.genome
import RunBoyRun.training.reporters.checkpointer as checkpointer
import pickle


def load_config(config_path: str) -> neat.config.Config:
    """
    Load the NEAT configuration from the specified file path.
    
    :param config_path: Path to the NEAT configuration file.
    :return: NEAT configuration object.
    """
    return neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def create_population(population_path: str) -> neat.population.Population:
     # Create the NEAT population
    if os.path.exists(population_path):
        population: neat.Population = checkpointer.CheckPointerReporter.load_population(population_path)
    else:
        config: neat.config.Config = load_config(os.path.dirname(__file__).join('neat-config.ini'))
        population: neat.Population = neat.Population(config)
    
    # Add reporters to track progress
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(checkpointer.CheckPointerReporter(population_path))

    return population


def save_history(reporter: neat.StatisticsReporter, out_dir: str) -> None:
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fitenss_history_fpath: str = os.path.join(out_dir, 'fitness_history.csv')
    speciation_fpath: str = os.path.join(out_dir, 'speciation.csv')
    species_fitness_fpath: str = os.path.join(out_dir, 'species_fitness.csv')
    training_hisotry_fpath: str = os.path.join(out_dir, 'best_genomes.csv')
    reporter.save_genome_fitness(delimiter=',', filename=fitenss_history_fpath)
    reporter.save_species_count(delimiter=',', filename=speciation_fpath)
    reporter.save_species_fitness(delimiter=',', filename=species_fitness_fpath)
    save_best_genomes(reporter, delimiter=',', filename=species_fitness_fpath)


def save_best_genomes(reporter: neat.StatisticsReporter, delimiter=' ', filename='best_genomes.csv') -> None:
    internal_delimeter = ';'
    if delimiter == internal_delimeter:
        internal_delimeter = ','

    with open(filename, 'w') as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(['Generation', 'Fitness', 'FitnessesMean','SpeciesAmount', 'NodesAmount', 'ConnectionsAmount'])
        
        genome: neat.genome.DefaultGenome
        generation: int
        for generation, genome in enumerate(reporter.most_fit_genomes):
            w.writerow((
                generation,
                genome.fitness,
                reporter.get_fitness_mean()[generation],
                reporter.get_species_sizes()[generation],
                len(genome.nodes),
                len(genome.connections)
            ))


def save_winner(genome: neat.genome.DefaultGenome, filename='best_genome.pkl') -> None:
    with open(filename, 'wb') as output_file:
        pickle.dump(genome, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_winner(filename='best_genome.pkl') -> neat.genome.DefaultGenome:
    with open(filename, 'rb') as output_file:
        genome: neat.genome.DefaultGenome = pickle.load(filename)
                
        

class INEATStepRecorder:
    @abc.abstractmethod
    def append(self, genome_id: int, result: typing.Optional[model.interfaces.E_GameOverResult], input_vector: typing.List[float], decision: float, pre_fitness: float, evaluated_fitness: float, topo_fitness: float, post_fitness: float, nClick: int, nNoClick: int, nMissClick: int, nNoMissClick: int) -> None: ...


class NEATStepRecorder(INEATStepRecorder):
    def __init__(self, delimiter=' ', filename='NEAT_steps.csv') -> None:
        INEATStepRecorder.__init__(self)
        self.__internal_delimeter: str = ';'
        if delimiter == self.__internal_delimeter:
            self.__internal_delimeter = ','
        
        self.__output_file: io.TextIOWrapper = open(filename, 'w')
        self.__csv_writer: csv.Writer = csv.writer(self.__output_file, delimiter=delimiter)
        self.__csv_writer.writerow(['Genome ID', 'GameResult' ,'InputMap', 'PositionY', 'Direction', 'Cell', 'Decision', 'Pre-Fitness', 'Evaluatedf-Fitnes', 'Topo-Fitness','Post-Fitness', 'NClick', 'NNoClick', 'NMissClick', 'NNoMissClick'])
    
        
    def append(self, genome_id: int, result: typing.Optional[model.interfaces.E_GameOverResult], input_vector: typing.List[float], decision: float, pre_fitness: float, evaluated_fitness: float, topo_fitness: float, post_fitness: float, nClick: int, nNoClick: int, nMissClick: int, nNoMissClick: int) -> None:
        input_map: str = self.__internal_delimeter.join(str(value) for value in input_vector[:-3])
        self.__csv_writer.writerow([genome_id, 'N/A' if result is None else result.name, input_map, input_vector[-3], 'N' if input_vector[-2] != 0 else 'S', chr(int(input_vector[-1])), decision, pre_fitness, evaluated_fitness, topo_fitness, post_fitness, nClick, nNoClick, nMissClick, nNoMissClick])


    def __del__(self) -> None:
        if self.__output_file:
            self.__output_file.close()
            self.__output_file = None


class NullNEATStepRecorder:
    def __init__(self) -> None:
        INEATStepRecorder.__init__(self)
    
        
    def append(self, genome_id: int, result: typing.Optional[model.interfaces.E_GameOverResult], input_vector: typing.List[float], decision: float, pre_fitness: float, evaluated_fitness: float, topo_fitness: float, post_fitness: float, nClick: int, nNoClick: int, nMissClick: int, nNoMissClick: int) -> None:
        return


    def __del__(self) -> None:
        pass