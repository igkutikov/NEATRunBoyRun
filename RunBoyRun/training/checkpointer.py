import typing
import pickle
import gzip
import neat.config
import neat.population
import neat.reporting
import neat.genome
import neat.species

class CheckPointerReporter(neat.reporting.BaseReporter):
    def __init__(self, out_fpath: str, generation_interval: int = 4):
        self.__output_filepath: str = out_fpath
        self.__generation_interval: int = generation_interval
        self.__last_generation_checkpoint: int = 0
        self.__current_generation: typing.Optional[int] = None

    
    def start_generation(self, generation: int) -> None:
        self.__current_generation = generation


    def end_generation(self, config, population: typing.Dict[int, neat.genome.DefaultGenome], species_set: neat.species.DefaultSpeciesSet) -> None:
        dg: int = self.__current_generation - self.__last_generation_checkpoint
        if dg >= self.__generation_interval:
            CheckPointerReporter.save_checkpoint(self.__output_filepath, self.__current_generation, config, population, species_set)
            self.__last_generation_checkpoint = self.__current_generation

    
    @staticmethod
    def save_checkpoint(population_fpath: str, generation: int, config: neat.Config, population: typing.Dict[int, neat.genome.DefaultGenome], species_set: neat.species.DefaultSpeciesSet):
        with gzip.open(population_fpath, 'wb', compresslevel=5) as output_file:
            data = (generation, config, population, species_set)
            pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def restore_checkpoint(population_fpath: str) -> neat.population.Population:
        generation: int
        config: neat.Config
        population: typing.Dict[int, neat.genome.DefaultGenome]
        species_set: neat.species.DefaultSpeciesSet
        with gzip.open(population_fpath) as population_file:
            generation, config, population, species_set = pickle.load(population_file)
            return neat.population.Population(config, (population, species_set, generation))
        

    @staticmethod
    def save_population(population_fpath: str, population: neat.Population) -> None:
        with gzip.open(population_fpath, 'wb', compresslevel=5) as output_file:
            data = (population.generation, population.config, population.population, population.species)
            pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_population(population_fpath: str) -> neat.population.Population:
        generation: int
        config: neat.config.Config
        population: typing.Dict[int, neat.genome.DefaultGenome]
        species_set: neat.species.DefaultSpeciesSet
        with gzip.open(population_fpath) as population_file:
            generation, config, population, species_set = pickle.load(population_file)
            return neat.population.Population(config, (population, species_set, generation))