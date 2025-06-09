import typing 
import os
import io
import json
import csv
import copy
import model
import matplotlib.pyplot
import matplotlib.figure
import neat.config
import neat.genome
import neat.population
import neat.species
import neat.statistics
import neat.six_util
import neat.math_util

try:
    from json.encoder import c_make_encoder
except ImportError:
    c_make_encoder = None


class TrainingStatisticsEntry(typing.TypedDict):
    generation: int
    steps: int
    fitness: float
    average_fitness: float
    species_amount: int
    nodes_amount: int
    connections_amount: int


class StatisticsReporter(neat.statistics.BaseReporter):
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genome/species fitness and species sizes.
    """
    def __init__(self) -> None:
        neat.statistics.BaseReporter.__init__(self)
        self.most_fit_genomes: typing.List[neat.genome.DefaultGenome] = []
        self.generation_statistics: typing.List[typing.Dict[int, float]] = []
        self.genomes_steps: typing.Dict[int, typing.Dict[int, int]] = {}
        self.generations: int = 0


    def start_generation(self, generation: int):
        self.generations = generation
        self.genomes_steps[generation] = {}


    def end_generation(self, config: neat.config.Config, population: neat.population.Population, species_set: neat.DefaultSpeciesSet) -> None:
        pass


    def post_evaluate(self, config: neat.config.Config, population: neat.population.Population, species_set: neat.DefaultSpeciesSet, best_genome: neat.genome.DefaultGenome) -> None:
        self.most_fit_genomes.append(copy.deepcopy(best_genome))

        # Store the fitnesses of the members of each currently active species.
        species_stats: typing.Dict[int, float] = {}

        sid: int
        specie: neat.species.Species
        for sid, specie in neat.six_util.iteritems(species_set.species):
            species_stats[sid] = dict((k, v.fitness) for k, v in neat.six_util.iteritems(typing.cast(typing.Dict[int, neat.genome.DefaultGenome], specie.members)))
        self.generation_statistics.append(species_stats)


    def post_step(self, config: neat.config.Config, population: neat.population.Population, steps: typing.Dict[int, int]) -> None:
        self.genomes_steps[self.generations] = dict(steps)


    def get_steps(self, genome_id: int) -> typing.List[int]:
        null_steps: typing.Dict[int, int] = {}
        steps: typing.List[int] = []

        for generation in range(self.generations + 1):
            generations_steps: typing.Dict[int, int] = self.genomes_steps.get(generation, null_steps)
            step: int = generations_steps.get(genome_id, 0)
            steps.append(step)

        return steps


    def get_fitness_stat(self, f: typing.Callable[[typing.Iterable[float]], float]) -> typing.List[float]:
        stat = []
        for stats in self.generation_statistics:
            scores = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(f(scores))

        return stat


    def get_fitness_mean(self) -> typing.List[float]:
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat(neat.math_util.mean)


    def get_fitness_stdev(self) -> typing.List[float]:
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(neat.math_util.stdev)


    def get_fitness_median(self) -> typing.List[float]:
        """Get the per-generation median fitness."""
        return self.get_fitness_stat(neat.math_util.median2)


    def best_unique_genomes(self, n: int) -> typing.List[neat.genome.DefaultGenome]:
        """Returns the most n fit genomes, with no duplication."""
        best_unique: typing.Dict[int, neat.genome.DefaultGenome] = {}
        g: neat.genome.DefaultGenome
        for g in self.most_fit_genomes:
            best_unique[g.key] = g
        best_unique_list: typing.List[int, neat.genome.DefaultGenome] = list(best_unique.values())

        return sorted(best_unique_list, key=lambda genome: genome.fitness, reverse=True)[:n]


    def best_genomes(self, n: int) -> typing.List[neat.genome.DefaultGenome]:
        """Returns the n most fit genomes ever seen."""
        return sorted(self.most_fit_genomes, key=lambda genome: genome.fitness, reverse=True)[:n]


    def best_genome(self) -> neat.genome.DefaultGenome:
        """Returns the most fit genome ever seen."""
        return self.best_genomes(1)[0]


    def get_species_sizes(self) -> typing.List[typing.List[int]]:
        all_species: typing.Set[int] = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species: int = max(all_species)
        species_counts: typing.List[typing.List[int]] = []
        for gen_data in self.generation_statistics:
            species: typing.List[int] = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
            species_counts.append(species)

        return species_counts


    def get_species_fitness(self, null_value: str='') -> typing.List[typing.List[float]]:
        all_species: typing.Set[int] = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species: int = max(all_species)
        species_fitness: typing.List[typing.List[float]] = []
        for gen_data in self.generation_statistics:
            member_fitness: typing.List[float] = [gen_data.get(sid, []) for sid in range(1, max_species + 1)]
            fitness: typing.List[float] = []
            for mf in member_fitness:
                if mf:
                    fitness.append(neat.math_util.mean(mf))
                else:
                    fitness.append(null_value)
            species_fitness.append(fitness)

        return species_fitness


class TrainingTurnStatisticsEntry(typing.TypedDict):
    nSteps: int
    nLastPltafromStep: int
    Fitness: float
    EvaluatedFitness: float
    Decision: float
    DecisionActive: bool
    DecisionKind: int


class TrainingGenerationStatisticsEntry(typing.NamedTuple):
    turns: typing.List[typing.Dict[int, TrainingTurnStatisticsEntry]]
    fittest: typing.List[neat.genome.DefaultGenome]


class TrainingTrunStatsTracer:
    class JSONEncoder:
        def __init__(self, indent: typing.Optional[typing.Union[int, str]]=None) -> None:
            self.__indent: str = ''
            if indent:
                if isinstance(indent, str):
                    self.__indent += indent
                elif isinstance(indent, int):
                    self.__indent += ' ' * indent


        def encode(self, value: typing.Any, initial_indent: typing.Optional[typing.Union[int, str]]=None) -> str:
            if initial_indent:
                if isinstance(initial_indent, int):
                    initial_indent += ' ' * initial_indent
            else:
                initial_indent = ''

            stream: io.StringIO = io.StringIO()
            if isinstance(value, typing.Mapping):
                stream.write(initial_indent)
                stream.write('{')
                if initial_indent:
                    stream.write('\n')
                self.__encode_mapping(stream, initial_indent + self.__indent, typing.cast(typing.Mapping[typing.Any, typing.Any], value))
                if initial_indent:
                    stream.write('\n')
                stream.write(initial_indent)
                stream.write('}')
            elif isinstance(value, typing.Sequence):
                stream.write(initial_indent)
                stream.write('[')
                if initial_indent:
                    stream.write('\n')
                self.__encode_list(stream, initial_indent + self.__indent, typing.cast(typing.Sequence[typing.Any], value))
                if initial_indent:
                    stream.write('\n')
                stream.write(initial_indent)
                stream.write(']')
            else:
                raise ValueError(f'unsupported json type {type(value)}')

            return stream.getvalue()


        def __encode_scalar(self, stream: io.StringIO, value: typing.Optional[typing.Union[str, float, int, bool]]) -> None:
            if value is None:
                stream.write('null')
            elif value is True:
                stream.write('true')
            elif value is False:
                stream.write('false')
            elif isinstance(value, str):
                stream.write('"')
                stream.write(typing.cast(str, value))
                stream.write('"')
            elif isinstance(value, int):
                stream.write(str(typing.cast(int, value)))
            elif isinstance(value, float):
                stream.write(format(typing.cast(float, value), '.6f'))
            else:
                raise ValueError(f'unsupported json type {type(value)}')


        def __encode_list(self, stream: io.StringIO, indent: str, value: typing.Sequence[typing.Any]) -> None:
            for entry in value:
                if isinstance(entry, typing.Mapping):
                    stream.write(indent)
                    stream.write('{')
                    if indent:
                        stream.write('\n')
                    self.__encode_mapping(stream, indent + self.__indent, typing.cast(typing.Mapping[typing.Any, typing.Any], entry))
                    if indent:
                        stream.write('\n')
                    stream.write(indent)
                    stream.write('}')
                elif isinstance(entry, typing.Sequence):
                    stream.write(indent)
                    stream.write('[')
                    if indent:
                        stream.write('\n')
                    self.__encode_list(stream, indent + self.__indent, typing.cast(typing.Sequence[typing.Any], entry))
                    if indent:
                        stream.write('\n')
                    stream.write(indent)
                    stream.write(']')
                else:
                    stream.write(indent)
                    self.__encode_scalar(stream, entry)

                stream.write(',')
                if indent:
                    stream.write('\n')

            if len(value) > 0:
                if indent:
                    stream.seek(stream.tell() - 2, io.SEEK_SET)
                else:
                    stream.seek(stream.tell() - 1, io.SEEK_SET)


        def __encode_key(self, key: typing.Optional[typing.Union[str, float, int, bool]]) -> str:
            if key is None:
                return '"null"'

            if key is True:
                return "true"

            if key is False:
                return '"false"'

            if isinstance(key, str):
                return '"' + key + '"'

            if isinstance(key, int):
                return f'"{key}"'

            if isinstance(key, float):
                return f'"{key:.6f}"'

            raise ValueError(f'unsupported json key type {type(value)}')


        def __encode_mapping(self, stream: io.StringIO, indent: str, value: typing.Dict[typing.Any, typing.Any]) -> None:
            for key in value:
                stream.write(indent)
                stream.write(self.__encode_key(key))
                stream.write(': ')

                val: typing.Any = value[key]
                if isinstance(val, typing.Mapping):
                    stream.write('{')
                    if indent:
                        stream.write('\n')
                    self.__encode_mapping(stream, indent + self.__indent, typing.cast(typing.Mapping[typing.Any, typing.Any], val))
                    if indent:
                        stream.write('\n')
                    stream.write(indent)
                    stream.write('}')
                elif isinstance(val, typing.Sequence):
                    stream.write('[')
                    if indent:
                        stream.write('\n')
                    self.__encode_list(stream, indent + self.__indent, typing.cast(typing.Sequence[typing.Any], val))
                    if indent:
                        stream.write('\n')
                    stream.write(indent)
                    stream.write(']')
                else:
                    self.__encode_scalar(stream, val)

                stream.write(',')
                if indent:
                    stream.write('\n')

            if len(value) > 0:
                if indent:
                    stream.seek(stream.tell() - 2, io.SEEK_SET)
                else:
                    stream.seek(stream.tell() - 1, io.SEEK_SET)


    def __init__(self, output_file: str):
        self.__generation_statistics: typing.List[TrainingGenerationStatisticsEntry] = []

        self.__deaths: typing.Dict[int, bool] = {}
        self.__output_file: io.BufferedWriter = open(output_file, 'wb')
        self.__json_writer = TrainingTrunStatsTracer.JSONEncoder(indent='\t')


    def __del__(self) -> None:
        self.__output_file.close()


    def start_training(self, nGenerations: int, config: neat.config.Config, population: neat.population.Population) -> None:
        self.__output_file.write(b'{\n\t"trace": [\n\t]\n}')
        self.__output_file.flush()


    def start_generation(self, generation: int, config: neat.config.Config, population: neat.population.Population, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]]):
        self.__generation_statistics.append(TrainingGenerationStatisticsEntry(turns=[],fittest=[]))


    def start_turn(self, trun: int, config: neat.config.Config, population: neat.population.Population, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], characters: typing.Collection[model.characters.INEATTraineeCharacter]) -> None:
        self.__generation_statistics[-1].turns.append({})


    def end_turn(self, trun: int, config: neat.config.Config, population: neat.population.Population, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], characters: typing.Collection[model.characters.INEATTraineeCharacter]) -> None:
        genomes_stats: typing.Dict[int, TrainingTurnStatisticsEntry] = self.__generation_statistics[-1].turns[-1]
        for character in characters:
            if character.Dead:
                if self.__deaths.get(character.GenomeId, False):
                    continue
                self.__deaths[character.GenomeId] = True

            genomes_stats[character.GenomeId] = TrainingTurnStatisticsEntry(
                nSteps=character.nSteps,
                nLastPltafromStep=character.Evaluator.nLastPlatformStep,
                Fitness=character.Fitness,
                EvaluatedFitness=character.Evaluator.Fitness,
                Decision=character.Evaluator.Decision,
                DecisionActive=character.Evaluator.DecisionActive,
                DecisionKind=character.Evaluator.DecisionKind
            )


    def end_generation(self, generation: int, config: neat.config.Config, population: neat.population.Population, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]]):
        best_fitness: float = 0.0
        best_generation_genomes: typing.List[neat.genome.DefaultGenome] = []
        for genome in typing.cast(typing.Iterable[neat.genome.DefaultGenome], neat.six_util.itervalues(population.population)):
            if genome.fitness >= best_fitness:
                best_fitness = genome.fitness

        if best_fitness > 0.0:
            for genome in typing.cast(typing.Iterable[neat.genome.DefaultGenome], neat.six_util.itervalues(population.population)):
                if genome.fitness == best_fitness:
                    best_generation_genomes.append(copy.deepcopy(genome))

        self.__generation_statistics[-1].fittest.append(best_generation_genomes)

        if len(self.__generation_statistics) > 1:
            self.__output_file.seek(-5, io.SEEK_END)  # back ending of TrainingGenerationTurnGenomeStatisticsEntry and ending TrainingTurnGenomeStatisticsEntry and ending TrainingGenomeStatisticsEntry
            self.__output_file.write(b',\n')
        else:
            self.__output_file.seek(-4, io.SEEK_END)  # back ending of TrainingGenerationTurnGenomeStatisticsEntry and ending TrainingTurnGenomeStatisticsEntry and ending TrainingGenomeStatisticsEntry

        self.__output_file.write(
            self.__json_writer.encode(
                {
                    'fittest': dict((genome.key, genome.fitness) for genome in best_generation_genomes),
                    'turns': self.__generation_statistics[-1].turns
                },
                initial_indent='\t\t'
            ).encode()
        )

        self.__output_file.write(b'\n\t]\n}')
        self.__output_file.flush()

        # sid: int
        # specie: neat.species.Species
        # for sid, specie in neat.six_util.iteritems(typing.cast(neat.species.DefaultSpeciesSet ,population.species).species):
        #     for id, genome in neat.six_util.iteritems(typing.cast(typing.Dict[int, neat.genome.DefaultGenome], specie.members)):
        #         print(f'genome id: {id} =? {genome.key}')
        #         for g in best_generation_genomes:
        #             if genome.key == g.key:
        #                 print(f'genome({genome.key}) => {genome.fitness} =? {g.fitness}')


    def end_training(self, nGenerations: int, config: neat.config.Config, population: neat.population.Population) -> None:
        pass


def generate_statistics(reporter: StatisticsReporter) -> typing.List[TrainingStatisticsEntry]:
    return [
        TrainingStatisticsEntry(
            generation = generation,
            fitness = typing.cast(float, best_genome.fitness),
            steps = reporter.get_steps(best_genome.key)[generation],
            average_fitness = typing.cast(float, reporter.get_fitness_mean()[generation]),
            species_amount = len(reporter.get_species_sizes()[generation]),
            nodes_amount = len(best_genome.nodes),
            connections_amount = len(best_genome.connections))    
        for generation, best_genome in enumerate(reporter.most_fit_genomes)
    ]


def save_statistics_json(output_file: str, statistics: typing.List[TrainingStatisticsEntry]) -> None:            
    with open(output_file, "w") as jsonf:
        json.dump(statistics, jsonf, indent=4)


def load_statistics_json(input_file: str) -> typing.List[TrainingStatisticsEntry]:
    with open(input_file, "r") as jsonf:
        statistics: typing.List[TrainingStatisticsEntry] = json.load(jsonf)
    return statistics


def save_statistics_csv(output_file: str, statistics: typing.List[TrainingStatisticsEntry]) -> None:    
    with open(output_file, "w", newline='') as csvf:
        writer: csv.DictWriter = csv.DictWriter(csvf, typing.get_type_hints(TrainingStatisticsEntry).keys(), delimiter=',', lineterminator=os.linesep)
        writer.writeheader()
        writer.writerows(statistics)


def load_statistics_csv(input_file: str) -> typing.List[TrainingStatisticsEntry]:
    statistics: typing.List[TrainingStatisticsEntry] = []
    with open(input_file, "r", newline='') as csvf:
        reader: csv.DictReader = csv.DictReader(csvf, typing.get_type_hints(TrainingStatisticsEntry).keys(), delimiter=',', lineterminator=os.linesep)
        next(reader, None)
        for row in reader:
            statistics.append(TrainingStatisticsEntry(row))
    return statistics


def plot_best_fitness(stats: typing.List[TrainingStatisticsEntry]) -> None:    
    data: typing.List[typing.Tuple[int, float, int]] = [(entry.get('generation'), entry.get('fitness'), entry.get('steps')) for entry in stats]
    data.sort(key = lambda datum: datum[0])

    data: typing.Tuple[typing.List[int], typing.List[float], typing.List[float]] = (
        [datum[0] for datum in data],
        [datum[1] for datum in data],
        [datum[2] for datum in data],
    )
    

    figure: matplotlib.figure.Figure = matplotlib.pyplot.figure(1, figsize=(10, 6))
    matplotlib.pyplot.plot(data[0], data[1], marker='o', linestyle='-', color='b')
    matplotlib.pyplot.plot(data[0], data[2], marker='o', linestyle='-', color='g')
    matplotlib.pyplot.title("Best Genome Fitness Over Generations")
    matplotlib.pyplot.xlabel("Generation")
    matplotlib.pyplot.ylabel("Best Fitness")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show(block=False)
    figure.canvas.draw()


def plot_average_fitness(stats: typing.List[TrainingStatisticsEntry]) -> None:
    data: typing.List[typing.Tuple[int, float]] = [(entry.get('generation'), entry.get('average_fitness')) for entry in stats]
    data.sort(key = lambda datum: datum[0])
    
    data: typing.Tuple[typing.List[int], typing.List[float]] = (
        [datum[0] for datum in data],
        [datum[1] for datum in data],
    )

    figure: matplotlib.figure.Figure = matplotlib.pyplot.figure(2, figsize=(10, 6))
    matplotlib.pyplot.plot(data[0], data[1], marker='o', linestyle='-', color='b')
    matplotlib.pyplot.title("Average Fitness Over Generations")
    matplotlib.pyplot.xlabel("Generation")
    matplotlib.pyplot.ylabel("Avg. Fitness")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show(block=False)
    figure.canvas.draw()


