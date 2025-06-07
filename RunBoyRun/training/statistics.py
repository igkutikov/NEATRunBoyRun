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


    def get_fitness_stat(self, f):
        stat = []
        for stats in self.generation_statistics:
            scores = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(f(scores))

        return stat


    def get_fitness_mean(self):
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat(neat.math_util.mean)


    def get_fitness_stdev(self):
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(neat.math_util.stdev)


    def get_fitness_median(self):
        """Get the per-generation median fitness."""
        return self.get_fitness_stat(neat.math_util.median2)


    def get_average_cross_validation_fitness(self): # pragma: no cover
        """Get the per-generation average cross_validation fitness."""
        avg_cross_validation_fitness = []
        for stats in self.generation_cross_validation_statistics:
            scores = []
            for fitness in stats.values():
                scores.extend(fitness)
            avg_cross_validation_fitness.append(neat.math_util.mean(scores))

        return avg_cross_validation_fitness


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


# key is the genome id
TrainingGenomeStatisticsEntry=typing.Dict[int, TrainingTurnStatisticsEntry]
# each turn new entry
TrainingTurnGenomeStatisticsEntry = typing.List[TrainingGenomeStatisticsEntry]
# index is the generation number
TrainingGenerationTurnGenomeStatisticsEntry = typing.List[TrainingTurnGenomeStatisticsEntry]


class TrainingTrunStatsTracer:
    class JSONEncoder(json.JSONEncoder):
        def __init__(self, *args, skipkeys: bool=False, ensure_ascii: bool=True, check_circular: bool=True, allow_nan: bool=True, sort_keys: bool=False, indent: typing.Optional[int]=None, separators: typing.Optional[str]=None, default: typing.Callable=None) -> None:
            json.JSONEncoder.__init__(self, *args, skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular, allow_nan=allow_nan, sort_keys=sort_keys, indent=indent, separators=separators, default=default)

        def format_float(self, value: float) -> str:
            return format(value, '.6f')


        def floatstr(self, o: float, _inf=json.encoder.INFINITY, _neginf=-json.encoder.INFINITY):
                # Check for specials.  Note that this type of test is processor
                # and/or platform-specific, so do tests which don't depend on the
                # internals.

                if o != o:
                    text = 'NaN'
                elif o == _inf:
                    text = 'Infinity'
                elif o == _neginf:
                    text = '-Infinity'
                else:
                    return self.format_float(o)

                if not self.allow_nan:
                    raise ValueError("Out of range float values are not JSON compliant: " + repr(o))

                return text

        def iterencode(self, o, _one_shot=False):
            if self.check_circular:
                markers = {}
            else:
                markers = None
            if self.ensure_ascii:
                _encoder = json.encoder.encode_basestring_ascii
            else:
                _encoder = json.encoder.encode_basestring


            if (_one_shot and c_make_encoder is not None and self.indent is None):
                _iterencode = c_make_encoder(
                    markers, self.default, _encoder, self.indent,
                    self.key_separator, self.item_separator, self.sort_keys,
                    self.skipkeys, self.allow_nan)
            else:
                _iterencode = json.encoder._make_iterencode(
                    markers, self.default, _encoder, self.indent, self.floatstr,
                    self.key_separator, self.item_separator, self.sort_keys,
                    self.skipkeys, _one_shot)
            return _iterencode(o, 0)


    class JSONDecoder(json.JSONDecoder):
         def __init__(self, *args, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, strict=True, object_pairs_hook=None) -> None:
             json.JSONDecoder.__init__(self, *args, object_hook=object_hook, parse_float=parse_float, parse_int=parse_int, strict=strict, object_pairs_hook=object_pairs_hook)
    
    def __init__(self, output_file: str):
        self.__generation_statistics: TrainingGenerationTurnGenomeStatisticsEntry = []
        self.__deaths: typing.Dict[int, bool] = {}
        self.__output_file: io.BufferedWriter = open(output_file, 'wb')
        self.__json_writer = TrainingTrunStatsTracer.JSONEncoder(indent=1)

        self.__output_file.writelines(
            (
                b'{\n', # statrt of TrainingGenerationTurnGenomeStatisticsEntry
                b'\t"trace": [\n'
                b'\t]\n'
                b'}' # end of TrainingGenerationTurnGenomeStatisticsEntry
            )
        )

        self.__output_file.flush()

    def __del__(self) -> None:
        self.__output_file.close()


    def start_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config):
        self.__generation_statistics.append([])
        if generation > 0:
            self.__output_file.seek(-5, io.SEEK_END) # back ending of json and TrainingTurnGenomeStatisticsEntry and TrainingGenerationTurnGenomeStatisticsEntry
            self.__output_file.write(b',\n') # end of previous TrainingTurnGenomeStatisticsEntry
        else:            
            self.__output_file.seek(-4, io.SEEK_END) # back ending of json and TrainingGenerationTurnGenomeStatisticsEntry
        
        self.__output_file.writelines(
            (
                b'\t\t[\n', # start of TrainingTurnGenomeStatisticsEntry
                b'\t\t]\n', # end of TrainingTurnGenomeStatisticsEntry
                b'\t]\n' # end of TrainingGenerationTurnGenomeStatisticsEntry
                b'}' # end of json
            )
        )
        self.__output_file.flush()


    def end_generation(self, generation: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config):
        self.__deaths.clear()
        self.__output_file.seek(-8, io.SEEK_END) # back ending of json TrainingTurnGenomeStatisticsEntry and TrainingGenerationTurnGenomeStatisticsEntry
        self.__output_file.writelines(
            (
                b'\t\t]\n', # end of TrainingTurnGenomeStatisticsEntry
                b'\t]\n' # end of TrainingGenerationTurnGenomeStatisticsEntry
                b'}' # end of json
            )
        )
        self.__output_file.flush()


    def start_turn(self, trun: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, characters: typing.Collection[model.characters.NEATTraineeCharacter]):
        self.__generation_statistics[-1].append({})
        if trun > 0:
            self.__output_file.seek(-9, io.SEEK_END) # back ending of json and TrainingGenomeStatisticsEntry and TrainingTurnGenomeStatisticsEntry and TrainingGenerationTurnGenomeStatisticsEntry
            self.__output_file.write( b',\n') # end of previous TrainingGenomeStatisticsEntry
        else:
            self.__output_file.seek(-8, io.SEEK_END) # back ending of json and TrainingTurnGenomeStatisticsEntry and TrainingGenerationTurnGenomeStatisticsEntry
        
        self.__output_file.writelines(
            (
                b'\t\t\t{\n', # start TrainingGenomeStatisticsEntry
                b'\t\t\t}\n', # end TrainingGenomeStatisticsEntry
                b'\t\t]\n',  # end of TrainingTurnGenomeStatisticsEntry
                b'\t]\n' # end of TrainingGenerationTurnGenomeStatisticsEntry
                b'}' # end of TrainingGenerationTurnGenomeStatisticsEntry
            )
        )

        self.__output_file.flush()


    def end_turn(self, trun: int, genomes: typing.List[typing.Tuple[int, neat.genome.DefaultGenomeConfig]], config: neat.config.Config, characters: typing.Collection[model.characters.NEATTraineeCharacter]) -> None:
        turn_stats: TrainingTurnGenomeStatisticsEntry = self.__generation_statistics[-1]
        genomes_stats: TrainingGenomeStatisticsEntry = turn_stats[-1]
        for character in characters:
            if character.Dead:
                if self.__deaths.get(character.Id, False):
                    continue
                self.__deaths[character.Id] = True

            genomes_stats[character.GenomeId] = TrainingTurnStatisticsEntry(
                nSteps=character.nSteps,
                nLastPltafromStep=character.Evaluator.nLastPlatformStep,
                Fitness=character.Fitness,
                EvaluatedFitness=character.Evaluator.Fitness,
                Decision=character.Evaluator.Decision,
                DecisionActive=character.Evaluator.DecisionActive,
                DecisionKind=character.Evaluator.DecisionKind
            )

            if len(genomes_stats) > 1:
                self.__output_file.seek(-14, io.SEEK_END)  # back ending of TrainingGenerationTurnGenomeStatisticsEntry and ending TrainingTurnGenomeStatisticsEntry and ending TrainingGenomeStatisticsEntry
                self.__output_file.write(b',\n')
                self.__output_file.flush()
            else:
                self.__output_file.seek(-13, io.SEEK_END)  # back ending of TrainingGenerationTurnGenomeStatisticsEntry and ending TrainingTurnGenomeStatisticsEntry and ending TrainingGenomeStatisticsEntry

            json_str: str = self.__json_writer.encode(genomes_stats[character.GenomeId])
            json_str: typing.List[bytes] = [('\t\t\t\t' + line.replace(' ', '\t', 2) + '\n').encode() for line in json_str.splitlines()]
            json_str[0] = f'\t\t\t\t"{character.GenomeId}": '.encode() + json_str[0].lstrip()
            json_str.extend(
                (
                    b'\t\t\t}\n', # end TrainingGenomeStatisticsEntry
                    b'\t\t]\n',  # end of TrainingTurnGenomeStatisticsEntry
                    b'\t]\n' # end of TrainingGenerationTurnGenomeStatisticsEntry
                    b'}' # end of json
                )
            )
            self.__output_file.writelines(json_str)
            self.__output_file.flush()
    

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


