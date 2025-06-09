import typing
import graphviz
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot
import neat.config
import neat.genes
import numpy
import neat.genome
import pydot
from . import statistics


def plot_stats(statistics: statistics.StatisticsReporter, ylog=False) -> None:
    generation: typing.List[int] = range(len(statistics.most_fit_genomes))
    best_fitness: typing.List[float] = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness: numpy.ndarray = numpy.array(statistics.get_fitness_mean())
    stdev_fitness: numpy.ndarray = numpy.array(statistics.get_fitness_stdev())

    matplotlib.pyplot.plot(generation, avg_fitness, 'b-', label="average")
    matplotlib.pyplot.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    matplotlib.pyplot.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    matplotlib.pyplot.plot(generation, best_fitness, 'r-', label="best")

    matplotlib.pyplot.title("Population's average and best fitness")
    matplotlib.pyplot.xlabel("Generations")
    matplotlib.pyplot.ylabel("Fitness")
    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend(loc="best")

    if ylog:
        matplotlib.pyplot.gca().set_yscale('symlog')

    matplotlib.pyplot.show(block=False)


def plot_spikes(spikes) -> None:
    """ Plots the trains for a single spiking neuron. """
    t_values: typing.List[int] = [t for t, I, v, u, f in spikes]
    v_values: typing.List[int] = [v for t, I, v, u, f in spikes]
    u_values: typing.List[int] = [u for t, I, v, u, f in spikes]
    I_values: typing.List[int] = [I for t, I, v, u, f in spikes]
    f_values: typing.List[int] = [f for t, I, v, u, f in spikes]

    fig: matplotlib.figure.Figure = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(4, 1, 1)
    matplotlib.pyplot.ylabel("Potential (mv)")
    matplotlib.pyplot.xlabel("Time (in ms)")
    matplotlib.pyplot.grid()
    matplotlib.pyplot.plot(t_values, v_values, "g-")

    matplotlib.pyplot.title("Izhikevich's spiking neuron model")

    matplotlib.pyplot.subplot(4, 1, 2)
    matplotlib.pyplot.ylabel("Fired")
    matplotlib.pyplot.xlabel("Time (in ms)")
    matplotlib.pyplot.grid()
    matplotlib.pyplot.plot(t_values, f_values, "r-")

    matplotlib.pyplot.subplot(4, 1, 3)
    matplotlib.pyplot.ylabel("Recovery (u)")
    matplotlib.pyplot.xlabel("Time (in ms)")
    matplotlib.pyplot.grid()
    matplotlib.pyplot.plot(t_values, u_values, "r-")

    matplotlib.pyplot.subplot(4, 1, 4)
    matplotlib.pyplot.ylabel("Current (I)")
    matplotlib.pyplot.xlabel("Time (in ms)")
    matplotlib.pyplot.grid()
    matplotlib.pyplot.plot(t_values, I_values, "r-o")

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def plot_species(statistics: statistics.StatisticsReporter) -> None:
    """ Visualizes speciation throughout evolution. """
    
    species_sizes: typing.List[typing.List[int]] = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = numpy.array(species_sizes).T

    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    fig, ax = matplotlib.pyplot.subplots()
    
    ax.stackplot(range(num_generations), *curves)

    matplotlib.pyplot.title("Speciation")
    matplotlib.pyplot.ylabel("Size per Species")
    matplotlib.pyplot.xlabel("Generations")

    matplotlib.pyplot.show(block=False)


def draw_net(config: neat.config.Config, genome: neat.genome.DefaultGenome, node_names: typing.Optional[typing.Dict[int, str]]=None, show_disabled: bool=True, prune_unused: bool=False, node_colors: typing.Optional[typing.Dict[int, str]]=None):
    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert isinstance(node_names, typing.Mapping)

    if node_colors is None:
        node_colors = {}

    assert isinstance(node_colors, typing.Mapping)

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }

    graph: pydot.Dot = pydot.Dot()
    #graph: graphviz.Digraph = graphviz.Digraph(format='svg', node_attr=node_attrs)

    inputs = set()
    for k in typing.cast(neat.genome.DefaultGenomeConfig, config.genome_config).input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        graph.add_node(pydot.Node(name, attr={'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}))
        #input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        #graph.node(name, _attributes=input_attrs)

    outputs = set()
    for k in typing.cast(neat.genome.DefaultGenomeConfig, config.genome_config).output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        graph.add_node(pydot.Node(name, attr={'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}))
        #node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        #graph.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        graph.add_node(pydot.Node(str(n), attr={'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}))
        #attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        #graph.node(str(n), _attributes=attrs)

    for cg in typing.cast(typing.Dict[int, neat.genes.BaseGene], genome.connections).values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            graph.add_edge(pydot.Edge(a, b, attr={'style': style, 'color': color, 'penwidth': width}))
            #graph.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    rendered = graph.create()
    #rendered: str = graph.render(view=True, cleanup=False)

    print(rendered)