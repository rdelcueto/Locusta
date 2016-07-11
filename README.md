# Locusta

This is the main source code repository for [Locusta].
It contains the library code, and documentation.

## Building from Source

1. TODO

## Building Documentation

1. TODO

## Notes

A Massively Parallel Evolutionary Computation Metaheuristic Framework.

Locusta provides a framework to build various population based evolutionary metaheuristics.

The current implementation uses CUDA to describe massively parallel kernels, to
compute: the evaluation of the fitness function on a population of genomes, the
methaheuristic processes (ej. mutation, crossover) and the generation of
pseudorandom numbers.

The collection of metaheuristics also has a CPU parallel implementation, written
using OpenMP. Implementations are not design to perform invariant
transformation, between architectures. Given that each platform, implies a
different problem to solve in the data-oriented design, each implementation can
vary widely in implementation. The framework provides a way to compare and measure
the performance throughoutput of each implementation, taking into account the
strenghts of each targeted architecture.

| Metaheuristic Solver           | OMP |  CUDA |
|--------------------------------|-----|-------|
| Particle Swarm Optimization    | ✓   | ✓    |
| Genetic Algorithm              | ✓   | ✓    |
| Differential Evolution         | ✓   | ✓    |

## License

Locusta is distributed under the terms of the GNU Lesser General Public Licence

See [LICENSE-LGPL-3.0](LICENSE-LGPL-3.0)
