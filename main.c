#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

// Config
const int POPULATION_SIZE = 30;
const int MAX_NUM_OF_GENERATIONS = 3;

const int TOURNAMENT_SELECTION_SIZE = 2; // > 1
const int CROSSOVER_PROBABILITY = 80; // 1..100
const int MUTATION_PROBABILITY = 1; // 1..100

const int NUM_OF_TOURS = 5; // >= 1
const int NUM_OF_PLAYERS = 7; // > NUM_OF_TOURS
const int NUM_OF_PLAYGROUNDS = 8; // >= NUM_OF_PLAYERS

const int NUM_OF_SHUFFLE_ITERATIONS_MULTIPLIER = 2;

const bool RANDOM_SEED = true;
const unsigned int SEED_INIT = 241441;

#define USE_OPENMP

// ____________________________________________________

unsigned int SEED;
int* population;
int* speciesFitness; // 0 - minNumOfOpponents, 1 - minNumOfPlaygrounds

const int NO_VALUE_PLAYGROUND = -1;

const int halfOfPopulationSize = POPULATION_SIZE / 2;
const int halfOfNumOfTours = NUM_OF_TOURS / 2;
const int halfOfNumOfPlayers = NUM_OF_PLAYERS / 2;
const int numOfToursMinus1 = NUM_OF_TOURS - 1;

#define populationIndex(i, j, k) ((NUM_OF_TOURS) * (NUM_OF_PLAYERS) * (i) + (NUM_OF_PLAYERS) * (j) + (k))
#define population(i, j, k) (population[populationIndex(i, j, k)])

#define speciesFitness(i, j) (speciesFitness[((i) * 2 + (j))])

void printSpecies(int speciesIndex, bool printFitness);
void printPopulation(bool printFitness);
void printFitness();
void printArray(const int* array, int size);
void printGenerationInfo(int generationNum, int bestSpeciesIndex);

void swap(int* array, const int i, const int j) {
    const int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

void tournamentSwap(
    const int speciesIndex1,
    const int tournamentIndex1,
    const int speciesIndex2,
    const int tournamentIndex2
) {
    for (int i = 0; i < NUM_OF_PLAYERS; ++i) {
        swap(
            population,
            populationIndex(speciesIndex1, tournamentIndex1, i),
            populationIndex(speciesIndex2, tournamentIndex2, i)
        );
    }
}

void shuffle(int* array, const int size, unsigned int* seed) {
    for (int i = 0; i < size * NUM_OF_SHUFFLE_ITERATIONS_MULTIPLIER; ++i) {
        swap(
            array,
            rand_r(seed) % size,
            rand_r(seed) % size
        );
    }
}

void initPopulation() {
    int* playgroundNumbers = malloc(NUM_OF_PLAYGROUNDS * sizeof(int));

    for (int i = 0; i < NUM_OF_PLAYGROUNDS; ++i) {
        playgroundNumbers[i] = i;
    }

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        int* localPlaygroundNumbers = malloc(NUM_OF_PLAYGROUNDS * sizeof(int));
        memcpy(localPlaygroundNumbers, playgroundNumbers, NUM_OF_PLAYGROUNDS * sizeof(int));

        int* playgroundNumbersIndexesToSelect = malloc(NUM_OF_PLAYERS * sizeof(int));
        unsigned int seed = (unsigned int)time(NULL) + (unsigned int)omp_get_thread_num();

        for (int j = 0; j < NUM_OF_TOURS; ++j) {
            shuffle(localPlaygroundNumbers, NUM_OF_PLAYGROUNDS, &seed);
            for (int k = 0; k < halfOfNumOfPlayers; ++k) {
                playgroundNumbersIndexesToSelect[k] = localPlaygroundNumbers[k];
            }
            for (int k = 0; k < halfOfNumOfPlayers; ++k) {
                playgroundNumbersIndexesToSelect[k + halfOfNumOfPlayers] = localPlaygroundNumbers[k];
            }
            if (NUM_OF_PLAYERS % 2 == 1) {
                playgroundNumbersIndexesToSelect[NUM_OF_PLAYERS - 1] = NO_VALUE_PLAYGROUND;
            }
            shuffle(playgroundNumbersIndexesToSelect, NUM_OF_PLAYERS, &seed);
            for (int k = 0; k < NUM_OF_PLAYERS; ++k) {
                population(i, j, k) = playgroundNumbersIndexesToSelect[k];
            }
        }
        free(playgroundNumbersIndexesToSelect);
        free(localPlaygroundNumbers);
    }
    free(playgroundNumbers);
}

bool isSpeciesBetterThenOther(const int speciesIndex, const int otherSpeciesIndex) {
    const int speciesMinNumOfOpponents = speciesFitness(speciesIndex, 0);
    const int otherSpeciesMinNumOfOpponents = speciesFitness(otherSpeciesIndex, 0);
    if (speciesMinNumOfOpponents == otherSpeciesMinNumOfOpponents) {
        return speciesFitness(speciesIndex, 1) > speciesFitness(otherSpeciesIndex, 1);
    }
    return speciesMinNumOfOpponents > otherSpeciesMinNumOfOpponents;
}

int getBestSpeciesIndex() {
    int bestSpeciesIndex = 0;
    for (int i = 1; i < POPULATION_SIZE; ++i) {
        if (isSpeciesBetterThenOther(i, bestSpeciesIndex)) {
            bestSpeciesIndex = i;
        }
    }
    return bestSpeciesIndex;
}

int getIndexOfBestSpeciesFromIndicesArray(const int* indicesArray, const int indicesArraySize) {
    int bestSpeciesIndex = indicesArray[0];
    for (int i = 1; i < indicesArraySize; ++i) {
        if (isSpeciesBetterThenOther(indicesArray[i], bestSpeciesIndex)) {
            bestSpeciesIndex = indicesArray[i];
        }
    }
    return bestSpeciesIndex;
}

void calculateMinNumOfOpponentsForSpecies(const int speciesIndex) {
    int opponentCountMin = -1;

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int player = 0; player < NUM_OF_PLAYERS; ++player) {
        bool *distinctOpponents = calloc(NUM_OF_PLAYERS, sizeof(bool));
        int opponentCount = 0;
        for (int tour = 0; tour < NUM_OF_TOURS; ++tour) {
            const int playerPlayground = population(speciesIndex, tour, player);
            if (playerPlayground == NO_VALUE_PLAYGROUND) {
                continue;
            }
            for (int otherPlayer = 0; otherPlayer < NUM_OF_PLAYERS; ++otherPlayer) {
                if (otherPlayer != player
                    && playerPlayground == population(speciesIndex, tour, otherPlayer)) {
                    if (!distinctOpponents[otherPlayer]) {
                        distinctOpponents[otherPlayer] = true;
                        ++opponentCount;
                    }
                }
            }
        }
        #ifdef USE_OPENMP
        #pragma omp critical
        #endif
        {
            if (opponentCountMin == -1 || opponentCount < opponentCountMin) {
                speciesFitness(speciesIndex, 0) = opponentCount;
                opponentCountMin = opponentCount;
            }
        }
        free(distinctOpponents);
    }
}

void calculateMinNumOfPlaygroundsForSpecies(const int speciesIndex) {
    int playgroundCountMin = -1;

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int player = 0; player < NUM_OF_PLAYERS; ++player) {
        bool *distinctPlaygrounds = calloc(NUM_OF_PLAYGROUNDS, sizeof(bool));
        int playgroundCount = 0;
        for (int tour = 0; tour < NUM_OF_TOURS; ++tour) {
            const int playerPlayground = population(speciesIndex, tour, player);
            if (playerPlayground == NO_VALUE_PLAYGROUND) {
                continue;
            }
            if (!distinctPlaygrounds[playerPlayground]) {
                distinctPlaygrounds[playerPlayground] = true;
                ++playgroundCount;
            }
        }
        #ifdef USE_OPENMP
        #pragma omp critical
        #endif
        {
            if (playgroundCountMin == -1 || playgroundCount < playgroundCountMin) {
                speciesFitness(speciesIndex, 1) = playgroundCount;
                playgroundCountMin = playgroundCount;
            }
        }
        free(distinctPlaygrounds);
    }
}


float calculateAvgFitnessParam(const int fitnessParamIndex) {
    float avg = 0;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        avg += speciesFitness(i, fitnessParamIndex);
    }
    avg /= (float)POPULATION_SIZE;
    return avg;
}

float calculateAvgMinNumOfOpponents() {
    return calculateAvgFitnessParam(0);
}

float calculateAvgMinNumOfPlaygrounds() {
    return calculateAvgFitnessParam(1);
}

void fitness() {
    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(1)
    #endif
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        calculateMinNumOfOpponentsForSpecies(i);
        calculateMinNumOfPlaygroundsForSpecies(i);
    }
}

void selection() {
    // Tournament selection
    int* tempPopulation = malloc(POPULATION_SIZE * NUM_OF_TOURS * NUM_OF_PLAYERS * sizeof(int));

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < POPULATION_SIZE; ++ i) {
        unsigned int seed = (unsigned int)SEED ^ (unsigned int)omp_get_thread_num();
        int* speciesIndicesForTournament = malloc(TOURNAMENT_SELECTION_SIZE * sizeof(int));
        for (int j = 0; j < TOURNAMENT_SELECTION_SIZE; ++j) {
            speciesIndicesForTournament[j] = rand_r(&seed) % POPULATION_SIZE;
        }
        const int tournamentBestSpeciesIndex = getIndexOfBestSpeciesFromIndicesArray(
            speciesIndicesForTournament,
            TOURNAMENT_SELECTION_SIZE
        );
        for (int j = 0; j < NUM_OF_TOURS; ++j) {
            for (int k = 0; k < NUM_OF_PLAYERS; ++k) {
                tempPopulation[populationIndex(i, j, k)] = population(tournamentBestSpeciesIndex, j, k);
            }
        }
        free(speciesIndicesForTournament);
    }
    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(3)
    #endif
    for (int i = 0; i < POPULATION_SIZE; ++ i) {
        for (int j = 0; j < NUM_OF_TOURS; ++j) {
            for (int k = 0; k < NUM_OF_PLAYERS; ++k) {
                population(i, j, k) = tempPopulation[populationIndex(i, j, k)];
            }
        }
    }
    free(tempPopulation);
}

void crossover() {
    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(1)
    #endif
    for (int i = 0; i < halfOfPopulationSize; ++i) {
        unsigned int seed = (unsigned int)SEED ^ (unsigned int)omp_get_thread_num();
        if (rand_r(&seed) % 100 >= CROSSOVER_PROBABILITY) {
            continue;
        }
        const int speciesSplitIndex = 1 + rand_r(&seed) % numOfToursMinus1;

        if (speciesSplitIndex < halfOfNumOfTours) {
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int j = 0; j < speciesSplitIndex; ++j) {
                tournamentSwap(
                    i,
                    j,
                    i + halfOfPopulationSize,
                    j
                );
            }
        } else {
            #ifdef USE_OPENMP
            #pragma omp parallel for
            #endif
            // we could swap last tours because we get in result population the same species
            for (int j = speciesSplitIndex; j < NUM_OF_TOURS; ++j) {
                tournamentSwap(
                    i,
                    j,
                    i + halfOfPopulationSize,
                    j
                );
            }
        }
    }
}

void mutation() {
    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(1)
    #endif
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (int j = 0; j < NUM_OF_TOURS; ++j) {
            unsigned int seed = (unsigned int)SEED ^ (unsigned int)omp_get_thread_num();
            for (int k = 0; k < NUM_OF_PLAYERS; ++k) {
                if (rand_r(&seed) % 100 < MUTATION_PROBABILITY) {
                    swap(
                        population,
                        populationIndex(i, j, k),
                        populationIndex(i, j, rand_r(&seed) % NUM_OF_PLAYERS) // random index
                    );
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int generationNum = 1;

    population = (int*)malloc(POPULATION_SIZE * NUM_OF_TOURS * NUM_OF_PLAYERS * sizeof(int));
    speciesFitness = (int*)malloc(POPULATION_SIZE * 2 * sizeof(int));

    if (RANDOM_SEED) {
        SEED = time(NULL);
    } else {
        SEED = SEED_INIT;
    }

    omp_set_nested(true);

    const double startTime = omp_get_wtime();

    initPopulation();

    fitness();

    while (generationNum <= MAX_NUM_OF_GENERATIONS) {
        fitness();
        selection();
        crossover();
        mutation();
        ++generationNum;
    }
    const double resultTime = omp_get_wtime() - startTime;

    printSpecies(getBestSpeciesIndex(), true);

    printf("Time: %fs", resultTime);

    free(population);
    free(speciesFitness);
    return 0;
}

void printSpecies(const int speciesIndex, const bool printFitness) {
    printf("Species[%d]\n", speciesIndex);
    for (int j = 0; j < NUM_OF_TOURS; ++j) {
        for (int k = 0; k < NUM_OF_PLAYERS; ++k) {
            if (k != 0) {
                printf(", ");
            }
            printf("%d", population(speciesIndex, j, k));
        }
        printf("\n");
    }
    if (printFitness) {
        printf("MinNumOfOpponents: %d\n", speciesFitness(speciesIndex, 0));
        printf("MinNumOfPlaygrounds: %d\n", speciesFitness(speciesIndex, 1));
    }
}

void printPopulation(const bool printFitness) {
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        printSpecies(i, printFitness);
        printf("\n");
    }
    if (printFitness) {
        printf("AvgMinNumOfOpponents: %f\n", calculateAvgMinNumOfOpponents());
        printf("AvgMinNumOfPlaygrounds: %f\n", calculateAvgMinNumOfPlaygrounds());
    }
    printf("\n");
}

void printFitness() {
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        printf("[%d]\n", i);
        printf("MinNumOfOpponents: %d\n", speciesFitness(i, 0));
        printf("MinNumOfPlaygrounds: %d\n", speciesFitness(i, 1));
    }
}

void printArray(const int* array, const int size) {
    for (int i = 0; i < size; ++i) {
        if (i != 0) {
            printf(", ");
        }
        printf("%d", array[i]);
    }
    printf("\n");
}

void printGenerationInfo(const int generationNum, const int bestSpeciesIndex) {
    printf("Generation: %d\n", generationNum);
    printf("AvgMinNumOfOpponents: %f\n", calculateAvgMinNumOfOpponents());
    printf("AvgMinNumOfPlaygrounds: %f\n", calculateAvgMinNumOfPlaygrounds());
    printf("Best ");
    printSpecies(bestSpeciesIndex, true);
    printf("\n");
}
