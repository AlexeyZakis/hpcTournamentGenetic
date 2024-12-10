#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

// Config
const int POPULATION_SIZE = 100;
const int MAX_NUM_OF_GENERATIONS = 20;

const int TOURNAMENT_SELECTION_SIZE = 4; // > 1
const int CROSSOVER_PROBABILITY = 80; // 1..100
const int MUTATION_PROBABILITY = 1; // 1..100

const int NUM_OF_SHUFFLE_ITERATIONS_MULTIPLIER = 2;

const bool RANDOM_SEED = true;
const unsigned int SEED_INIT = 241441;

#define NUM_OF_EXPERIMENTS 3
#define NUM_OF_EXPERIMENT_PARAMS 3

int experiments[NUM_OF_EXPERIMENTS][NUM_OF_EXPERIMENT_PARAMS] = {
    // NUM_OF_TOURS (>= 1), NUM_OF_PLAYERS (> NUM_OF_TOURS), NUM_OF_PLAYGROUNDS (>= NUM_OF_PLAYERS)
    {3, 4, 5},
    {10, 20, 30},
    {40, 60, 80},
    // {200, 400, 500},
    // {500, 800, 1000},
    // {1000, 3000, 5000},
};

#define THREADS_NUM_LIST_SIZE 4
int threadsNumList[THREADS_NUM_LIST_SIZE] = { 1, 2, 4, 8, 16, 32 };

#define USE_OPENMP

char* RESULT_FILE_NAME = "result.txt";

// ____________________________________________________

unsigned int SEED;
int* population;
int* speciesFitness; // 0 - minNumOfOpponents, 1 - minNumOfPlaygrounds

const int NO_VALUE_PLAYGROUND = -1;

const int halfOfPopulationSize = POPULATION_SIZE / 2;

int NUM_OF_TOURS;
int NUM_OF_PLAYERS;
int NUM_OF_PLAYGROUNDS;

int halfOfNumOfTours;
int halfOfNumOfPlayers;
int numOfToursMinus1;

#define populationIndex(i, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS) ((NUM_OF_TOURS) * (NUM_OF_PLAYERS) * (i) + (NUM_OF_PLAYERS) * (j) + (k))
#define population(i, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS) (population[(populationIndex((i), (j), (k), (NUM_OF_TOURS), (NUM_OF_PLAYERS)))])

#define speciesFitness(i, j) (speciesFitness[((i) * 2 + (j))])

void printSpecies(int speciesIndex, bool printFitness);
void printPopulation(bool printFitness);
void printFitness();
void printArray(const int* array, int size);
void printGenerationInfo(int generationNum, int bestSpeciesIndex);
void printExperimentStart(int index, int numOfThreads);
void printExperimentResult(double resultTime, int numOfThreads);
void writeResultToFile(double resultTime, int numOfThreads);
void writeResultFileHeaders();

void initConstants(const int experiment[NUM_OF_EXPERIMENT_PARAMS]) {
    NUM_OF_TOURS = experiment[0];
    NUM_OF_PLAYERS = experiment[1];
    NUM_OF_PLAYGROUNDS = experiment[2];

    halfOfNumOfTours = NUM_OF_TOURS / 2;
    halfOfNumOfPlayers = NUM_OF_PLAYERS / 2;
    numOfToursMinus1 = NUM_OF_TOURS - 1;
}

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
            populationIndex(speciesIndex1, tournamentIndex1, i, NUM_OF_TOURS, NUM_OF_PLAYERS),
            populationIndex(speciesIndex2, tournamentIndex2, i, NUM_OF_TOURS, NUM_OF_PLAYERS)
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
                population(i, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS) = playgroundNumbersIndexesToSelect[k];
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
            const int playerPlayground = population(speciesIndex, tour, player, NUM_OF_TOURS, NUM_OF_PLAYERS);
            if (playerPlayground == NO_VALUE_PLAYGROUND) {
                continue;
            }
            for (int otherPlayer = 0; otherPlayer < NUM_OF_PLAYERS; ++otherPlayer) {
                if (otherPlayer != player
                    && playerPlayground == population(speciesIndex, tour, otherPlayer, NUM_OF_TOURS, NUM_OF_PLAYERS)) {
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
            const int playerPlayground = population(speciesIndex, tour, player, NUM_OF_TOURS, NUM_OF_PLAYERS);
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
        unsigned int seed = SEED ^ (unsigned int)omp_get_thread_num();
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
                tempPopulation[populationIndex(i, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS)] = population(tournamentBestSpeciesIndex, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS);
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
                population(i, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS) = tempPopulation[populationIndex(i, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS)];
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
        unsigned int seed = SEED ^ (unsigned int)omp_get_thread_num();
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
                        populationIndex(i, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS),
                        populationIndex(i, j, rand_r(&seed) % NUM_OF_PLAYERS, NUM_OF_TOURS, NUM_OF_PLAYERS) // random index
                    );
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    omp_set_nested(true);

    if (RANDOM_SEED) {
        SEED = time(NULL);
    } else {
        SEED = SEED_INIT;
    }

    FILE* resultFile = fopen(RESULT_FILE_NAME, "w");
    fclose(resultFile);

    writeResultFileHeaders();

    for (int i = 0; i < THREADS_NUM_LIST_SIZE; ++i) {
        omp_set_num_threads(threadsNumList[i]);
        for (int j = 0; j < NUM_OF_EXPERIMENTS; ++j) {
            initConstants(experiments[j]);

            printExperimentStart(j, threadsNumList[i]);

            int generationNum = 1;

            population = (int*)malloc(POPULATION_SIZE * NUM_OF_TOURS * NUM_OF_PLAYERS * sizeof(int));
            speciesFitness = (int*)malloc(POPULATION_SIZE * 2 * sizeof(int));

            const double startTime = omp_get_wtime();

            initPopulation();

            while (generationNum <= MAX_NUM_OF_GENERATIONS) {
                fitness();
                selection();
                crossover();
                mutation();
                ++generationNum;
            }
            const double resultTime = omp_get_wtime() - startTime;

            printExperimentResult(resultTime, threadsNumList[i]);
            writeResultToFile(resultTime, threadsNumList[i]);

            free(population);
            free(speciesFitness);
        }
    }
    return 0;
}

void printSpecies(const int speciesIndex, const bool printFitness) {
    printf("Species[%d]\n", speciesIndex);
    for (int j = 0; j < NUM_OF_TOURS; ++j) {
        for (int k = 0; k < NUM_OF_PLAYERS; ++k) {
            if (k != 0) {
                printf(", ");
            }
            printf("%d", population(speciesIndex, j, k, NUM_OF_TOURS, NUM_OF_PLAYERS));
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

void printExperimentStart(
    const int index,
    const int numOfThreads
) {
    printf("Experiment #%i: [%d](%d, %d, %d)\n", index + 1, numOfThreads, NUM_OF_TOURS, NUM_OF_PLAYERS, NUM_OF_PLAYGROUNDS);
}

void printExperimentResult(const double resultTime, const int numOfThreads) {
    printf("Num of threads: %d\n", numOfThreads);
    printf("Num of tours: %d\n", NUM_OF_TOURS);
    printf("Num of players: %d\n", NUM_OF_PLAYERS);
    printf("Num of playgrounds: %d\n", NUM_OF_PLAYGROUNDS);
    printf("Time: %fs\n ", resultTime);
    printf("__________________________________________\n");
}

void writeResultFileHeaders() {
    FILE* file = fopen(RESULT_FILE_NAME, "a");
    fprintf(file, "NumOfThreads;NumOfTours;NumOfPlayers;NumOfPlaygrounds;ResultTime\n");
    fclose(file);
}

void writeResultToFile(
    const double resultTime,
    const int numOfThreads
) {
    FILE* file = fopen(RESULT_FILE_NAME, "a");

    fprintf(file, "%d;", numOfThreads);
    fprintf(file, "%d;", NUM_OF_TOURS);
    fprintf(file, "%d;", NUM_OF_PLAYERS);
    fprintf(file, "%d;", NUM_OF_PLAYGROUNDS);
    fprintf(file, "%f", resultTime);

    fprintf(file, "\n");

    fclose(file);
}
