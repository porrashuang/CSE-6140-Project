/*
 * This program solves the Traveling Salesman Problem (TSP) using three different approaches:
 * 1. Exact Algorithm (BF): A brute-force method to find the shortest route among all permutations of cities.
 * 2. Approximate Algorithm (Approx): An approximation approach based on a Minimum Spanning Tree (MST) and preorder traversal.
 * 3. Local Search (LS): A genetic algorithm that combines selection, crossover, and mutation to optimize the tour.
 * 
 * Features:
 * - Reads TSP input files with city coordinates and creates a distance matrix.
 * - Saves results to a solution file in a specific format.
 * - Handles early termination via signal handling (SIGUSR1).
 * 
 * Usage:
 *   ./program_name -inst <filename> -alg [BF | Approx | LS] -time <cut_off> [-seed <random_seed>]
 *   - `filename`: Input file containing city coordinates.
 *   - `BF`: Exact brute-force algorithm.
 *   - `Approx`: Approximation algorithm.
 *   - `LS`: Local search using genetic algorithm (requires seed).
 *   - `cut_off`: Time limit for execution (not applicable for Approx).
 *   - `random_seed`: Random seed for local search (LS).
 * 
 * Output:
 * - Prints the best route distance and sequence of cities.
 * - Saves the solution to a file in the format "<filename>_<method>_<time>.sol".
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <chrono>
#include <random>
#include <algorithm> // For std::random_shuffle and std::find
#include <queue>          // For priority_queue
#include <unordered_set>  // For unordered_set
#include <functional>     // For std::function
#include <csignal>       // For signal handling

const static int MAX_GENERATIONS = 250;
const static int POPULATION_SIZE = 20;
const static double MUTATION_RATE = 0.4;
const static double TOUR_PERCENTAGE = 0.3;

using namespace std;
// Struct for storing point data
struct Point {
    int id;
    double x, y;
    
    Point(int id, double x, double y) : id(id), x(x), y(y) {}
};

// Struct for a dataset
struct Dataset {
    string name;
    vector<Point> points;
    vector<vector<double>> distanceMatrix;
    Dataset(string name, vector<Point> &&points, vector<vector<double>> &&distanceMatrix) : name(name), points(points), distanceMatrix(distanceMatrix) {}
};

struct Settings {
    string filename;
    string method;
    int timeLimit;
    int seed;
    std::chrono::time_point<std::chrono::system_clock> startTime;
    Settings(): timeLimit(0), seed(0), startTime(std::chrono::system_clock::now()) {}
    void Init(const std::string& filename, const std::string& method, int timeLimit, int seed) 
    {
        this->filename = filename;
        this->method = method;
        this->timeLimit = timeLimit;
        this->seed = seed;
        this->startTime = std::chrono::system_clock::now();
    }
    
};

struct Answer {
    double totalDistance;
    vector<int> sequence;
    Answer(): totalDistance(__DBL_MAX__) {}
};

static Answer answer;
static Settings settings;
// Function to calculate Euclidean distance between two points
double calculateDistance(const Point& a, const Point& b) 
{
    return round(sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)));
}

Dataset parseTSPFile(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::string name;
    std::vector<Point> points;

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return Dataset("", {}, {});
    }

    while (std::getline(file, line)) {
        // Parse the name
        if (line.rfind("NAME", 0) == 0) {
            std::istringstream iss(line);
            std::string discard;
            iss >> discard >> name;
        }
        
        // Start reading points after the NODE_COORD_SECTION line
        else if (line == "NODE_COORD_SECTION") {
            while (std::getline(file, line) && line != "EOF") {
                std::istringstream iss(line);
                int id;
                double x, y;
                if (iss >> id >> x >> y) {
                    points.emplace_back(id, x, y);
                }
            }
        }
    }
    vector<std::vector<double>> distanceMatrix(points.size(), vector<double>(points.size(), 0.0));
    for (int i = 0; i < points.size(); ++i) {
        for (int j = i + 1; j < points.size(); ++j) {
            double dx = points[i].x - points[j].x;
            double dy = points[i].y - points[j].y;
            double distance = std::round(std::sqrt(dx * dx + dy * dy));
            distanceMatrix[i][j] = distance;
            distanceMatrix[j][i] = distance; // Symmetric matrix
        }
    }


    file.close();
    return Dataset(name, std::move(points), std::move(distanceMatrix));
}

// calculate tour distance
double calcTourDist(const vector<int>& tour, const Dataset& dataset) {
    double totalDist = 0;
    for (int i = 0; i < tour.size() - 1; i++) {
        totalDist += dataset.distanceMatrix[tour[i]][tour[i + 1]];
    }
    // Add last point -> starting point distance
    totalDist += dataset.distanceMatrix[tour.back()][tour[0]];
    return totalDist;
}

void bruteForceRecursive(int start, int end, vector<int> &sequence, const Dataset& dataset)
{
    if (start == end)
    {
        // Calculate accumulated distance
        double accumulated = calcTourDist(sequence, dataset);
        // Store the answer
        if (accumulated < answer.totalDistance)
        {
            answer.totalDistance = accumulated;
            answer.sequence = sequence;
        }
        return;
    }
    for (int i = start; i <= end; i++)
    {
        swap(sequence[i], sequence[start]);
        bruteForceRecursive(start + 1, end, sequence, dataset);
        swap(sequence[i], sequence[start]);
    }
}
// Placeholder for exact algorithm (brute-force)
void exactAlgorithm(const Dataset& dataset) {
    // Implement brute-force algorithm here
    vector<int> sequence;
    for (int i = 0; i < dataset.points.size(); i++)
    {
        sequence.push_back(i);
    }
    bruteForceRecursive(0, dataset.points.size() - 1, sequence, dataset);
}

void approximateAlgorithm(const Dataset& dataset) {
    const auto& points = dataset.points;
    const auto& distanceMatrix = dataset.distanceMatrix;
    int n = points.size();

    // Build the MST using Prim's algorithm
    vector<bool> visited(n, false);
    vector<vector<int>> mst(n);
    priority_queue<pair<double, pair<int, int>>, vector<pair<double, pair<int, int>>>, greater<>> minHeap;

    visited[0] = true; // Start with city 0
    for (int i = 1; i < n; ++i) {
        minHeap.push({distanceMatrix[0][i], {0, i}});
    }

    while (!minHeap.empty()) {
        auto top = minHeap.top();
        double cost = top.first;
        auto edge = top.second;
        int u = edge.first;
        int v = edge.second;
        minHeap.pop();

        if (visited[v]) continue;
        visited[v] = true;
        mst[u].push_back(v);
        mst[v].push_back(u);

        for (int w = 0; w < n; ++w) {
            if (!visited[w]) {
                minHeap.push({distanceMatrix[v][w], {v, w}});
            }
        }
    }

    // Preorder traversal of the MST to approximate the TSP tour
    vector<int> tour;
    unordered_set<int> visitedNodes;
    double totalDistance = 0.0;

    function<void(int)> preorderTraversal = [&](int node) {
        visitedNodes.insert(node);
        tour.push_back(node);

        // Record the path when going through the best route
        if (tour.size() > 1) {
            totalDistance += calculateDistance(points[tour[tour.size() - 2]], points[tour.back()]);
        }

        for (int neighbor : mst[node]) {
            if (visitedNodes.find(neighbor) == visitedNodes.end()) {
                preorderTraversal(neighbor);
            }
        }
    };
    preorderTraversal(0);

    // After finishing the traversal, calculate the return distance (last to first)
    if (!tour.empty()) {
        totalDistance += calculateDistance(points[tour.back()], points[tour.front()]);
        tour.push_back(tour.front());
    }
    // Store the results in the global `answer` object
    answer.sequence = tour;
    answer.totalDistance = totalDistance;
}



// initialize populations
vector<vector<int>> initPopulation(const Dataset& dataset, int numCities, default_random_engine& rng) {
    vector<vector<int>> population;
    
    approximateAlgorithm(dataset);
    vector<int> baseTour(numCities);
    
    cout << "Base tour from 2-approx: " << endl;
    for (int i = 0; i < numCities; ++i) {
        baseTour[i] = answer.sequence[i];
        cout << answer.sequence[i] << " ";
    }
    cout << endl << "2-approx total Distance: " << answer.totalDistance << endl << endl;
    population.push_back(baseTour);
    for (int i = 1; i < POPULATION_SIZE; ++i) {
        // Shuffle the base tour to create a new, randomized tour
        shuffle(baseTour.begin(), baseTour.end(), rng);
        
        // Add the randomized tour to the population
        population.push_back(baseTour);
    }
    
    return population;
}

// selection of parents tour
vector<int> tourSelect(const vector<vector<int>>& population, const Dataset& dataset, default_random_engine& rng) {
    uniform_int_distribution<int> randIdx(0, population.size() - 1);
    
    int tourSize = dataset.points.size() * TOUR_PERCENTAGE;
    vector<int> bestTour = population[randIdx(rng)];
    double bestFitness = calcTourDist(bestTour, dataset);
    
    for (int i = 1;i < tourSize; i++) {
        vector<int> contender = population[randIdx(rng)];
        
        double contenderFitness = calcTourDist(contender, dataset);
        
        if (contenderFitness < bestFitness) {
            bestTour = contender;
            bestFitness = contenderFitness;
        }
    }
    
    return bestTour;
}

// Crossover
vector<int> Crossover(const vector<int>& parent1, const vector<int>& parent2, default_random_engine& rng) {
    int size = parent1.size();
    vector<int> child(size, -1);
    
    uniform_int_distribution<int> randRange(0, size - 1);
    
    // Randomly choose sub-tour to inherit from parent1
    int start = randRange(rng);
    int end = start + randRange(rng) % (size - start);
    
    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }
    
    // Fill in remaining tour from parent2
    int childIdx = (end + 1) % size;
    for (int i = 0; i < size; ++i) {
        int parent2City = parent2[(end + 1 + i) % size];
        if (find(child.begin(), child.end(), parent2City) == child.end()) {
            child[childIdx] = parent2City;
            childIdx = (childIdx + 1) % size;
        }
    }
    return child;
}
// Mutate
void Mutate(vector<int>& tour, default_random_engine& rng) {
    uniform_int_distribution<int> randIdx(0, tour.size() - 1);
    
    int i = randIdx(rng);
    int j = randIdx(rng);
    swap(tour[i], tour[j]);
}

// Placeholder for local search algorithm (e.g., Simulated Annealing)
void localSearchAlgorithm(const Dataset& dataset, int seed) {
    cout << "Local Search Algo function running...\n";
    
    // Implement local search algorithm here
    int numCities = dataset.points.size();
    
    // Initialize random engine
    default_random_engine rng(seed);
    
    // initialize population
    vector<vector<int>> population = initPopulation(dataset, numCities, rng);
    
    // Main loop for generations
    for (int curGeneration = 0; curGeneration < MAX_GENERATIONS; ++curGeneration) {
        vector<vector<int>> newPopulation;
        
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            
            // selection of parents 
            vector<int> parent1 = tourSelect(population, dataset, rng);
            vector<int> parent2;
            do {
                parent2 = tourSelect(population, dataset, rng);
            } while (parent1 == parent2);

            // crossover
            vector<int> child = Crossover(parent1, parent2, rng);
             
            // mutation 
            uniform_real_distribution<double> randReal(0.0, 1.0);
            if ((randReal(rng) < MUTATION_RATE)) {
                Mutate(child, rng);
            }
            // add in child to new population
            newPopulation.push_back(child);
        }
        
        // replace old population with new one
        population = newPopulation;
        
        // Print best tour in curGeneration
        double bestDist = calcTourDist(population[0], dataset);
        vector<int> bestTour = population[0];
        for (const auto& tour : population) {
            double dist = calcTourDist(tour, dataset);
            if (dist < bestDist) {
                bestDist = dist;
                bestTour = tour;
            }
        }
        answer.totalDistance = bestDist;
        answer.sequence = bestTour;
    }
    
    // last check of best tour
    double bestDist = calcTourDist(population[0], dataset);
    vector<int> bestTour = population[0];
    for (const auto& tour : population) {
        double dist = calcTourDist(tour, dataset);
        if (dist < bestDist) {
            bestDist = dist;
            bestTour = tour;
        }
    }
    

    answer.totalDistance = bestDist;
    answer.sequence = bestTour;
    return;

}

// Function to save solution to file
void saveSolution() {
    string filename = settings.filename + "_" + settings.method + "_" + 
        (settings.method == "Approx" ? "" : to_string(settings.timeLimit)) + 
        (settings.method == "LS" ? "_" + to_string(settings.seed) : "") + 
        ".sol";
    ofstream outFile(filename);
    
    if (outFile.is_open()) {
        // outFile << "Time Spent: " << chrono::duration<double>(chrono::system_clock::now() - settings.startTime).count() << "\n";
        outFile << answer.totalDistance << "\n";
        for (int i = 0; i < answer.sequence.size(); ++i) {
            outFile << answer.sequence[i] << (i < answer.sequence.size() - 1 ? "," : "");
        }
        outFile.close();
        cout << "Solution saved to " << filename << endl;
    } else {
        cerr << "Error: Unable to open file " << filename << " for writing." << endl;
    }
}

void printAndSaveAnswer() {
    printf("The time taken is %f\n", chrono::duration<double>(chrono::system_clock::now() - settings.startTime).count());
    printf("The best route distance is %f\n", answer.totalDistance);
    printf("The sequence is: ");
    for (auto idx : answer.sequence)
    {
        printf("%d, ", idx);
    }
    printf("\n");
    saveSolution();
}

void signalHandler(int signal) {
    if (signal == SIGUSR1) {
        printf("Early termination, saving the current best solution...\n");
        printAndSaveAnswer();
        exit(0);
    }
}

// Main function to handle input and select algorithm
int main(int argc, char* argv[]) {
    if (argc < 7) {
        cerr << "Usage: " << argv[0] << " -inst <filename> -alg [BF | Approx | LS] -time <cut_off> [-seed <random_seed>]" << endl;
        return 1;
    }

    string filename;
    string method;
    int timeLimit = 0;
    int seed = 0;

    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-inst" && i + 1 < argc) {
            filename = argv[++i];
        } else if (string(argv[i]) == "-alg" && i + 1 < argc) {
            method = argv[++i];
        } else if (string(argv[i]) == "-time" && i + 1 < argc) {
            timeLimit = atoi(argv[++i]);
        } else if (string(argv[i]) == "-seed" && i + 1 < argc) {
            seed = atoi(argv[++i]);
        }
    }

    if (filename.empty() || method.empty() || timeLimit == 0) {
        cerr << "Error: Missing required arguments" << endl;
        return 1;
    }
    settings.Init(filename, method, timeLimit, seed);
    signal(SIGUSR1, signalHandler);
    Dataset dataset = parseTSPFile(filename);
    if (method == "BF") 
    {
        exactAlgorithm(dataset);        
    } 
    else if (method == "Approx") 
    {
        approximateAlgorithm(dataset);        
    } 
    else if (method == "LS") 
    {
        localSearchAlgorithm(dataset, seed);        
    } 
    else 
    {
        cerr << "Error: Unknown method " << method << endl;
        return 1;
    }
    printAndSaveAnswer();
    
    return 0;
}