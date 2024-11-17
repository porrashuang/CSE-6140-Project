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
struct Answer {
    double totalDistance;
    vector<size_t> sequence;
    Answer(): totalDistance(__DBL_MAX__) {}
};

static Answer answer;
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
void bruteForceRecursive(size_t start, size_t end, vector<size_t> &sequence, const vector<vector<double>> &distanceMatrix)
{
    if (start == end)
    {
        // Calculate accumulated distance
        double accumulated = 0.0;
        for (int i = 1; i < sequence.size(); i++)
        {
            accumulated += distanceMatrix[sequence[i]][sequence[i - 1]];
        }
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
        bruteForceRecursive(start + 1, end, sequence, distanceMatrix);
        swap(sequence[i], sequence[start]);
    }
}
// Placeholder for exact algorithm (brute-force)
vector<int> exactAlgorithm(const Dataset& dataset, int timeLimit) {
    // Implement brute-force algorithm here
    vector<size_t> sequence;
    for (int i = 0; i < dataset.points.size(); i++)
    {
        sequence.push_back(i);
    }
    bruteForceRecursive(0, dataset.points.size() - 1, sequence, dataset.distanceMatrix);
    return {0}; // Replace with computed tour
}

// Placeholder for approximation algorithm (MST-based 2-approximation)
vector<int> approximateAlgorithm(const vector<Point>& points) {
    // Implement 2-approximation algorithm here
    // Placeholder for result
    return {0}; // Replace with computed tour
}

// Functions for local search (GA)
// calculate tour distance
double calc_tour_dist(const vector<int>& tour, const Dataset& dataset) {
    double total_dist = 0;
    for (size_t i = 0; i < tour.size() - 1; i++) {
        total_dist += dataset.distanceMatrix[tour[i]][tour[i + 1]];
    }
    // Add last point -> starting point distance
    total_dist += dataset.distanceMatrix[tour.back()][tour[0]];
    return total_dist;
}

// initialize populations
vector<vector<int>> init_population(int population_size, int num_cities) {
    vector<vector<int>> population;
    vector<int> base_tour(num_cities);
    for (int i = 0; i < num_cities; ++i) base_tour[i] = i;
    
    // Create random device and random engine
    random_device rd;
    
    for (int i = 0; i < population_size; ++i) {
        // Seed a new random engine for each shuffle to ensure randomness
        default_random_engine rng(rd());
        
        // Shuffle the base tour to create a new, randomized tour
        shuffle(base_tour.begin(), base_tour.end(), rng);
        
        // Add the randomized tour to the population
        population.push_back(base_tour);
    }
    
    return population;
}

// selection of parents tour
vector<int> tour_select(const vector<vector<int>>& population, const Dataset& dataset) {
    int tour_size = dataset.points.size() * 0.1; // larger tour size, more likely to get the best in all tours
    vector<int> best_tour = population[rand() % population.size()];
    double best_fitness = calc_tour_dist(best_tour, dataset);
    
    for (int i = 1;i < tour_size; i++) {
        vector<int> contender = population[rand() % population.size()];
        double contender_fitness = calc_tour_dist(contender, dataset);
        if (contender_fitness < best_fitness) {
            best_tour = contender;
            best_fitness = contender_fitness;
        }
    }
    
    return best_tour;
}

// Crossover
vector<int> Crossover(const vector<int>& parent1, const vector<int>& parent2) {
    int size = parent1.size();
    vector<int> child(size, -1);
    
    // Randomly choose sub-tour to inherit from parent1
    int start = rand() % size;
    int end = start + (rand() % (size - start));
    
    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }
    
    // Fill in remaining tour from parent2
    int child_idx = (end + 1) % size;
    for (int i = 0; i < size; ++i) {
        int parent2_city = parent2[(end + 1 + i) % size];
        if (find(child.begin(), child.end(), parent2_city) == child.end()) {
            child[child_idx] = parent2_city;
            child_idx = (child_idx + 1) % size;
        }
    }
    return child;
}
// Mutate
void Mutate(vector<int>& tour) {
    cout << "Mutation!" << endl;
    int i = rand() % tour.size();
    int j = rand() % tour.size();
    swap(tour[i], tour[j]);
}

// Placeholder for local search algorithm (e.g., Simulated Annealing)
void localSearchAlgorithm(const Dataset& dataset, int seed) {
    cout << "Local Search Algo function running...\n";
    //srand(seed);
    // Implement local search algorithm here
    int num_cities = dataset.points.size();
    
    // Manual set parameters
    int population_size = 50;
    int max_generations = 100;
    int mutation_rate = 0.2; // between 0 ~ 1
    
    // initialize population
    vector<vector<int>> population = init_population(population_size, num_cities);
    
    // Main loop for generations
    for (int cur_generation = 0; cur_generation < max_generations; ++cur_generation) {
        vector<vector<int>> new_population;
        
        for (int i = 0; i < population_size; ++i) {
            
            // selection of parents 
            vector<int> parent1 = tour_select(population, dataset);
            vector<int> parent2;
            do {
                parent2 = tour_select(population, dataset);
            } while (parent1 == parent2);
            
            /*
            cout << "p1: ";
            for (int i: parent1) cout << i << ' ';
            cout << "p2: ";
            for (int i: parent2) cout << i << ' ';
            cout << endl;
            */
            
            // crossover
            vector<int> child = Crossover(parent1, parent2);
             
            // mutation 
            if ((rand() < (double)RAND_MAX * mutation_rate)) {
                Mutate(child);
            }
            // add in child to new population
            new_population.push_back(child);
        }
        
        // replace old population with new one
        population = new_population;
        
        // Print best tour in cur_generation
        double best_dist = calc_tour_dist(population[0], dataset);
        vector<int> best_tour = population[0];
        for (const auto& tour : population) {
            double dist = calc_tour_dist(tour, dataset);
            if (dist < best_dist) {
                best_dist = dist;
                best_tour = tour;
            }
        }
        cout << "Gen " << cur_generation << " - Best Distance: " << best_dist << endl;
    }
    
    // last check of best tour
    double best_dist = calc_tour_dist(population[0], dataset);
    vector<int> best_tour = population[0];
    for (const auto& tour : population) {
        double dist = calc_tour_dist(tour, dataset);
        if (dist < best_dist) {
            best_dist = dist;
            best_tour = tour;
        }
    }
    
    cout << "Final result: " << best_dist << endl;
    cout << "Tour: ";
    for (int i: best_tour) cout << i << ' ';
    cout << endl;
    
    vector<size_t> best_tour_convert(best_tour.size());
    transform(best_tour.begin(), best_tour.end(), best_tour_convert.begin(), [](int val) { return static_cast<size_t>(val); });
    
    answer.totalDistance = best_dist;
    answer.sequence = best_tour_convert;
    return;
    
    //return best_tour;
    // Placeholder for result
    //return {0}; // Replace with computed tour
}

// Function to save solution to file
void saveSolution(const string& instance, const string& method, int timeLimit, int seed, double quality, const vector<size_t>& tour) {
    string filename = instance + " " + method + " " + to_string(timeLimit) + (method == "LS" ? " " + to_string(seed) : "") + ".sol";
    ofstream outFile(filename);
    
    if (outFile.is_open()) {
        outFile << quality << "\n";
        for (size_t i = 0; i < tour.size(); ++i) {
            outFile << tour[i] << (i < tour.size() - 1 ? "," : "");
        }
        outFile.close();
        cout << "Solution saved to " << filename << endl;
    } else {
        cerr << "Error: Unable to open file " << filename << " for writing." << endl;
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
    Dataset dataset = parseTSPFile(filename);
    if (method == "BF") 
    {
        exactAlgorithm(dataset, timeLimit);        
    } 
    else if (method == "Approx") 
    {
        approximateAlgorithm(dataset.points);        
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
    printf("The best route distance is %f\n", answer.totalDistance);
    printf("The sequence is: ");
    for (auto idx : answer.sequence)
    {
        printf("%d, ", idx);
    }
    printf("\n");
    // TODO: Calculate the quality
    // Save the result
    saveSolution(filename, method, timeLimit, seed, 0, answer.sequence);
    
    return 0;
}