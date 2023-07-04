import numpy as np
import tsplib95
import random2
import time
start = time.time()
# Use tsplib95 package to load data and create a complete distance matrix
def tsplib_distance_matrix(tsplib_file: str) -> np.ndarray:
    
    tsp_problem = tsplib95.load(tsplib_file)
    distance_matrix_flattened = np.array(
        [tsp_problem.get_weight(*edge) for edge in tsp_problem.get_edges()]
    )
    distance_matrix = np.reshape(
        distance_matrix_flattened,
        (tsp_problem.dimension, tsp_problem.dimension),
    )
    # A diagonal filled with zeros
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix
#2-opt Solver
class Solver:
    def __init__(self, distance_matrix, initial_route):
        self.distance_matrix = distance_matrix
        self.num_cities = len(self.distance_matrix)
        self.initial_route = initial_route
        self.best_route = []
        self.best_distance = 0
        self.distances = []

    def update(self, new_route, new_distance):
        self.best_distance = new_distance
        self.best_route = new_route
        return self.best_distance, self.best_route

    @staticmethod
    #calculates the total distance between the first city to the last city in the given path.
    def calculate_path_dist(distance_matrix, path):
        path_distance = 0
        for ind in range(len(path) - 1):
            path_distance += distance_matrix[path[ind]][path[ind + 1]]
        return float("{0:.2f}".format(path_distance))

    @staticmethod
    def swap(path, swap_first, swap_last):
        path_updated = np.concatenate((path[0:swap_first],
                                       path[swap_last:-len(path) + swap_first - 1:-1],
                                       path[swap_last + 1:len(path)]))
        return path_updated.tolist()

  
#2-opt function 
    def two_opt(self, improvement_threshold=0.01):
        self.best_route = self.initial_route
        self.best_distance = self.calculate_path_dist(self.distance_matrix, self.best_route)
        improvement_factor = 1
        
        while improvement_factor > improvement_threshold:
            previous_best = self.best_distance
            for swap_first in range(1, self.num_cities - 3):
                for swap_last in range(swap_first + 1, self.num_cities - 2):
                    before_start = self.best_route[swap_first - 1]
                    start = self.best_route[swap_first]
                    end = self.best_route[swap_last]
                    after_end = self.best_route[swap_last+1]
                    before = self.distance_matrix[before_start][start] + self.distance_matrix[end][after_end]
                    after = self.distance_matrix[before_start][end] + self.distance_matrix[start][after_end]
                    if after < before:
                        new_route = self.swap(self.best_route, swap_first, swap_last)
                        new_distance = self.calculate_path_dist(self.distance_matrix, new_route)
                        self.update(new_route, new_distance)

            improvement_factor = 1 - self.best_distance/previous_best
        return self.best_route, self.best_distance, self.distances

# selected the best result from 10 different randomized initial points
class RouteFinder:
    def __init__(self, distance_matrix, cities_names, iterations=5, writer_flag=False, method='py2opt'):
        self.distance_matrix = distance_matrix
        self.iterations = iterations
        self.writer_flag = writer_flag
        self.cities_names = cities_names

    def solve(self):
               
        iteration = 0
        best_distance = 0
        best_route = []
        
        while iteration < self.iterations:
            num_cities = len(self.distance_matrix)
            initial_route = [0] + random2.sample(range(1, num_cities-1), num_cities - 2)
            tsp = Solver(self.distance_matrix, initial_route)
            new_route, new_distance, distances = tsp.two_opt()

            if iteration == 0:
                best_distance = new_distance
                best_route = new_route
            else:
                pass

            if new_distance < best_distance:
                best_distance = new_distance
                best_route = new_route
                         
            iteration += 1

        if self.writer_flag:
            self.writer(best_route, best_distance, self.cities_names)

        if self.cities_names:
            best_route = [self.cities_names[i] for i in best_route]
            return best_distance, best_route
        else:
            return best_distance, best_route

#Input data: tsplib95 file, cities list, distance matrix to solve the problem
tsp_problem = tsplib95.load('Input file location')
cities_names = list(tsp_problem.get_nodes())
dist_mat = tsplib_distance_matrix('Input file location')
route_finder = RouteFinder(dist_mat, cities_names, iterations=10)
best_distance, best_route = route_finder.solve()

print('Minimal tour length: ' ,best_distance)
print('Best tour order: ', best_route)
end = time.time() 
print('Time complexity: ',end-start)