# coding: utf-8
import queue
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import psutil
import tqdm
from termcolor import colored
from node import Node

# Global variables
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
goal_node = Node
initial_state = list()
board_length = 0
board_side = 0
nodes_to_expand = 0
max_depth = 0
moves = list()
costs = set()
depth = 0


def A_Star_Manhattan(start_state):
    """
        a_star: Implements the A-star search

            A-star search (a_star)

            A-star search (a_star) is a computer algorithm that is widely used in pathfinding and graph traversal,
            which is the process of finding a path between multiple points, called "nodes".
            It enjoys widespread use due to its performance and accuracy.
        input: Start state or initial state of the 8 puzzle game
    """
    global goal_node, max_depth, goal_state

    open_list, closed_list = queue.PriorityQueue(), dict()

    nodes_visited = 0
    nodes_expanded = 0

    node = Node(start_state, None, None, 0, 0, 0)

    open_list.put_nowait((node.cost, node.depth, node))

    while open_list:

        # print('Father Node:')
        # print_board(node.state)
        node = open_list.get_nowait()[2]  # Return the first element

        if node.state == goal_state:
            # print('Answer found:\n')
            # print_board(node.state)
            goal_node = node
            nodes_visited = len(closed_list)
            return open_list, nodes_visited, nodes_expanded, goal_node.depth

        closed_list[node.map] = node.map

        successor = expand(node)

        # print('Children:')
        # print('-----------------------------')

        nodes_expanded += 1

        for child in successor:
            if not check_dict(closed_list, child.map):
                # Euclidian_Distance(child)
                Manhattan_Distance(child)
                # euclidian(child)
                child.cost = child.depth + child.heuristic  # f(n) = g(n) + h(n)

                if child.depth > max_depth:
                    max_depth += 1

                open_list.put_nowait((child.cost, node.depth, child))


def A_Star_Euclidian(start_state):
    """
        a_star: Implements the A-star search

            A-star search (a_star)

            A-star search (a_star) is a computer algorithm that is widely used in pathfinding and graph traversal,
            which is the process of finding a path between multiple points, called "nodes".
            It enjoys widespread use due to its performance and accuracy.
        input: Start state or initial state of the 8 puzzle game
    """
    global goal_node, max_depth, goal_state

    open_list, closed_list = queue.PriorityQueue(), dict()

    nodes_visited = 0
    nodes_expanded = 0

    node = Node(start_state, None, None, 0, 0, 0)

    open_list.put((node.cost, node.depth, node))

    while open_list:

        # print('Father Node:')
        # print_board(node.state)
        node = open_list.get()[2]  # Return the first element

        if node.state == goal_state:
            # print('Answer found:\n')
            # print_board(node.state)
            goal_node = node
            nodes_visited = len(closed_list)
            return open_list, nodes_visited, nodes_expanded, goal_node.depth

        closed_list[node.map] = node.map

        successor = expand(node)

        # print('Children:')
        # print('-----------------------------')

        nodes_expanded += 1

        for child in successor:
            if not check_dict(closed_list, child.map):
                Euclidian_Distance(child)
                # Manhattan_Distance(child)
                # euclidian(child)
                child.cost = child.depth + child.heuristic  # f(n) = g(n) + h(n)

                if child.depth > max_depth:
                    max_depth += 1

                open_list.put_nowait((child.cost, node.depth, child))


def Euclidian_Distance(node):
    """
        evaluation_function: Implement the heuristic for construe a cost estimate. This heuristics is the Euclidian
        distance to achieve the goal state (1, 2, 3, 4, 5, 6, 7, 8, 0)

            input: The current node
            output: The evaluation value of the given node
    """
    global goal_state

    if len(node.state) != len(goal_state):
        raise ValueError('List of different length!')
    v1, v2 = np.array(goal_state), np.array(node.state)
    diff = v1 - v2
    quad_dist = np.dot(diff, diff)
    node.heuristic = math.sqrt(quad_dist)


def Manhattan_Distance(node):
    """
        evaluation_function: Implement the heuristic for construe a cost estimate. This heuristics consist in the
        Manhattan distance to the goal state (1, 2, 3, 4, 5, 6, 7, 8, 0)

            input: The current node
            output: The evaluation value of the given node
    """
    global goal_state

    cost2goal_sum = 0

    if len(node.state) != len(goal_state):
        raise ValueError('List of different length!')
    #
    for i in range(len(goal_state)):
        for j in range(i + 1, len(goal_state)):
            cost2goal_sum += (abs(node.state[i] - node.state[j]) + abs(goal_state[i] - goal_state[j]))

    node.heuristic = cost2goal_sum


def print_board(state):
    print('---------')
    print(state[0], state[1], state[2])
    print('---------')
    print(state[3], state[4], state[5])
    print('---------')
    print(state[6], state[7], state[8])
    print('---------')

    return


def expand(node):
    """
        expand: Implement the expansion for search strategies algorithms

            input: The current node that shal be expanded
            output: The children nodes expanded from the father node
    """

    global nodes_to_expand
    nodes_to_expand += 1

    children = list()

    for possible_move in range(1, 5):
        children.append((Node(move(node.state, possible_move), node, possible_move, node.depth + 1, node.cost + 1, 0)))

    children = [child for child in children if child.state]

    return children


def move(state, position):
    new_state = state[:]

    index = new_state.index(0)

    if position == 1:  # Up

        if index not in range(0, board_side):

            temp = new_state[index - board_side]
            new_state[index - board_side] = new_state[index]
            new_state[index] = temp

            return new_state
        else:
            return None

    if position == 2:  # Down

        if index not in range(board_length - board_side, board_length):

            temp = new_state[index + board_side]
            new_state[index + board_side] = new_state[index]
            new_state[index] = temp

            return new_state
        else:
            return None

    if position == 3:  # Left

        if index not in range(0, board_length, board_side):

            temp = new_state[index - 1]
            new_state[index - 1] = new_state[index]
            new_state[index] = temp

            return new_state
        else:
            return None

    if position == 4:  # Right

        if index not in range(board_side - 1, board_length, board_side):

            temp = new_state[index + 1]
            new_state[index + 1] = new_state[index]
            new_state[index] = temp

            return new_state
        else:
            return None


def user_input(conf):
    """
        user_input: Read the important information to run the algorithm.
    """
    global board_length, board_side, initial_state
    initial_state = []
    data = conf.split(",")

    for element in data:
        initial_state.append(int(element))

    board_length = len(initial_state)

    board_side = int(board_length ** 0.5)  # Simple geometry: side = sqtr(board_length)

    return initial_state


def read_file():
    # Open the file for reading.
    with open('entries_fifteen.txt', 'r') as infile:
        data = infile.read()  # Read the contents of the file into memory.

    # Return a list of the lines, breaking at line boundaries.
    my_list = data.splitlines()
    # print(my_list)
    return my_list


def check_dict(list,key):
    try:
        test_dict = list[str(key)]
        return True
    except KeyError:
        return False


def backtrace():

    current_node = goal_node

    while initial_state != current_node.state:

        if current_node.move == 1:
            movement = 'Up'
        elif current_node.move == 2:
            movement = 'Down'
        elif current_node.move == 3:
            movement = 'Left'
        else:
            movement = 'Right'
        moves.insert(0, movement)
        current_node = current_node.parent

    return moves


def export(algorithm, input_state, visited, explored, solution_depth, time_):

    global goal_node, moves

    moves = list()

    moves = backtrace()

    file = open('Output for algorithm: ' + str(algorithm) + ' instance: [' + str(input_state) + '] 15-Puzzle.txt', 'w')
    file.write("Path to goal: " + str(moves))
    file.write("\n\nWhat is the initial state?: [" + input_state + ']')
    file.write("\nWhat is the final state?: " + str(goal_node.state))
    file.write("\nHow much moves?: " + str(len(moves)))
    file.write("\nHow much nodes were explored?: " + str(explored))
    file.write("\nHow much nodes were visited?: " + str(visited))
    file.write("\nAnswer's depth: " + str(solution_depth))
    file.write("\nMax depth reached: " + str(max_depth))
    file.write("\nTime to run the instance: " + format(time_, '.3f') + ' seconds')
    file.close()


def plot_partial_result(algorithm, time_, explored, visited, solution_depth):
    x1 = time_
    x2 = explored
    x3 = visited
    x4 = solution_depth

    plt.figure(figsize=(20, 10))
    #
    plt.subplot(4, 1, 1)

    plt.plot(x1, color='r')
    plt.title('Resultados para o algoritmo: ' + str(algorithm), fontsize=20)
    plt.ylabel('Tempo (s)', fontsize=15)
    # plt.yscale(value='log')

    plt.subplot(4, 1, 2)
    plt.plot(x2, color='b')
    plt.ylabel('Explorados', fontsize=15)
    # plt.yscale(value='log')

    plt.subplot(4, 1, 3)
    plt.plot(x3, color='g')
    # plt.xlabel('Instancias do problema', fontsize=15)
    plt.ylabel('Visitados', fontsize=15)
    # plt.yscale(value='log')

    plt.subplot(4, 1, 4)
    plt.plot(x4, color='black')
    plt.xlabel('Instancias do problema', fontsize=15)
    plt.ylabel('Prof. resposta', fontsize=15)
    # plt.yscale(value='log')

    # plt.savefig('/images/results_'+str(algorithm)+'.png')
    plt.savefig('Resultados_' + str(algorithm) + '.png')


def plot_combined_results(algorithm, time_, explored, visited, solution_depth):
    # print(algorithm, time, explored, visited, solution_depth)
    # Time Plot
    plt.figure()
    for i in range(len(explored)):
        plt.plot(explored[i], label=algorithm[i])
    plt.yscale('log')
    plt.title('Nós Explorados', fontsize=20)
    plt.xlabel('Instancias')
    plt.ylabel('Nós explorados')
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('Nodes Explorados.png')

    plt.figure()
    for i in range(len(time_)):
        plt.plot(time_[i], label=algorithm[i])
    plt.yscale('log')
    plt.title('Tempo de Execução', fontsize=20)
    plt.xlabel('Instancias')
    plt.ylabel('Tempo de Execução')
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('Tempo de Execução.png')

    plt.figure()
    for i in range(len(visited)):
        plt.plot(visited[i], label=algorithm[i])
    plt.yscale('log')
    plt.title('Nós visitados', fontsize=20)
    plt.xlabel('Instancias')
    plt.ylabel('Nós visitados')
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('Nodes visitados.png')

    plt.figure()
    for i in range(len(solution_depth)):
        plt.plot(solution_depth[i], label=algorithm[i])
    # plt.yscale('log')
    plt.title('Profundidade da solução', fontsize=20)
    plt.xlabel('Instancias')
    plt.ylabel('Profundidade da sulução')
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('Profundidade encontrada.png')


def main():
    board = read_file()  # READ FILE

    global goal_node

    function_map = {'a_star_euclidian': A_Star_Euclidian}

    # , 'a_star_manhattan': A_Star_Manhattan

    _visited_ = list()
    _explored_ = list()
    _time_ = list()
    _solution_depth_ = list()
    _algorithm_name_ = list()

    for algorithm_ in function_map.values():

        visited_ = list()
        explored_ = list()
        time_ = list()
        solution_depth_ = list()

        print(colored('Solving using algorithm: ' + str(algorithm_.__name__), 'red'))
        for input_state in tqdm.tqdm(board):
            # print(colored('Instance to be solved: ' + str(input_state), 'red'))

            user_input(input_state)

            start = time.time()

            try:
                algorithm, visited, explored, solution_depth = algorithm_(initial_state)
            except:
                algorithm, visited, explored, solution_depth = list(), 0, 0, 0

            stop = time.time()
            time_elapsed = stop - start

            try:
                time_.append(time_elapsed)
                visited_.append(visited)
                explored_.append(explored)
                solution_depth_.append(solution_depth)
            except:
                time_.append(0)
                visited_.append(0)
                explored_.append(0)
                solution_depth_.append(solution_depth)

            export(algorithm_.__name__, input_state, visited, explored, solution_depth, time_elapsed)

        plot_partial_result(str(algorithm_.__name__), time_, explored_, visited_, solution_depth_)

        _visited_.append(visited_)

        _explored_.append(explored_)

        _time_.append(time_)

        _solution_depth_.append(solution_depth_)
        _algorithm_name_.append(str(algorithm_.__name__))

    plot_combined_results(_algorithm_name_, _time_, _explored_, _visited_, _solution_depth_)


if __name__ == '__main__':
    main()