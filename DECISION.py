import numpy as np
import copy


class decision:

    def __init__(self, size_world, my_belief_target, id_robot, id_contact, position_robot_estimate, my_sensor_target, path_depth, number_of_directions, step_distance):
        self.size_world = size_world
        self.my_belief_target = my_belief_target
        self.position_robot_estimate = position_robot_estimate
        self.id_robot = id_robot
        self.id_contact = id_contact
        self.my_sensor_target = my_sensor_target

        # Parameters that determine the motion of the robot
        self.path_depth = path_depth
        self.number_of_directions = number_of_directions
        self.step_distance = step_distance
        self.step_angle = 2 * np.pi / self.number_of_directions


    def decide(self):
        # Find out who is involved in the decision
        id_robot = []
        for x in range(len(self.id_contact)):
            if self.id_contact[x][1] == 1:
                id_robot = id_robot + [x]

        # Initialize all possible positions, according novelty matrices and values
        position_observe = [[[0]]]
        novelty = [[[[[0]]]]]
        value = [[0]]
        for k in range(1, self.path_depth + 1):
            for j in range(0, len(id_robot)):
                layer = (k - 1) * len(id_robot) + j + 1
                position_observe = position_observe + [[[0, 0] for i in range((self.number_of_directions + 1) ** (layer))]]

                value = value + [[0 for i in range((self.number_of_directions + 1) ** (layer))]]

        # Determine start level
        for k in range(1, 2):
            for j in range(0, len(id_robot)):
                layer = (k - 1) * len(id_robot) + j + 1

                # Determine start position depending on which robot we are looking at
                if id_robot[j] == self.id_robot:
                    position_start = copy.deepcopy(self.position_robot_estimate[id_robot[j]])
                else:
                    dx = copy.deepcopy(self.position_robot_estimate[id_robot[j]][1] * np.cos(self.position_robot_estimate[id_robot[j]][0]))
                    dy = copy.deepcopy(self.position_robot_estimate[id_robot[j]][1] * np.sin(self.position_robot_estimate[id_robot[j]][0]))
                    position_start = [copy.deepcopy(self.position_robot_estimate[self.id_robot][0]) + dx, copy.deepcopy(self.position_robot_estimate[self.id_robot][1]) + dy]

                # Fill the in the first depth level
                for i in range((self.number_of_directions + 1) ** layer):
                    if (i % self.number_of_directions != 0) | (i == 0):
                        p_x = position_start[0] + self.step_distance * np.cos((i % self.number_of_directions) * self.step_angle)
                        p_y = position_start[1] + self.step_distance * np.sin((i % self.number_of_directions) * self.step_angle)
                        position_observe[layer][i] = [p_x, p_y]
                    else:
                        position_observe[layer][i] = position_start

        # Fill in the deeper path tree
        for k in range(2, self.path_depth + 1):
            for j in range(0, len(id_robot)):
                layer = (k - 1) * len(id_robot) + j + 1

                # Fill the in the higher levels
                for i in range((self.number_of_directions + 1) ** layer):
                    if (i % self.number_of_directions != 0) | (i == 0):
                        p_x = position_observe[layer-len(id_robot)][int(i / ((self.number_of_directions + 1) ** len(id_robot)))][0] + self.step_distance * np.cos((i % self.number_of_directions) * self.step_angle)
                        p_y = position_observe[layer-len(id_robot)][int(i / ((self.number_of_directions + 1) ** len(id_robot)))][1] + self.step_distance * np.sin((i % self.number_of_directions) * self.step_angle)
                        position_observe[layer][i] = [p_x, p_y]
                    else:
                        p_x = position_observe[layer-len(id_robot)][int(i / ((self.number_of_directions + 1) ** len(id_robot)))][0]
                        p_y = position_observe[layer-len(id_robot)][int(i / ((self.number_of_directions + 1) ** len(id_robot)))][1]
                        position_observe[layer][i] = [p_x, p_y]

        # Generate and fill in novelty tree
        layer_max = (self.path_depth - 1) * len(id_robot) + len(id_robot)
        novelty = [[[[0] for y in range(layer_max)] for x in range(layer_max)] for i in range((self.number_of_directions + 1) ** (layer_max))]

        # Every branch in the highest level has it's own unique path
        for b in range((self.number_of_directions + 1) ** layer_max):
            for y in range(1, layer_max + 1):
                for x in range(1, layer_max + 1):
                    position_x = position_observe[x][int(b / ((self.number_of_directions + 1) ** (layer_max - x)))]
                    position_y = position_observe[y][int(b / ((self.number_of_directions + 1) ** (layer_max - y)))]

                    # Fill in distance between two points
                    if x != y:
                        novelty[b][y-1][x-1] = np.sqrt((position_x[0] - position_y[0]) ** 2 + (position_x[1] - position_y[1]) ** 2)
                    else:
                        novelty[b][y-1][x-1] = np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)

        # Calculate the novelty index
        novelty = 1 - self.my_sensor_target.likelihood(novelty)

        # Fill in value tree
        for k in range(1, self.path_depth + 1):
            for j in range(len(id_robot)):
                layer = (k - 1) * len(id_robot) + j + 1
                for i in range((self.number_of_directions + 1) ** layer):

                    # If path leads outside of world
                    if (position_observe[layer][i][0] < 0) | (position_observe[layer][i][0] >= self.size_world[0]) | (position_observe[layer][i][1] < 0) | (position_observe[layer][i][1] >= self.size_world[1]):
                        value[layer][i] = 0

                    # Rate every path through marginalisation
                    else:
                        x = np.linspace(0,self.size_world[0] - 1, self.size_world[0])
                        y = np.linspace(0, self.size_world[1] - 1, self.size_world[1])
                        xy = np.meshgrid(x,y)
                        distance = np.sqrt(np.subtract(xy[0], position_observe[layer][i][0])**2 + np.subtract(xy[1], position_observe[layer][i][1])**2)
                        weight_true = np.sum(np.multiply(self.my_sensor_target.likelihood(distance), self.my_belief_target.belief_state))
                        weight_false = 1 - weight_true
                        weight_novelty = 
                        value[layer][i] = weight_true * self.kullback_leibler(self.my_belief_target.test_true(position_observe[layer][i]), self.my_belief_target.belief_state) + weight_false * self.kullback_leibler(self.my_belief_target.test_false(position_observe[layer][i]), self.my_belief_target.belief_state)

        # Sum up the value tree and store in last level
        for k in range(1, self.path_depth + 1):
            for j in range(len(id_robot)):
                layer = (k - 1) * len(id_robot) + j + 1
                for i in range((self.number_of_directions + 1) ** layer):
                    value[layer][i] = value[layer][i] + value[layer-1][int(i / (self.number_of_directions + 1))]

        # Choose path
        my_layer = (self.path_depth - 1) * len(id_robot) + len(id_robot) - id_robot.index(self.id_robot) - 1
        choice = int(np.argmax(value[-1]) / ((self.number_of_directions + 1) ** my_layer)) % self.number_of_directions

        return [choice * self.step_angle, self.step_distance]


    def kullback_leibler(self, x, y):
        result = 0
        for i_y in range(0, self.size_world[1]):
            for i_x in range(0, self.size_world[0]):
                result = result + abs(np.multiply(x[i_y][i_x], np.log(abs(np.divide(x[i_y][i_x], y[i_y][i_x])))))
        return result