import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import yaml


class decision:

    def __init__(self, size_world, size_world_real, my_belief_target, my_belief_position, id_robot, id_contact,
                 position_robot_estimate, my_sensor_target, path_depth, number_of_directions, step_distance):

        # Get yaml parameter
        parser = argparse.ArgumentParser()
        parser.add_argument('yaml_file')
        args = parser.parse_args()

        with open(args.yaml_file, 'rt') as fh:
            self.yaml_parameters = yaml.safe_load(fh)

        # Initialize
        self.size_world = size_world
        self.size_world_real = size_world_real
        self.scaling = size_world[0] / size_world_real[0]
        self.my_belief_target = my_belief_target
        self.my_belief_position = my_belief_position
        self.position_robot_estimate = position_robot_estimate
        self.id_robot = id_robot
        self.id_contact = id_contact
        self.my_sensor_target = my_sensor_target

        # Parameters that determine the motion of the robot
        self.path_depth = path_depth
        self.number_of_directions = number_of_directions
        self.step_distance = step_distance
        self.step_angle = 2 * np.pi / self.number_of_directions

        # Parameters that determine when to rise to the surface
        self.diving_depth = 1
        self.rise_gain_initial = 0 - self.yaml_parameters['rise_time']
        self.rise_gain = self.rise_gain_initial


    def decide_cheap(self):
        # Am I on the same depth level as the rest
        if self.id_contact[-1][0] > 0:
            self.id_contact[-1][0] = self.id_contact[-1][0] - 1
            return [0, 0]

        else:
            # Find out who is involved in the decision
            id_robot = []
            for x in range(len(self.id_contact) - 1):
                if self.id_contact[x][1] == 1:
                    id_robot = id_robot + [x]

            # Initialize all possible positions, according novelty lists and values
            position_observe = [[[0]]]
            novelty = [[0]]
            value = [[0]]
            for k in range(1, self.path_depth + 1):
                for j in range(0, len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1
                    position_observe = position_observe + [[[0, 0] for i in range((self.number_of_directions) ** (layer))]]

                    novelty = novelty + [[[[0] for i in range(layer - 1)] for j in range((self.number_of_directions) ** (layer))]]

                    value = value + [[0 for i in range((self.number_of_directions) ** (layer))]]

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
                    for i in range((self.number_of_directions) ** layer):
                        p_x = position_start[0] + self.step_distance * np.cos((i % self.number_of_directions) * self.step_angle)
                        p_y = position_start[1] + self.step_distance * np.sin((i % self.number_of_directions) * self.step_angle)
                        position_observe[layer][i] = [p_x, p_y]

            # Fill in the deeper path tree
            for k in range(2, self.path_depth + 1):
                for j in range(0, len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1

                    # Fill the in the higher levels
                    for i in range((self.number_of_directions) ** layer):
                        p_x = position_observe[layer-len(id_robot)][int(i / ((self.number_of_directions) ** len(id_robot)))][0] + self.step_distance * np.cos((i % self.number_of_directions) * self.step_angle)
                        p_y = position_observe[layer-len(id_robot)][int(i / ((self.number_of_directions) ** len(id_robot)))][1] + self.step_distance * np.sin((i % self.number_of_directions) * self.step_angle)
                        position_observe[layer][i] = [p_x, p_y]

            # Fill in novelty tree
            for k in range(1, self.path_depth + 1):
                for j in range(0, len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1

                    # This concept doesn't make sense for the first layer
                    if layer > 1:
                        for i in range((self.number_of_directions) ** layer):
                            for n in range(0, len(novelty[layer][i])):
                                p_dx = position_observe[layer][i][0] - position_observe[layer-(n+1)][int(i / ((self.number_of_directions) ** (n+1)))][0]
                                p_dy = position_observe[layer][i][1] - position_observe[layer-(n+1)][int(i / ((self.number_of_directions) ** (n+1)))][1]
                                novelty[layer][i][n] = 1 - self.my_sensor_target.likelihood(np.sqrt(p_dx ** 2 + p_dy ** 2))

                            # Store normalised novelty weight in last value
                            novelty[layer][i][-1] = np.sum(novelty[layer][i]) / (len(novelty[layer][i]))

            # Fill in value tree
            for k in range(1, self.path_depth + 1):
                for j in range(len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1
                    for i in range((self.number_of_directions) ** layer):

                        # If path leads outside of world
                        if (position_observe[layer][i][0] < 0) | (position_observe[layer][i][0] >= self.size_world[0]) | (position_observe[layer][i][1] < 0) | (position_observe[layer][i][1] >= self.size_world[1]):
                            value[layer][i] = 0

                        # Rate every path through marginalisation
                        else:
                            x = np.linspace(0,self.size_world[0] - 1, self.size_world[0])
                            y = np.linspace(0, self.size_world[1] - 1, self.size_world[1])
                            xy = np.meshgrid(x,y)
                            distance = np.sqrt(np.subtract(xy[0], position_observe[layer][i][0])**2 + np.subtract(xy[1], position_observe[layer][i][1])**2)

                            weight_true = np.sum(np.multiply(self.my_belief_target.belief_state, self.my_sensor_target.likelihood(distance)))
                            weight_false = 1 - weight_true

                            # In the first layer the novelty doesn't exist, so I set it do a default value with respect to its last know point
                            if layer == 1:
                                weight_novelty = 1 - self.my_sensor_target.likelihood(self.step_distance)

                            # After that it does
                            else:
                                weight_novelty = novelty[layer][i][-1]

                            # Filling in value tree
                            value[layer][i] = weight_novelty * (weight_true * self.kullback_leibler(self.my_belief_target.test_true(position_observe[layer][i]), self.my_belief_target.belief_state) + weight_false * self.kullback_leibler(self.my_belief_target.test_false(position_observe[layer][i]), self.my_belief_target.belief_state))

            # Sum up the value tree and store in last level
            for k in range(1, self.path_depth + 1):
                for j in range(len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1
                    for i in range((self.number_of_directions) ** layer):
                        value[layer][i] = value[layer][i] + value[layer-1][int(i / (self.number_of_directions))]

            # Update rise_gain
            if (self.yaml_parameters['rise_gain'] == 'on') & (len(self.id_contact) > 2): # It doesn't make sense for one robot to surface
                self.rise_gain = self.rise_gain + 1

                # Do not surface if you've seen the target in the past n steps
                n = self.yaml_parameters['rise_n']
                if self.rise_gain >= 0:
                    if len(self.my_belief_target.observation_log[self.id_robot]) < n:
                        for test in self.my_belief_target.observation_log[self.id_robot]:
                            if test != 'no_measurement':
                                self.rise_gain = self.rise_gain_initial
                    else:
                        for test in self.my_belief_target.observation_log[self.id_robot][-n::]:
                            if test != 'no_measurement':
                                self.rise_gain = self.rise_gain_initial

            if self.rise_gain < 0:
                # Choose path
                my_layer = (self.path_depth - 1) * len(id_robot) + len(id_robot) - id_robot.index(self.id_robot) - 1
                choice = int(np.argmax(value[-1]) / ((self.number_of_directions) ** my_layer)) % self.number_of_directions

                return [choice * self.step_angle, self.step_distance]

            else:
                # Choose to rise
                self.rise_gain = self.rise_gain_initial + 2 * self.diving_depth
                self.id_contact[-1][0] = 2 * self.diving_depth
                return [0, 0]


    def decide_expensive(self):
        # Am I on the same depth level as the rest
        if self.id_contact[-1][0] > 0:
            self.id_contact[-1][0] = self.id_contact[-1][0] - 1
            return [0, 0]

        else:
            # Find out who is involved in the decision
            id_robot = []
            for x in range(len(self.id_contact) - 1):
                if self.id_contact[x][1] == 1:
                    id_robot = id_robot + [x]

            # Initialize all trees
            position_observe = [[[0]]]
            belief_state_future = [[self.my_belief_target.belief_state]]
            weight = [[1]]
            value = [[0]]
            for k in range(1, self.path_depth + 1):
                for j in range(0, len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1
                    position_observe = position_observe + [[[0, 0] for i in range((self.number_of_directions) ** layer)]]

                    belief_state_future = belief_state_future + [[0 for i in range(2 ** layer * self.number_of_directions ** layer)]]  # because there are always two options (true or false)

                    weight = weight + [[0 for i in range(2 ** layer * self.number_of_directions ** layer)]]

                    value = value + [[0 for i in range(layer * self.number_of_directions ** layer)]]

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
                        position_start = [copy.deepcopy(self.position_robot_estimate[self.id_robot][0]) + dx,copy.deepcopy(self.position_robot_estimate[self.id_robot][1]) + dy]

                    # Fill the in the first depth level
                    for i in range((self.number_of_directions) ** layer):
                        p_x = position_start[0] + self.step_distance * np.cos((i % self.number_of_directions) * self.step_angle)
                        p_y = position_start[1] + self.step_distance * np.sin((i % self.number_of_directions) * self.step_angle)
                        position_observe[layer][i] = [p_x, p_y]

            # Fill in the deeper path tree
            for k in range(2, self.path_depth + 1):
                for j in range(0, len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1

                    # Fill the in the higher levels
                    for i in range((self.number_of_directions) ** layer):
                        p_x = position_observe[layer - len(id_robot)][int(i / ((self.number_of_directions) ** len(id_robot)))][0] + self.step_distance * np.cos((i % self.number_of_directions) * self.step_angle)
                        p_y = position_observe[layer - len(id_robot)][int(i / ((self.number_of_directions) ** len(id_robot)))][1] + self.step_distance * np.sin((i % self.number_of_directions) * self.step_angle)
                        position_observe[layer][i] = [p_x, p_y]

            # Upscale and reorder the path tree (basically convert i to p)
            for k in range(1, self.path_depth + 1):
                for j in range(0, len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1

                    y = position_observe[layer]
                    n = 2 ** (layer - 1)
                    for i in range(n - 1):
                        y = y + position_observe[layer]

                    for i in range(len(y)):
                        print(self.id_robot)
                        print(layer)
                        print(len(position_observe[layer]))
                        print(i)
                        print(int(i % self.number_of_directions) + int(i / (2 * self.number_of_directions)) * self.number_of_directions)
                        print('\n')
                        y[i] = position_observe[layer][int(i % self.number_of_directions) + int(i / (2 * self.number_of_directions * n)) * self.number_of_directions]

                    position_observe[layer] = y
            print('\n')

            # Fill in the other trees
            for k in range(1, self.path_depth + 1):
                for j in range(len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1
                    for p in range(2 ** layer * self.number_of_directions ** layer):

                        # Filling in weight tree
                        x = np.linspace(0, self.size_world[0] - 1, self.size_world[0])
                        y = np.linspace(0, self.size_world[1] - 1, self.size_world[1])
                        xy = np.meshgrid(x, y)

                        distance = np.sqrt(np.subtract(xy[0], position_observe[layer][int(p / 2)][0]) ** 2 + np.subtract(xy[1], position_observe[layer][int(p / 2)][1]) ** 2)

                        if p % 2 == 0:
                            weight[layer][p] = np.sum(np.multiply(self.my_belief_target.belief_state, self.my_sensor_target.likelihood(distance)))
                        else:
                            weight[layer][p] = np.sum(np.multiply(self.my_belief_target.belief_state, 1 - self.my_sensor_target.likelihood(distance)))


                        # Figure out all possible configurations
                        if p % 2 == 0:
                            belief_state_future[layer][p] = self.my_belief_target.test_true(position_observe[layer][int(p / 2)])

                        else:
                            belief_state_future[layer][p] = self.my_belief_target.test_false(position_observe[layer][int(p / 2)])

            # Multiply the fuck out of everything
            for k in range(1, self.path_depth + 1):
                for j in range(len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1
                    if layer > 1:
                        for p in range(2 ** layer * self.number_of_directions ** layer):
                            weight[layer][p] = weight[layer][p] * weight[layer - 1][int(p/(2 * self.number_of_directions))]
                            belief_state_future[layer][p] = np.multiply(belief_state_future[layer][p], belief_state_future[layer - 1][int(p/(2 * self.number_of_directions))])
                            belief_state_future[layer][p] = belief_state_future[layer][p] / np.sum(belief_state_future[layer][p])

            # Determine value tree
            for k in range(self.path_depth, self.path_depth + 1):
                for j in range(len(id_robot) - 1, len(id_robot)):
                    layer = (k - 1) * len(id_robot) + j + 1
                    for p in range(2 ** layer * self.number_of_directions ** layer):
                        if p % 2 == 0:
                            i = int((p / 2) % self.number_of_directions + int((p / 2) / (2 ** layer / 2 * self.number_of_directions)) * self.number_of_directions)

                            # If path leads outside of world
                            if (position_observe[layer][int(p / 2)][0] < 0) | (position_observe[layer][int(p / 2)][0] >= self.size_world[0]) | (position_observe[layer][int(p / 2)][1] < 0) | (position_observe[layer][int(p / 2)][1] >= self.size_world[1]):
                                value[layer][i] = 0
                            else:
                                value[layer][i] = value[layer][i] + weight[layer][p] * self.kullback_leibler(belief_state_future[layer][p], self.my_belief_target.belief_state) + weight[layer][p + 1] * self.kullback_leibler(belief_state_future[layer][p + 1], self.my_belief_target.belief_state)

            # Update rise_gain
            if (self.yaml_parameters['rise_gain'] == 'on') & (len(self.id_contact) > 2):  # It doesn't make sense for one robot to surface
                self.rise_gain = self.rise_gain + 1

                # Do not surface if you've seen the target in the past n steps
                n = self.yaml_parameters['rise_n']
                if self.rise_gain >= 0:
                    if len(self.my_belief_target.observation_log[self.id_robot]) < n:
                        for test in self.my_belief_target.observation_log[self.id_robot]:
                            if test != 'no_measurement':
                                self.rise_gain = self.rise_gain_initial
                    else:
                        for test in self.my_belief_target.observation_log[self.id_robot][-n::]:
                            if test != 'no_measurement':
                                self.rise_gain = self.rise_gain_initial

            if self.rise_gain < 0:
                # Choose path
                my_layer = (self.path_depth - 1) * len(id_robot) + len(id_robot) - id_robot.index(self.id_robot) - 1
                choice = int(np.argmax(value[-1]) / ((self.number_of_directions) ** my_layer)) % self.number_of_directions
                return [choice * self.step_angle, self.step_distance]

            else:
                # Choose to rise
                self.rise_gain = self.rise_gain_initial + 2 * self.diving_depth
                self.id_contact[-1][0] = 2 * self.diving_depth
                return [0, 0]


    def kullback_leibler(self, x, y):
        result = 0
        for i_y in range(0, self.size_world[1]):
            for i_x in range(0, self.size_world[0]):
                result = result + abs(np.multiply(x[i_y][i_x], np.log(abs(np.divide(x[i_y][i_x], y[i_y][i_x])))))
        return result