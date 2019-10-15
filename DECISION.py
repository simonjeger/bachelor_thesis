import numpy as np
import copy


class decision:

    def __init__(self, size_world, my_belief_target, my_belief_position, id_robot, id_contact, position_robot_exact, position_robot_estimate, my_sensor_target, path_depth, number_of_directions, step_distance):
        self.size_world = size_world
        self.my_belief_target = my_belief_target
        self.my_belief_position = my_belief_position #under construction
        self.position_robot_estimate = position_robot_estimate
        self.id_robot = id_robot
        self.position_robot_exact = position_robot_exact
        self.id_contact = id_contact
        self.my_sensor_target = my_sensor_target

        # Parameters that determine the motion of the robot
        self.path_depth = path_depth
        self.number_of_directions = number_of_directions
        self.step_distance = step_distance
        self.step_angle = 2 * np.pi / self.number_of_directions


    def decide(self):
        # Initialize all possible positions and according values
        position_observe = []
        value = []
        for j in range(0, self.path_depth + 1):
            position_observe = position_observe + [[[0, 0] for i in range(self.number_of_directions ** j)]]
            value = value + [[0 for i in range(self.number_of_directions ** j)]]

        # Determine start point & start value of path tree
        position_observe[0][0][0] = copy.deepcopy(self.position_robot_estimate[self.id_robot][0])
        position_observe[0][0][1] = copy.deepcopy(self.position_robot_estimate[self.id_robot][1])
        value[0][0] = 0

        # Fill in path tree
        for j in range(1, self.path_depth + 1):
            for i in range(self.number_of_directions ** j):
                position_observe[j][i][0] = position_observe[j-1][int(i / self.number_of_directions)][0] + self.step_distance * np.cos(int(i % self.number_of_directions) * self.step_angle)
                position_observe[j][i][1] = position_observe[j-1][int(i / self.number_of_directions)][1] + self.step_distance * np.sin(int(i % self.number_of_directions) * self.step_angle)

        # Fill in value tree
        for j in range(1, self.path_depth + 1):
            for i in range(self.number_of_directions ** j):

                # If path leads outside of world
                if (position_observe[j][i][0] < 0) | (position_observe[j][i][0] >= self.size_world[0]) | (position_observe[j][i][1] < 0) | (position_observe[j][i][1] >= self.size_world[1]):
                    value[j][i] = 0

                # Rate every path through marginalisation
                else:
                    x = np.linspace(0,self.size_world[0] - 1, self.size_world[0])
                    y = np.linspace(0, self.size_world[1] - 1, self.size_world[1])
                    xy = np.meshgrid(x,y)
                    distance = np.sqrt(np.subtract(xy[0], position_observe[j][i][0])**2+np.subtract(xy[1], position_observe[j][i][1])**2)
                    weight_true = np.sum(np.multiply(self.my_sensor_target.likelihood(distance), self.my_belief_target.belief_state))
                    weight_false = 1 - weight_true
                    value[j][i] = weight_true * self.kullback_leibler(self.my_belief_target.test_true(position_observe[j][i]), self.my_belief_target.belief_state) + weight_false * self.kullback_leibler(self.my_belief_target.test_false(position_observe[j][i]), self.my_belief_target.belief_state)

        # Sum up the value tree and store in last level
        for j in range(1, self.path_depth + 1):
            for i in range(self.number_of_directions ** j):
                value[j][i] = value [j][i] + value[j - 1][int(i / self.number_of_directions)]

        # Choose path
        choice = int(np.argmax(value[-1]) / (self.number_of_directions ** (self.path_depth - 1)))

        return [choice * self.step_angle, self.step_distance]


    def kullback_leibler(self, x, y):
        result = 0
        for i_y in range(0, self.size_world[1]):
            for i_x in range(0, self.size_world[0]):
                result = result + abs(np.multiply(x[i_y][i_x], np.log(abs(np.divide(x[i_y][i_x], y[i_y][i_x])))))
        return result