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

        # Initialize all possible positions and according values
        position_observe = []
        value = []
        for k in range(0, self.path_depth + 1):
            for j in range(1,len(id_robot) + 1):
                position_observe = position_observe + [[[0] * len(id_robot) for i in range((self.number_of_directions + 1) ** (k*j))]]
                value = value + [[0 for i in range((self.number_of_directions + 1) ** (k*j))]]

        # Determine start point & start value of path tree
        for j in range(0, len(id_robot)):
            position_observe[0][0][j] = copy.deepcopy(self.position_robot_estimate[id_robot[j]])
        value[0][0] = 0

        # Fill in path tree
        for k in range(1, self.path_depth + 1):
            for j in range(1, len(id_robot) + 1):
                for i in range((self.number_of_directions + 1) ** (k*j)):
                    if (i % (self.number_of_directions) != 0) | (i == 0):
                        p_x = position_observe[k*j-1][int(i / (self.number_of_directions + 1))][j-1][0] + self.step_distance * np.cos((i % self.number_of_directions) * self.step_angle)
                        p_y = position_observe[k*j-1][int(i / (self.number_of_directions + 1))][j-1][1] + self.step_distance * np.sin((i % self.number_of_directions) * self.step_angle)
                        position_observe[k*j][i][j-1] = [p_x, p_y]
                    else:
                        p_x = position_observe[k*j-1][int(i / (self.number_of_directions + 1))][j-1][0]
                        p_y = position_observe[k*j-1][int(i / (self.number_of_directions + 1))][j-1][1]
                        position_observe[k*j][i][j-1] = [p_x, p_y]

        # Fill in value tree
        for k in range(1, self.path_depth + 1):
            for j in range(0, len(id_robot) + 1):
                for i in range((self.number_of_directions + 1) ** (k*j)):

                    # If path leads outside of world
                    if (position_observe[k*j][i][j-1][0] < 0) | (position_observe[k*j][i][j-1][0] >= self.size_world[0]) | (position_observe[k*j][i][j-1][1] < 0) | (position_observe[k*j][i][j-1][1] >= self.size_world[1]):
                        value[k*j][i] = 0

                    # Rate every path through marginalisation
                    else:
                        x = np.linspace(0,self.size_world[0] - 1, self.size_world[0])
                        y = np.linspace(0, self.size_world[1] - 1, self.size_world[1])
                        xy = np.meshgrid(x,y)
                        distance = np.sqrt(np.subtract(xy[0], position_observe[k*j][i][j-1][0])**2+np.subtract(xy[1], position_observe[k*j][i][j-1][1])**2)
                        weight_true = np.sum(np.multiply(self.my_sensor_target.likelihood(distance), self.my_belief_target.belief_state))
                        weight_false = 1 - weight_true
                        value[k*j][i] = weight_true * self.kullback_leibler(self.my_belief_target.test_true(position_observe[k*j][i][j-1]), self.my_belief_target.belief_state) + weight_false * self.kullback_leibler(self.my_belief_target.test_false(position_observe[k*j][i][j-1]), self.my_belief_target.belief_state)

        # Sum up the value tree and store in last level
        for k in range(1, self.path_depth + 1):
            for j in range(0, len(id_robot) + 1):
                for i in range((self.number_of_directions + 1)** (k * j)):
                    value[k*j][i] = value [k*j][i] + value[k*j-1][int(i / (self.number_of_directions + 1))]

        # Choose path
        choice = int(np.argmax(value[-1]) / ((self.number_of_directions + 1) ** ((self.path_depth - 1) * len(id_robot))))

        return [choice * self.step_angle, self.step_distance]


    def kullback_leibler(self, x, y):
        result = 0
        for i_y in range(0, self.size_world[1]):
            for i_x in range(0, self.size_world[0]):
                result = result + abs(np.multiply(x[i_y][i_x], np.log(abs(np.divide(x[i_y][i_x], y[i_y][i_x])))))
        return result