import numpy as np
import matplotlib.pyplot as plt


class belief_target:

    def __init__(self, size_world, my_sensor_target, number_of_robots, id_robot):
        self.size_world = size_world
        self.my_sensor_target = my_sensor_target
        self.number_of_robots = number_of_robots
        self.id_robot = id_robot

        self.position_log_estimate = [[] for i in range(self.number_of_robots)]
        self.observation_log = [[] for i in range(self.number_of_robots)]
        self.belief_state = []

        self.map_update = [0 for i in range(self.number_of_robots)]


    def initialize(self):
        self.belief_state = [[1 / (self.size_world[0] * self.size_world[1]) for i in range(self.size_world[0])] for j in range(self.size_world[1])]


    def update(self, position_next):
        # Write position in position_log
        self.position_log_estimate[self.id_robot] = self.position_log_estimate[self.id_robot] + [position_next]

        # Sensormeasurement true or false
        if self.my_sensor_target.sense(position_next):
            self.observation_log[self.id_robot] = self.observation_log[self.id_robot] + [1]
        else:
            self.observation_log[self.id_robot] = self.observation_log[self.id_robot] + [0]
        self.map_construction()


    def map_construction(self):
        for y in range(len(self.position_log_estimate)):
            for x in range(self.map_update[y], len(self.position_log_estimate[y])):
                # Distance to point of measurement
                distance = [[1 for i in range(self.size_world[0])] for j in range(self.size_world[1])]
                for i_y in range(self.size_world[1]):
                    for i_x in range(self.size_world[0]):
                        distance[i_y][i_x] = np.sqrt((i_x - self.position_log_estimate[y][x][0]) ** 2 + (i_y - self.position_log_estimate[y][x][1]) ** 2)

                if self.observation_log[y][x] == 1:
                    likelihood = self.my_sensor_target.likelihood(distance)

                else:
                    likelihood = 1 - self.my_sensor_target.likelihood(distance)

                # Prior is our current belief
                prior = self.belief_state

                # Posterior (ignore normalization for now)
                posterior = likelihood * prior

                # Update belief
                self.belief_state = posterior

                # Now we'll normalize (target must be here somewhere...)
                self.belief_state = self.belief_state / self.belief_state.sum()

                self.map_update[y] = self.map_update[y] + 1


    def merge(self, id_robot, position_log_estimate, observation_log):
        self.position_log_estimate[id_robot] = position_log_estimate
        self.observation_log[id_robot] = observation_log


    def test_true(self, position_observe):
        # Prior is our current belief
        prior = self.belief_state

        # Distance to point of measurement
        distance = [[1 for i in range(self.size_world[0])] for j in range(self.size_world[1])]
        for i_y in range(self.size_world[1]):
            for i_x in range(self.size_world[0]):
                distance[i_y][i_x] = np.sqrt((i_x - position_observe[0]) ** 2 + (i_y - position_observe[1]) ** 2)

        # Sensormeasurement true
        likelihood = self.my_sensor_target.likelihood(distance)

        # Posterior (ignore normalization for now)
        posterior = likelihood * prior

        # Update belief
        test = posterior

        # Now we'll normalize (target must be here somewhere...)
        test = test / np.sum(self.belief_state)

        return test


    def test_false(self, position_observe):
        # Prior is our current belief
        prior = self.belief_state

        # Distance to point of measurement
        distance = [[1 for i in range(self.size_world[0])] for j in range(self.size_world[1])]
        for i_y in range(self.size_world[1]):
            for i_x in range(self.size_world[0]):
                distance[i_y][i_x] = np.sqrt((i_x - position_observe[0]) ** 2 + (i_y - position_observe[1]) ** 2)

        # Sensormeasurement false
        likelihood = 1 - self.my_sensor_target.likelihood(distance)

        # Posterior (ignore normalization for now)
        posterior = likelihood * prior

        # Update belief
        test = posterior

        # Now we'll normalize (target must be here somewhere...)
        test = test / np.sum(self.belief_state)

        return test



class belief_position:

    def __init__(self, size_world, id_robot, position_robot_estimate, my_sensor_position, my_sensor_motion, number_of_robots, step_distance, number_of_directions):
        self.size_world = size_world
        self.id_robot = id_robot
        self.my_sensor_position = my_sensor_position
        self.my_sensor_motion = my_sensor_motion
        self.position_robot_estimate = position_robot_estimate
        self.number_of_robots = number_of_robots
        self.belief_state = [0] * self.number_of_robots

        self.step_distance = step_distance
        self.number_of_directions = number_of_directions

        # distance & angle
        self.distance = [[[[1 for i in range(self.size_world[0])] for j in range(self.size_world[1])] for k in range(self.size_world[1])] for l in range(self.size_world[1])]
        self.angle = [[[[1 for i in range(self.size_world[0])] for j in range(self.size_world[1])] for k in range(self.size_world[1])] for l in range(self.size_world[1])]
        for i_l in range(self.size_world[1]):
            for i_k in range(self.size_world[0]):
                for i_y in range(self.size_world[1]):
                    for i_x in range(self.size_world[0]):
                        self.distance[i_l][i_k][i_y][i_x] = np.sqrt((i_x - i_k) ** 2 + (i_y - i_l) ** 2)

                        self.angle[i_l][i_k][i_y][i_x] = np.arctan2((i_y - i_l), (i_x - i_k))


    def initialize(self, id_robot):
        # Distance to point of measurement
        distance = [[[1 for i in range(self.size_world[0])] for j in range(self.size_world[1])] for x in range(self.number_of_robots)]
        result = [0] * self.number_of_robots
        for i_y in range(self.size_world[1]):
            for i_x in range(self.size_world[0]):
                distance[id_robot][i_y][i_x] = np.sqrt((i_x - self.position_robot_estimate[id_robot][0]) ** 2 + (i_y - self.position_robot_estimate[id_robot][1]) ** 2)
        self.belief_state[id_robot] = self.normal_distribution(distance[id_robot],0)


    def update_robot(self, angle_step_distance):
        measurement_angle_step_distance = self.my_sensor_motion.sense(angle_step_distance)
        likelihood_prior = self.my_sensor_motion.likelihood_robot(self.belief_state[self.id_robot], self.distance, self.angle, measurement_angle_step_distance)
        likelihood_prior = likelihood_prior / np.sum(likelihood_prior)

        # Posterior (ignore normalization for now)
        posterior = likelihood_prior

        # Update belief
        self.belief_state[self.id_robot] = posterior

        # Now we'll normalize (target must be here somewhere...)
        self.belief_state[self.id_robot] = self.belief_state[self.id_robot] / self.belief_state[self.id_robot].sum()

    def update_neighbour(self, angle_step_distance):
        for x in range(self.number_of_robots):

            # Determine likelihood * prior
            if x != self.id_robot:
                # Estimate robots position
                #likelihood_prior = self.my_sensor_motion.likelihood_neighbour(self.belief_state[x], self.distance, self.angle, self.number_of_directions, angle_step_distance)
                likelihood_prior = self.my_sensor_motion.likelihood_cheap(self.belief_state[x], self.distance, angle_step_distance)
                likelihood_prior = likelihood_prior / np.sum(likelihood_prior)

                # Sensormeasurement for distance measurement
                measurement = self.my_sensor_position.sense(x)

                # Measurement of distance to neighbour and updating the likelihood_prior
                likelihood_prior = self.my_sensor_position.likelihood(self.distance[int(self.position_robot_estimate[self.id_robot][1])][int(self.position_robot_estimate[self.id_robot][0])], measurement) * likelihood_prior
                likelihood_prior = likelihood_prior / np.sum(likelihood_prior)

                # Posterior (ignore normalization for now)
                posterior = likelihood_prior

                # Update belief
                self.belief_state[x] = posterior

                # Now we'll normalize (target must be here somewhere...)
                self.belief_state[x] = self.belief_state[x] / self.belief_state[x].sum()


    def normal_distribution(self, distance, mean):
        # This is only used for the initial position
        std = 1
        normal_distr =  1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.subtract(distance, mean) ** 2 / (2 * std ** 2))
        # Robot has to be somewhere
        return normal_distr / np.sum(normal_distr)


class belief_position_cheap:

    def __init__(self, size_world, my_sensor_position, my_sensor_motion_cheap, number_of_robots, id_robot, position_robot_exact):
        self.size_world = size_world
        self.my_sensor_position = my_sensor_position
        self.my_sensor_motion_cheap = my_sensor_motion_cheap
        self.id_robot = id_robot
        self.position_robot_exact = position_robot_exact
        self.number_of_robots = number_of_robots
        self.belief_state = [0] * self.number_of_robots


    def initialize(self, id_robot):
        # Distance to point of measurement
        distance = [[[1 for i in range(self.size_world[0])] for j in range(self.size_world[1])] for x in range(self.number_of_robots)]
        result = [0] * self.number_of_robots
        for i_y in range(self.size_world[1]):
            for i_x in range(self.size_world[0]):
                distance[id_robot][i_y][i_x] = np.sqrt((i_x - self.position_robot_exact[id_robot][0]) ** 2 + (i_y - self.position_robot_exact[id_robot][1]) ** 2)
        self.belief_state[id_robot] = self.normal_distribution(distance[id_robot],0)


    def normal_distribution(self, distance, mean):
        # This is only used for the initial position
        std = 1
        normal_distr =  1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.subtract(distance, mean) ** 2 / (2 * std ** 2))
        # Neighbour has to be somewhere
        return normal_distr / np.sum(normal_distr)


    def update(self, angle_step_distance):
        position_robot = [self.position_robot_exact[self.id_robot][0] + angle_step_distance[1] * np.cos(angle_step_distance[0]), self.position_robot_exact[self.id_robot][1] + angle_step_distance[1] * np.sin(angle_step_distance[0])]
        for x in range(self.number_of_robots):
            # Prior is our current belief
            prior = self.my_sensor_motion_cheap.predict_neighbour(self.belief_state[x])

            # Distance to point of measurement
            distance = [[1 for i in range(self.size_world[0])] for j in range(self.size_world[1])]
            for i_y in range(self.size_world[1]):
                for i_x in range(self.size_world[0]):
                    distance[i_y][i_x] = np.sqrt((i_x - position_robot[0]) ** 2 + (i_y - position_robot[1]) ** 2)

            # Sensormeasurement
            measurement = self.my_sensor_position.sense(x)

            # Determine likelihood
            likelihood = self.my_sensor_position.likelihood(distance, measurement)

            # Posterior (ignore normalization for now)
            posterior = likelihood * prior

            # Update belief
            self.belief_state[x] = posterior

            # Now we'll normalize (target must be here somewhere...)
            self.belief_state[x] = self.belief_state[x] / self.belief_state[x].sum()