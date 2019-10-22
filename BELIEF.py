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
        self.belief_state = [[1 / (self.size_world[0] * self.size_world[1]) for i in range(self.size_world[0])] for j in range(self.size_world[1])]

        self.map_update = [0 for i in range(self.number_of_robots)]


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

    def __init__(self, id_robot, position_robot_estimate, my_sensor_distance, my_sensor_motion, number_of_robots):
        self.id_robot = id_robot
        self.my_sensor_distance = my_sensor_distance
        self.my_sensor_motion = my_sensor_motion
        self.position_robot_estimate = position_robot_estimate
        self.number_of_robots = number_of_robots
        self.belief_state = [0] * self.number_of_robots

        # Parameters for belief_position_me
        self.mean_x = self.position_robot_estimate[self.id_robot][0]
        self.std_x = 2
        self.mean_y = self.position_robot_estimate[self.id_robot][1]
        self.std_y = 2

        self.belief_state[self.id_robot] = [[self.mean_x, self.std_x], [self.mean_y, self.std_y]]

    def initialize_neighbour(self, id_robot, belief_state):
        dx = belief_state[0][0] - self.position_robot_estimate[self.id_robot][0]
        dy = belief_state[1][0] - self.position_robot_estimate[self.id_robot][1]
        mean_phi = np.arctan2(dy, dx)
        mean_r = np.sqrt(dx ** 2 + dy ** 2)

        self.belief_state[id_robot] = [[mean_phi, belief_state[0][1] / mean_r], [mean_r, belief_state[1][1]]]


    def update_robot(self, angle_step_distance):
        measurement_angle_step_distance = self.my_sensor_motion.sense(angle_step_distance)

        prior_x = self.belief_state[self.id_robot][0]
        prior_y = self.belief_state[self.id_robot][1]

        likelihood_x = self.my_sensor_motion.likelihood_x(measurement_angle_step_distance)
        likelihood_y = self.my_sensor_motion.likelihood_y(measurement_angle_step_distance)

        posterior_x = [prior_x[0] + likelihood_x[0], np.sqrt(prior_x[1] ** 2 + likelihood_x[1] ** 2)]
        posterior_y = [prior_y[0] + likelihood_y[0], np.sqrt(prior_y[1] ** 2 + likelihood_y[1] ** 2)]

        self.belief_state[self.id_robot][0] = posterior_x
        self.belief_state[self.id_robot][1] = posterior_y

    def update_neighbour(self, angle_step_distance):
        for x in range(self.number_of_robots):
            if x != self.id_robot:

                # Transform into coordinate system with new orgin
                self.transform(angle_step_distance, x)

                # Make uncertanty grow
                prior_phi = self.belief_state[x][0]
                prior_r = self.belief_state[x][1]

                velocity_vector_phi = [prior_phi[0], self.my_sensor_motion.std_move / self.belief_state[x][1][0]]
                velocity_vector_r = [prior_r[0], self.my_sensor_motion.std_move]

                new_prior_phi = [prior_phi[0], np.sqrt(prior_phi[1] ** 2 + velocity_vector_phi[1] ** 2)]
                new_prior_r = [prior_r[0], np.sqrt(prior_r[1] ** 2 + velocity_vector_r[1] ** 2)]

                # Make measurement update
                measurement = self.my_sensor_distance.sense(x)
                likelihood_r = self.my_sensor_distance.likelihood(measurement)

                new_mean_r = (new_prior_r[0] * likelihood_r[1] ** 2 + likelihood_r[0] * new_prior_r[1] ** 2) / (new_prior_r[1] ** 2 + likelihood_r[1] ** 2)
                new_std_r = np.sqrt((new_prior_r[1] ** 2 * likelihood_r[1] ** 2) / (new_prior_r[1] ** 2 + likelihood_r[1] ** 2))

                posterior_phi = new_prior_phi
                posterior_r = [new_mean_r, new_std_r]

                # Increase uncertanty because observation point is uncertain
                posterior_phi[1] = np.sqrt(posterior_phi[1] ** 2 + (self.belief_state[self.id_robot][0][1] / posterior_r[0]) ** 2)
                posterior_r[1] = np.sqrt(posterior_r[1] ** 2 + self.belief_state[self.id_robot][0][1] ** 2)

                self.belief_state[x][0] = posterior_phi
                self.belief_state[x][1] = posterior_r

    def transform(self, angle_step_distance, id_robot):
        dx_before = self.belief_state[id_robot][1][0] * np.cos(self.belief_state[id_robot][0][0])
        dy_before = self.belief_state[id_robot][1][0] * np.sin(self.belief_state[id_robot][0][0])
        dx_trans = angle_step_distance[1] * np.cos(angle_step_distance[0])
        dy_trans = angle_step_distance[1] * np.sin(angle_step_distance[0])

        dx_now = dx_before - dx_trans
        dy_now = dy_before - dy_trans

        self.mean_phi = np.arctan2(dy_now, dx_now)
        self.mean_r = np.sqrt(dx_now ** 2 + dy_now ** 2)

        self.belief_state[id_robot] = [[self.mean_phi, self.belief_state[id_robot][0][1]],[self.mean_r, self.belief_state[id_robot][1][1]]]