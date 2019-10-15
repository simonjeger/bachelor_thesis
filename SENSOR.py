import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import copy


class sensor_target:

    def __init__(self, path, size_world, position_target):
        self.path = path
        self.position_target = position_target
        self.size_world = size_world
        self.distance_max = np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)

        # Parameters for the likelihood function
        self.cross_over = 1 / 14 * np.sqrt(self.size_world[0]**2 + self.size_world[1]**2) # circa 10
        self.width = 1 / 950 * np.sqrt(self.size_world[0]**2 + self.size_world[1]**2) # circa 0.15
        self.inclination = 30
        self.max_pos = 0.8
        self.max_neg = 0.005


    def sense(self, position_observe):
        # Depending on the distance from the observed position to the sensor give back true or false
        distance = np.sqrt((self.position_target[0] - position_observe[0]) ** 2 + (self.position_target[1] - position_observe[1]) ** 2)
        if np.random.random_sample(1) < self.likelihood(distance):
            return 1
        else:
            return 0


    def likelihood(self, distance):
        return self.max_pos - (self.max_pos - self.max_neg) * 1 / 2 * (1 + erf((np.multiply(1 / self.width, np.subtract(distance, self.cross_over))) / (self.inclination * np.sqrt(2))))


    def picture_save(self):
        # Initalize both axis
        x = np.linspace(0, self.distance_max - 1, int(self.distance_max))
        y = self.likelihood(x)

        # Plot sensor model
        plt.plot(x, y)
        plt.xlabel('Distance to target')
        plt.xlim((0, self.distance_max))
        plt.ylabel('Likelihood')
        plt.ylim((0, 1))
        plt.title('sensor_target')

        # Save picture in main folder
        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_target.png')
        plt.close()


class sensor_position:

    def __init__(self, path, size_world, id_robot, position_robot_exact):
        self.path = path
        self.size_world = size_world
        self.id_robot = id_robot
        self.position_robot_exact = position_robot_exact
        self.distance_max = np.sqrt(self.size_world[0]**2 + self.size_world[1]**2)


    def sense(self, id_robot):
        # Determine the mean of my likelihood function
        mean = np.sqrt((self.position_robot_exact[self.id_robot][0] - self.position_robot_exact[id_robot][0]) ** 2 + (self.position_robot_exact[self.id_robot][1] - self.position_robot_exact[id_robot][1]) ** 2)

        # Determine the distance vector
        distance = np.linspace(0, self.distance_max - 1, int(self.distance_max))

        # Save the PDF to create CDF
        PDF = self.likelihood(distance, mean)

        # Initialize CDF
        CDF = copy.deepcopy(PDF)
        for i in range(1, len(PDF)):
            CDF[i] = CDF[i-1] + PDF[i]

        # Make the measurement
        return self.find_nearest(CDF, np.random.random_sample(1))


    def find_nearest(self, list, value):
        list = np.asarray(list)
        idx = (np.abs(list - value)).argmin()
        return idx


    def likelihood(self, distance, mean):
        # If target in sensor_target_range
        sensor_target_range = 1 / 1.4 * np.sqrt(self.size_world[0]**2 + self.size_world[1]**2) # circa 100
        if mean < sensor_target_range:
            std = self.standard_deviation(mean)
            normal_distr =  1 / np.sqrt(2 * np.pi * std ** 2) * np.exp( - np.subtract(distance, mean) ** 2 / (2 * std ** 2))
            # Neighbour has to be somewhere
            return normal_distr / np.sum(normal_distr)

        # Otherwise
        else:
            std = 1 / 14 * np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)  # circa 10
            CDF = 1 / 2 * (1 + erf((np.subtract(distance, sensor_target_range) / (std * np.sqrt(2)))))

        return CDF / np.sum(CDF)

    def standard_deviation(self, mean):
        return 1 + mean * 5 / np.sqrt(self.size_world[0]**2 + self.size_world[1]**2) # circa mean / 30


    def picture_save(self):
        # Initalize both axis
        x = np.linspace(0, self.distance_max - 1, int(self.distance_max))

        # Plot likelyhood for n different means
        n = 5
        for i in range(0, n):
            y = self.likelihood(x, self.distance_max * i / n)
            plt.plot(x, y)

        plt.xlabel('Distance measurement of ' + str(n) + ' different distances')
        plt.xlim((0, self.distance_max))
        plt.ylabel('Likelihood')
        plt.ylim((0, 1))
        plt.title('sensor_position')

        # Save picture in main folder
        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_position.png')
        plt.close()


class sensor_motion:

    def __init__(self, path, size_world, id_robot, position_robot_exact):
        self.path = path
        self.size_world = size_world
        self.id_robot = id_robot
        self.position_robot_exact = position_robot_exact
        self.distance_max = np.sqrt(self.size_world[0]**2 + self.size_world[1]**2)


    def sense(self, angle_step_distance):

        angle = angle_step_distance[0]
        step_distance = angle_step_distance[1]

        # Determine the distance vector
        distance_angle = np.linspace(0, 2*np.pi, 360)
        distance_step_distance = np.linspace(0, self.distance_max, int(self.distance_max * 100))

        # Save the PDF to create CDF
        PDF_angle = self.gaussian_angle(1, [distance_angle], angle)[0]
        PDF_step_distance = self.gaussian_step_distance(1, [distance_step_distance], step_distance)[0]

        # Initialize CDF
        CDF_angle = copy.deepcopy(PDF_angle)
        CDF_step_distance = copy.deepcopy(PDF_step_distance)
        for i in range(1, len(PDF_angle)):
            CDF_angle[i] = CDF_angle[i-1] + PDF_angle[i]
        for i in range(1, len(PDF_step_distance)):
            CDF_step_distance[i] = CDF_step_distance[i-1] + PDF_step_distance[i]

        # Make the measurement
        new_angle = distance_angle[self.find_nearest(CDF_angle, np.random.random_sample(1))]
        new_step_distance = distance_step_distance[self.find_nearest(CDF_step_distance, np.random.random_sample(1))]

        return [new_angle, new_step_distance]


    def find_nearest(self, list, value):
        list = np.asarray(list)
        idx = (np.abs(list - value)).argmin()
        return idx


    def likelihood_robot(self, belief, distance, angle, angle_step_distance):

        # Initialize likelihood
        likelihood_distance = [[0 for i in range(self.size_world[0])] for j in range(self.size_world[1])]
        likelihood_angle = [[0 for i in range(self.size_world[0])] for j in range(self.size_world[1])]

        # Distance of likelihood
        for i_l in range(self.size_world[1]):
            for i_k in range(self.size_world[0]):
                likelihood_distance = likelihood_distance + self.gaussian_step_distance(belief[i_l][i_k], distance[i_l][i_k], angle_step_distance[1])
        likelihood_distance = likelihood_distance / np.sum(likelihood_distance)

        # Angle of likelihood
        for i_l in range(self.size_world[1]):
            for i_k in range(self.size_world[0]):
                likelihood_angle = likelihood_angle + self.gaussian_angle(belief[i_l][i_k], angle[i_l][i_k], angle_step_distance[0])
        likelihood_angle = likelihood_angle / np.sum(likelihood_angle)

        return likelihood_distance * likelihood_angle / np.sum(likelihood_distance * likelihood_angle)


    def likelihood_neighbour(self, belief, distance, angle, number_of_directions, angle_step_distance):

        # Initialize likelihood
        likelihood_distance = [[0 for i in range(self.size_world[0])] for j in range(self.size_world[1])]
        likelihood_angle = [[0 for i in range(self.size_world[0])] for j in range(self.size_world[1])]

        # Distance of likelihood
        for i_l in range(self.size_world[1]):
            for i_k in range(self.size_world[0]):
                likelihood_distance = likelihood_distance + self.gaussian_step_distance(belief[i_l][i_k], distance[i_l][i_k], angle_step_distance[1])
        likelihood_distance = likelihood_distance / np.sum(likelihood_distance)

        for i in range(number_of_directions):
            # Angle of likelihood
            for i_l in range(self.size_world[1]):
                for i_k in range(self.size_world[0]):
                    likelihood_angle = likelihood_angle + self.gaussian_angle(belief[i_l][i_k], angle[i_l][i_k], i * 2 * np.pi / number_of_directions)
        likelihood_angle = likelihood_angle / np.sum(likelihood_angle)

        return likelihood_distance * likelihood_angle / np.sum(likelihood_distance * likelihood_angle)


    def likelihood_cheap(self, belief, distance, angle_step_distance):
        # Initialize likelihood
        likelihood_distance = [[0 for i in range(self.size_world[0])] for j in range(self.size_world[1])]

        # Distance of likelihooda
        for i_l in range(self.size_world[1]):
            for i_k in range(self.size_world[0]):
                likelihood_distance = likelihood_distance + self.gaussian_step_distance(belief[i_l][i_k], distance[i_l][i_k], angle_step_distance[1])
        return likelihood_distance / np.sum(likelihood_distance)


    def gaussian_angle(self, belief, angle, mean):
        std = 0.08
        circle_difference = self.circle_difference(angle, mean)
        normal_distr = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.square(circle_difference) / (2 * std ** 2))
        return normal_distr / np.sum(normal_distr) * belief


    def circle_difference(self, angle, mean):
        result = [[1 for i in range(len(angle[0]))] for j in range(len(angle))]
        for j in range(len(angle)):
            for i in range(len(angle[0])):
                opt_1 = abs(angle[j][i] - mean)
                opt_2 = abs(angle[j][i] + 2 * np.pi - mean)
                result[j][i] = np.min([opt_1, opt_2])
        return result


    def gaussian_step_distance(self, belief, step_distance, mean):
        std = 0.08
        normal_distr = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.subtract(step_distance, mean) ** 2 / (2 * std ** 2))
        return normal_distr / np.sum(normal_distr) * belief

    def picture_save(self):
        # Initalize both axis of angle
        distance_angle = np.linspace(0, 2 * np.pi, 360)
        likelihood_angle = self.gaussian_angle(1, [distance_angle], np.pi / 2)[0]

        # Initalize both axis of step_distance
        distance_step_distance = np.linspace(0, self.distance_max - 1, int(self.distance_max * 100))
        likelihood_step_distance = self.gaussian_step_distance(1, [distance_step_distance], 5)[0]

        fig, ([stp_dist, angl]) = plt.subplots(1, 2)

        # Plot sensor model angle
        angl.plot(distance_angle, likelihood_angle)
        angl.set_xlabel('Angle')
        angl.set_xlim((0, 2 * np.pi))
        angl.set_ylim((0, 1))
        angl.set_title('sensor_motion angle')

        # Plot sensor model step_distance
        stp_dist.plot(distance_step_distance, likelihood_step_distance)
        stp_dist.set_xlabel('Step distance')
        stp_dist.set_xlim((0, 10))
        stp_dist.set_xlabel('Likelihood')
        stp_dist.set_ylim((0, 1))
        stp_dist.set_title('sensor_motion step_distance')

        # Save picture in main folder
        fig.savefig(self.path + '/sensor/' + self.path + '_sensor_motion.png')
        plt.close(fig)


class sensor_motion_cheap:

    def __init__(self, size_world, number_of_directions, step_distance):
        self.size_world = size_world
        self.step_distance = step_distance
        self.number_of_directions = number_of_directions
        self.step_angle = 2 * np.pi / self.number_of_directions


    def predict_neighbour(self, my_belief_position):
        prediction = my_belief_position

        # Determine to what positions my neighbour could move
        for j in range(self.size_world[1]):
            for i in range(self.size_world[0]):

                # Start empty and fill in
                prediction[j][i] = 0
                normalize = 0
                for h in range(self.number_of_directions):
                    add_x = int(self.step_distance * np.cos(int(h % self.number_of_directions) * self.step_angle))
                    add_y = int(self.step_distance * np.sin(int(h % self.number_of_directions) * self.step_angle))

                    if (i + add_x > 0) & (i + add_x < self.size_world[0]) & (j + add_y > 0) & (j + add_y < self.size_world[1]):
                        normalize = normalize + 1
                        prediction[j][i] = prediction[j][i] + my_belief_position[j + add_y][i + add_x]

                # If there are only a few positions to move to, then naturally they are more likely
                prediction[j][i] = prediction[j][i] / normalize

        return prediction / np.sum(prediction)