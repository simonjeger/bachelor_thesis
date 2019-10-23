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
        #self.width = 1 / 1250 * np.sqrt(self.size_world[0]**2 + self.size_world[1]**2) # circa 0.11
        self.width = 1 / 1250 * np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)  # circa 0.11
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


class sensor_motion:

    def __init__(self, path, size_world, step_distance):
        self.path = path
        self.size_world = size_world
        self.distance_max = np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)

        # Parameters for the likelihood function
        self.std_v = 0.5
        self.std_move = self.std_prob(0.9) * step_distance

    def std_prob(self,certainty):
        # This function determines what the factor to the step_distance should be, so that it contains a certan certainty
        x = np.linspace(-1000, 999, 1000)
        std = [0]
        mean = 500
        distance_min_max = [0]

        for i in range(1, 800):
            std = std + [i]
            cdf = self.cdf(x, [mean, std[-1]])

            distance_min_max = distance_min_max + [x[self.find_nearest(cdf, certainty)] - x[self.find_nearest(cdf, 1 - certainty)]]

        factor = (distance_min_max[200] - distance_min_max[0]) / (std[200] - std[0])
        return factor

    def sense(self, angle_step_distance):
        # My measurement will be a bit off the true x and y position
        distance_x = np.linspace(- np.max(self.size_world), np.max(self.size_world) - 1, 100 * np.max(self.size_world))
        distance_y = np.linspace(- np.max(self.size_world), np.max(self.size_world) - 1, 100 * np.max(self.size_world))

        likelihood_x = self.likelihood_x(angle_step_distance)
        likelihood_y = self.likelihood_y(angle_step_distance)

        cdf_x = self.cdf(distance_x, likelihood_x)
        cdf_y = self.cdf(distance_y, likelihood_y)

        idx_x = self.find_nearest(cdf_x, np.random.random_sample(1))
        idx_y = self.find_nearest(cdf_y, np.random.random_sample(1))

        result_x = distance_x[idx_x]
        result_y = distance_y[idx_y]

        result_phi = np.arctan2(result_y, result_x)
        result_r = np.sqrt(result_x ** 2 + result_y ** 2)

        return [result_phi, result_r]


    def cdf(self, x, mean_std):
        mean = mean_std[0]
        std = mean_std[1]
        cdf = 1 / 2 * (1 + erf((np.subtract(x, mean) / (std * np.sqrt(2)))))
        return cdf


    def find_nearest(self, list, value):
        list = np.asarray(list)
        idx = (np.abs(list - value)).argmin()
        return idx


    def likelihood_x(self, angle_step_distance):
        likelihood_x = [angle_step_distance[1] * np.cos(angle_step_distance[0]), self.std_v]
        return likelihood_x


    def likelihood_y(self, angle_step_distance):
        likelihood_y = [angle_step_distance[1] * np.sin(angle_step_distance[0]), self.std_v]
        return likelihood_y


    def gaussian(self, x, mean_std):
        mean = mean_std[0]
        std = mean_std[1]
        normal_distr = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.square(np.subtract(x, mean)) / (2 * std ** 2))
        return normal_distr


    def picture_save(self):
        # Initalize both axis
        x = np.linspace(0, self.distance_max - 1, int(self.distance_max))
        y = self.gaussian(x, self.likelihood_x([0, 5]))

        # Plot sensor model
        plt.plot(x, y)
        plt.xlabel('Change in position in e_x')
        plt.xlim((0, self.distance_max))
        plt.ylabel('Likelihood')
        plt.ylim((0, 1))
        plt.title('sensor_motion')

        # Save picture in main folder
        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_motion.png')
        plt.close()


class sensor_distance:

    def __init__(self, path, size_world, id_robot, position_robot_exact):
        self.path = path
        self.size_world = size_world
        self.id_robot = id_robot
        self.position_robot_exact = position_robot_exact
        self.distance_max = np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)


    def sense(self, id_robot):
        mean = np.sqrt((self.position_robot_exact[self.id_robot][0] - self.position_robot_exact[id_robot][0]) ** 2 + (self.position_robot_exact[self.id_robot][1] - self.position_robot_exact[id_robot][1]) ** 2)
        likelihood = self.likelihood(mean)
        distance = np.linspace(0, self.distance_max - 1, int(self.distance_max))
        cdf = self.cdf(distance, likelihood)

        # Make the measurement
        return self.find_nearest(cdf, np.random.random_sample(1))


    def likelihood(self, mean):
        std = self.standard_deviation(mean)
        return [mean, std]


    def standard_deviation(self, mean):
        return 1 + mean * 5 / np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)  # circa mean / 30


    def cdf(self, x, mean_std):
        mean = mean_std[0]
        std = mean_std[1]
        cdf = 1 / 2 * (1 + erf((np.subtract(x, mean) / (std * np.sqrt(2)))))
        return cdf


    def find_nearest(self, list, value):
        list = np.asarray(list)
        idx = (np.abs(list - value)).argmin()
        return idx


    def gaussian(self, x, mean_std):
        mean = mean_std[0]
        std = mean_std[1]
        normal_distr = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.square(np.subtract(x, mean)) / (2 * std ** 2))
        return normal_distr


    def picture_save(self):
        # Initalize both axis
        x = np.linspace(0, self.distance_max - 1, int(self.distance_max))

        # Plot likelyhood for n different means
        n = 5
        for i in range(0, n):
            y = self.gaussian(x,self.likelihood(self.distance_max * i / n))
            plt.plot(x, y)

        plt.xlabel('Distance between robots')
        plt.xlim((0, self.distance_max))
        plt.ylabel('Likelihood')
        plt.ylim((0, 1))
        plt.title('sensor_distance')

        # Save picture in main folder
        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_distance.png')
        plt.close()