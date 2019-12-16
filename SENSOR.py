import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import argparse
import yaml


class sensor_target_boolean:

    def __init__(self, path, size_world, size_world_real, position_target):

        # Get yaml parameter
        parser = argparse.ArgumentParser()
        parser.add_argument('yaml_file')
        args = parser.parse_args()

        with open(args.yaml_file, 'rt') as fh:
            self.yaml_parameters = yaml.safe_load(fh)

        # Initialize
        self.path = path
        self.position_target = position_target
        self.size_world = size_world
        self.size_world_real = size_world_real
        self.scaling = size_world[0] / size_world_real[0]

        # Parameters for the likelihood function
        self.cross_over = self.yaml_parameters['cross_over'] * self.scaling
        self.alpha = self.yaml_parameters['alpha']
        self.beta = self.yaml_parameters['beta']
        self.gamma = self.yaml_parameters['gamma']


    def sense(self, position_observe):
        # Depending on the distance from the observed position to the sensor give back true or false
        distance = np.sqrt((self.position_target[0] - position_observe[0]) ** 2 + (self.position_target[1] - position_observe[1]) ** 2)
        if np.random.random_sample(1) < self.likelihood(distance):
            return 1
        else:
            return 'no_measurement'

    def likelihood(self, distance):
        return self.alpha - (self.alpha + self.beta - 1) / (1 + np.exp(-self.gamma*(np.subtract(distance, self.cross_over))))


    def picture_save(self):
        # Initalize both axis
        x = np.linspace(0, 2 * self.cross_over, 1000)
        y = self.likelihood(x)

        # Plot sensor model
        fig = plt.figure(figsize=np.multiply([3,1], 3))
        ax = fig.add_subplot(111)

        ax.plot(x / self.scaling, y)
        plt.xlabel('Distance to target [m]')
        plt.xlim((0, 2 * self.cross_over / self.scaling))
        plt.ylabel('Likelihood')
        plt.ylim((0, 1))
        plt.title('sensor_target_detection')

        # Save picture in main folder
        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_target.pdf')
        plt.close()



class sensor_target_angle:

    def __init__(self, path, size_world, size_world_real, position_target):

        # Get yaml parameter
        parser = argparse.ArgumentParser()
        parser.add_argument('yaml_file')
        args = parser.parse_args()

        with open(args.yaml_file, 'rt') as fh:
            self.yaml_parameters = yaml.safe_load(fh)

        # Initialize
        self.path = path
        self.position_target = position_target
        self.size_world = size_world
        self.size_world_real = size_world_real
        self.scaling = size_world[0] / size_world_real[0]

        # Parameters for the likelihood function
        self.cross_over = self.yaml_parameters['cross_over'] * self.scaling
        self.alpha = self.yaml_parameters['alpha']
        self.beta = self.yaml_parameters['beta']
        self.gamma = self.yaml_parameters['gamma']


    def sense(self, position_observe):
        # Depending on the distance from the observed position to the sensor give back true or false
        distance = np.sqrt((self.position_target[0] - position_observe[0]) ** 2 + (self.position_target[1] - position_observe[1]) ** 2)
        angle = np.arctan2(self.position_target[1] - position_observe[1], self.position_target[0] - position_observe[0])
        distance_cdf = np.linspace(- 2 * np.pi, 2 * np.pi, 1000)

        if np.random.random_sample(1) < self.likelihood(distance):
            angle_cdf = self.angle_cdf(angle, distance_cdf, distance)
            measurement = distance_cdf[self.find_nearest(angle_cdf, np.random.random_sample(1))]
            return measurement

        else:
            return 'no_measurement'

    def likelihood(self, distance):
        return self.alpha - (self.alpha + self.beta - 1) / (1 + np.exp(-self.gamma*(np.subtract(distance, self.cross_over))))

    def likelihood_angle(self, angle_relativ):
        # std is independent of distance
        std = (self.std_angle / self.cross_over)
        normal_distr = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.square(angle_relativ) / (2 * std ** 2))
        return normal_distr

    def angle_cdf(self, angle, x, distance):
        # std gets normed by the distance
        std = (self.std_angle / distance) * (self.likelihood(self.cross_over) / self.likelihood(distance)) ** 3 # When far away -> random measurement
        cdf = 1 / 2 * (1 + erf((np.subtract(x, angle) / (std * np.sqrt(2)))))
        return cdf

    def find_nearest(self, list, value):
        list = np.asarray(list)
        idx = (np.abs(list - value)).argmin()
        return idx

    def picture_save(self):
        # Initalize both axis
        x1 = np.linspace(0, 2 * self.cross_over, 1000)
        x2 = np.linspace(- np.pi, np.pi, 1000)
        y1 = self.likelihood(x1)
        y2 = self.likelihood_angle(x2)

        # Plot sensor model
        fig = plt.figure(figsize=np.multiply([3,2.5], 3))
        ax1 = fig.add_subplot(211)

        ax1.plot(x1 / self.scaling, y1)
        plt.xlabel('Distance to target [m]')
        plt.xlim(0, 2 * self.cross_over / self.scaling)
        plt.ylabel('Likelihood')
        plt.ylim(0, 1)
        plt.title('sensor_target_detection')

        ax2 = fig.add_subplot(212)

        ax2.plot(x2 / self.scaling, y2)
        plt.xlabel('Distance to target [m]')
        plt.xlim(- np.pi, np.pi)
        plt.ylabel('Likelihood')
        plt.ylim(0, 1)
        plt.title('sensor_target_bearing')

        # Save picture in main folder
        ratio = 0.3
        x1left, x1right = ax1.get_xlim()
        y1bottom, y1top = ax1.get_ylim()
        x2left, x2right = ax2.get_xlim()
        y2bottom, y2top = ax2.get_ylim()
        ax1.set_aspect(abs((x1right - x1left) / (y1bottom - y1top)) * ratio)
        ax2.set_aspect(abs((x2right - x2left) / (y2bottom - y2top)) * ratio)

        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_target.pdf')
        plt.close()



class sensor_motion:

    def __init__(self, path, size_world, size_world_real, step_distance):

        # Get yaml parameter
        parser = argparse.ArgumentParser()
        parser.add_argument('yaml_file')
        args = parser.parse_args()

        with open(args.yaml_file, 'rt') as fh:
            self.yaml_parameters = yaml.safe_load(fh)

        # Inizialize
        self.path = path
        self.size_world = size_world
        self.size_world_real = size_world_real
        self.scaling = size_world[0] / size_world_real[0]
        self.step_distance = step_distance

        # Parameters for the likelihood function
        self.std_v = self.yaml_parameters['std_v'] * np.sqrt(self.step_distance)
        self.std_move = self.gaussian_approx() * np.sqrt(self.step_distance)


    def gaussian_approx(self):
        step_distance_norm = 1
        std_v_norm = self.yaml_parameters['std_v']
        m = [- step_distance_norm, step_distance_norm]
        s = [std_v_norm, std_v_norm]
        p = self.gaussian(m, [m, s])
        p_n = p / np.sum(p)

        m_new = p_n.dot(m)
        s_new = np.sqrt(p_n.dot(np.power(s, 2) + np.power(m, 2)) - np.power(p_n.dot(m), 2))

        return s_new


    def sense(self, angle_step_distance):
        # My measurement will be a bit off the true x and y position
        likelihood_x = self.likelihood_x(angle_step_distance)
        likelihood_y = self.likelihood_y(angle_step_distance)

        distance_x = np.linspace(likelihood_x[0] - 10 * likelihood_x[1], likelihood_x[0] + 10 * likelihood_x[1], 1000)
        distance_y = np.linspace(likelihood_y[0] - 10 * likelihood_y[1], likelihood_y[0] + 10 * likelihood_y[1], 1000)

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
        normal_distr = 1 / np.sqrt(2 * np.pi * np.power(std, 2)) * np.exp(- np.square(np.subtract(x, mean)) / (2 * np.power(std, 2)))
        return normal_distr


    def picture_save(self):
        # Displacement
        x = np.linspace(0, 2 * self.step_distance, 1000)
        y = self.gaussian(x, self.likelihood_x([0, self.step_distance]))

        # Plot sensor model
        fig = plt.figure(figsize=np.multiply([3,1], 3))
        ax = fig.add_subplot(111)

        ax.plot(x / self.scaling, y * self.scaling)
        plt.xlabel('Change in position in e_x [m]')
        plt.xlim((0, 2 * self.step_distance / self.scaling))
        plt.ylabel('Likelihood')
        #plt.ylim((0, 1))
        plt.title('sensor_motion_displacement')

        # Save picture in main folder
        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_motion_displacement.pdf')
        plt.close()

        # Approximation
        x = np.linspace(- 2, 2, 10000)
        y_0 = self.gaussian(x, [- 1, self.std_v / np.sqrt(self.step_distance)])
        y_1 = self.gaussian(x, [1, self.std_v / np.sqrt(self.step_distance)])
        y_a = self.gaussian(x, [0, self.std_move / np.sqrt(self.step_distance)])

        # Plot sensor model
        fig = plt.figure(figsize=np.multiply([3,1], 3))
        ax = fig.add_subplot(111)

        ax.plot(x, y_0 + y_1)
        ax.plot(x, y_a)
        plt.xlabel('Change in position in e_x [m]')
        plt.xlim((- 2, 2))
        plt.ylabel('Likelihood')
        plt.ylim((0, 1))
        plt.title('sensor_motion_approximation')

        # Save picture in main folder
        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_motion_approximation.pdf')
        plt.close()


class sensor_distance:

    def __init__(self, path, size_world, size_world_real, communication_range_neighbour, id_robot, position_robot_exact):

        # Get yaml parameter
        parser = argparse.ArgumentParser()
        parser.add_argument('yaml_file')
        args = parser.parse_args()

        with open(args.yaml_file, 'rt') as fh:
            self.yaml_parameters = yaml.safe_load(fh)

        # Initialize
        self.path = path
        self.size_world = size_world
        self.size_world_real = size_world_real
        self.scaling = size_world[0] / size_world_real[0]
        self.communication_range_neighbour = communication_range_neighbour
        self.id_robot = id_robot
        self.position_robot_exact = position_robot_exact
        self.distance_max = np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)


    def sense(self, id_robot):
        mean = np.sqrt((self.position_robot_exact[self.id_robot][0] - self.position_robot_exact[id_robot][0]) ** 2 + (self.position_robot_exact[self.id_robot][1] - self.position_robot_exact[id_robot][1]) ** 2)
        likelihood = self.likelihood(mean)
        distance = np.linspace(0.000001, self.distance_max, 10 * int(self.distance_max))
        cdf = self.cdf(distance, likelihood)

        # Make the measurement
        measurement = distance[self.find_nearest(cdf, np.random.random_sample(1))]

        if measurement < self.communication_range_neighbour:
            return measurement
        else:
            return 'no_measurement'


    def likelihood(self, mean):
        std = self.standard_deviation(mean)
        return [mean, std]


    def standard_deviation(self, mean):
        return self.yaml_parameters['std_const'] * self.scaling + mean * self.yaml_parameters['std_mean']


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
        x = np.linspace(0, 2 * self.communication_range_neighbour, 1000)

        # Plot likelyhood for n different means
        fig = plt.figure(figsize=np.multiply([3,1], 3))
        ax = fig.add_subplot(111)
        n = 5
        for i in range(0, n):
            y = self.gaussian(x, self.likelihood(self.communication_range_neighbour * i / n))
            ax.plot(x / self.scaling, y * self.scaling)

        plt.xlabel('Distance between robots [m]')
        plt.xlim((0, 2 * self.communication_range_neighbour / self.scaling))
        plt.ylabel('Likelihood')
        #plt.ylim((0, 1))
        plt.title('sensor_distance')

        # Save picture in main folder
        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

        plt.savefig(self.path + '/sensor/' + self.path + '_sensor_distance.pdf')
        plt.close()