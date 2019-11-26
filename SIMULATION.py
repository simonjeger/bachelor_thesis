import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata
import copy
import os
import argparse
import yaml

import AGENT
import RESULT


class simulation:

    def __init__(self):

        # Get yaml parameter
        parser = argparse.ArgumentParser()
        parser.add_argument('yaml_file')
        args = parser.parse_args()

        with open(args.yaml_file, 'rt') as fh:
            self.yaml_parameters = yaml.safe_load(fh)

        # Initialize simulation parameters
        self.name_of_simulation = self.yaml_parameters['name_of_simulation']
        self.path = self.name_of_simulation
        self.size_world = self.yaml_parameters['resolution']
        self.size_world_real = self.yaml_parameters['size_world']
        self.scaling = self.size_world[0] / self.size_world_real[0]

        if self.size_world[0] / self.size_world_real[0] != self.size_world[1] / self.size_world_real[1]:
            print ('size_world and resolution need to have the same aspect ratio')

        else:
            # Initialize parameter_position
            position_initial = self.yaml_parameters['position_initial']
            for i in range(len(position_initial)):
                position_initial[i] = [int(position_initial[i][0] * self.scaling), int(position_initial[i][1] * self.scaling)]
            self.position_initial = position_initial

            # Initialize cicle dependent parameters
            self.performance_position_target = []
            self.performance_number_of_iteration = []
            self.performance_time_computation = []
            self.cicle = 0

            # Make directories
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.path + '/video', exist_ok=True)
            os.makedirs(self.path + '/sensor', exist_ok=True)
            os.makedirs(self.path + '/performance', exist_ok=True)

    def run(self):

        if self.yaml_parameters['position_target'] == 'random':
            self.position_target = [np.random.randint(self.size_world_real[0]), np.random.randint(self.size_world_real[1])]

        else:
            self.position_target = self.yaml_parameters['position_target'][self.cicle]
            self.position_target = [int(self.position_target[0] * self.scaling), int(self.position_target[1] * self.scaling)]

        # Initialize decision rate parameter
        self.d = self.yaml_parameters['deciding_rate']

        # Name accordingly to cicle
        self.name_of_simulation = self.path + '_' + str(self.cicle)

        # Initialize each robot
        self.my_robot = [0] * len(self.position_initial)
        self.my_decision = [0] * len(self.position_initial)
        for x in range(len(self.position_initial)):
            self.my_robot[x] = AGENT.bluefin(self.path, self.size_world, self.size_world_real, self.position_target, len(self.position_initial), x, self.position_initial)

        # Initialize the homebase
        self.my_homebase = AGENT.homebase(self.path, self.size_world, self.size_world_real, self.position_target, len(self.position_initial), self.position_initial)

        # Fill in position vectors
        for x in range(len(self.my_robot)):
            for y in range(len(self.my_robot)):
                self.my_robot[x].position_robot_exact[y] = self.my_robot[y].position_robot_exact[y]
                self.my_robot[x].position_robot_estimate[y] = self.my_robot[y].position_robot_estimate[y]

        # Initialize belief_position
        for x in range(len(self.position_initial)):
            for y in range(len(self.position_initial)):
                if x != y:
                    self.my_robot[x].my_belief_position.initialize_neighbour(y, self.my_robot[y].my_belief_position.belief_state[y])

        # Initialize the common result
        self.my_result = RESULT.result(self.path, self.name_of_simulation, self.size_world, self.size_world_real, self.position_target, self.my_robot, self.my_homebase)

        # Save picture of sensor_model (from the first robot), they are all the same
        self.my_robot[0].my_sensor_target.picture_save()
        self.my_robot[0].my_sensor_motion.picture_save()
        self.my_robot[0].my_sensor_distance.picture_save()

        # Save picture of the initial belief_state
        if self.yaml_parameters['visual'] == 'on':
            self.my_result.picture_save()

        # Find maximum in belief_state to know when the search is over
        belief_maximum = [0] * len(self.my_robot)
        for x in range(len(self.my_robot)):
            belief_maximum[x] = np.max(self.my_robot[x].my_belief_target.belief_state)

        # Looking for target until belief_state is accurate enough or runtime max is reached
        self.i = 0
        max_belief = self.yaml_parameters['max_belief']

        if self.yaml_parameters['max_runtime'] == '':
            max_runtime = self.my_robot[-1].range / self.yaml_parameters['step_distance']
        else:
            max_runtime = self.yaml_parameters['max_runtime'] / self.yaml_parameters['step_distance']

        while (np.max(belief_maximum) < max_belief) & (self.i < max_runtime):

            # Find new maximum in belief state to check if I'm certain enough about position of target
            belief_maximum = [0] * len(self.my_robot)
            for x in range(len(self.my_robot)):
                belief_maximum[x] = np.max(self.my_robot[x].my_belief_target.belief_state)

            # Determine which robot has the best estimate for the performance analysis
            self.performance_id_leader = np.argmax(belief_maximum)

            # Update all position estimates based on my belief
            for x in range(len(self.my_robot)):
                self.my_robot[x].update_estimate_robot()
                self.my_robot[x].update_estimate_neighbour()

            # Deside where to go next, go there and update my beliefs accordingly
            if self.d >= self.yaml_parameters['deciding_rate']:
                self.d = 1
                angle_step_distance = [0] * len(self.my_robot)
                for x in range(len(self.my_robot)):
                    # Decide on next position
                    if self.yaml_parameters['decision'] == 'cheap':
                        angle_step_distance[x] = self.my_robot[x].my_decision.decide_cheap()
                    if self.yaml_parameters['decision'] == 'expensive':
                        angle_step_distance[x] = self.my_robot[x].my_decision.decide_expensive()
                    if self.yaml_parameters['decision'] == 'lawnmower':
                        angle_step_distance[x] = self.my_robot[x].my_decision.decide_lawnmower()

            else:
                self.d = self.d + 1

            for x in range(len(self.my_robot)):
                # Actually changing the position
                self.my_robot[x].update_exact(angle_step_distance[x])

                # Update exact position on each robot about the others for sensors
                for y in range(len(self.my_robot)):
                    self.my_robot[y].position_robot_exact[x] = copy.deepcopy(self.my_robot[x].position_robot_exact[x])

            for x in range(len(self.my_robot)):
                # Update belief about my position
                self.my_robot[x].my_belief_position.update_robot(angle_step_distance[x])

                # Update position estimate of myself
                self.my_robot[x].update_estimate_robot()

                # Update belief about neighbours position
                self.my_robot[x].my_belief_position.update_neighbour()

                # Update estimated position of neighbours based on belief_position
                self.my_robot[x].update_estimate_neighbour()

                # Update belief_target
                self.my_robot[x].my_belief_target.update(self.my_robot[x].position_robot_estimate[x])

            # Update homebase_belief_state
            self.my_homebase.my_belief_position.update()

            if self.d >= self.yaml_parameters['deciding_rate'] - 1:
                # Exchange belief if close enough just one step before taking a decision
                distance_estimate = [[1 for i in range(len(self.my_robot))] for j in range(len(self.my_robot))]
                distance_exact = [[1 for i in range(len(self.my_robot))] for j in range(len(self.my_robot))]

                # Communication when in range
                for x in range(len(self.my_robot)):
                    for y in range(len(self.my_robot)):

                        # I don't have to look how far away I am from myself
                        if (x != y):

                            # Do I think we are close enough, does my neighbour think that too & are we actually close enough & isn't he rising to the surface?
                            distance_estimate[x][y] = self.my_robot[x].my_belief_position.belief_state[y][1][0]
                            distance_exact[x][y] = np.sqrt((self.my_robot[x].position_robot_exact[x][0] - self.my_robot[x].position_robot_exact[y][0]) ** 2 + (self.my_robot[x].position_robot_exact[x][1] - self.my_robot[x].position_robot_exact[y][1]) ** 2)

                            if (distance_estimate[x][y] < self.my_robot[x].communication_range_observation) & (distance_exact[x][y] < self.my_robot[x].communication_range_observation) & (self.my_robot[y].id_contact[-1][0] == 0):
                                self.my_robot[x].id_contact[y][0] = 1
                                self.my_robot[y].id_contact[x][1] = 1

                                # Update the position estimate of my neighbour about me
                                self.my_robot[y].my_belief_position.initialize_neighbour(x, self.my_robot[x].my_belief_position.belief_state[x])

                                # Merge all the logs, if they contain more information than I already know
                                for z in range(len(self.my_robot)):
                                    if len(self.my_robot[x].my_belief_target.position_log_estimate[z]) < len(self.my_robot[y].my_belief_target.position_log_estimate[z]):
                                        self.my_robot[x].my_belief_target.merge(z, self.my_robot[y].my_belief_target.position_log_estimate[z], self.my_robot[y].my_belief_target.observation_log[z])

                            # If they are not close enough to communicate, they don't have contact
                            else:
                                self.my_robot[x].id_contact[y][0] = 0
                                self.my_robot[y].id_contact[x][1] = 0

            # If everybody is connected -> delay surfacing
            everybody_is_connected = 1
            for x in self.my_robot:
                for i in range(len(x.id_contact) - 1):
                    if x.id_contact[i] != [1, 1]:
                        everybody_is_connected = 0
            if everybody_is_connected:
                for x in self.my_robot:
                    x.my_decision.rise_gain = x.my_decision.rise_gain_initial

            # Communication when at the surface
            for x in range(len(self.my_robot)):
                if self.my_robot[x].id_contact[-1][0] == self.my_robot[x].my_decision.diving_depth:
                    for z in range(len(self.my_robot)):

                        # Reset all connections
                        if x != z:
                            self.my_robot[x].id_contact[z][0] = 0
                            self.my_robot[z].id_contact[x][1] = 0

                        # Update homebase about target
                        if len(self.my_homebase.my_belief_target.position_log_estimate[z]) < len(self.my_robot[x].my_belief_target.position_log_estimate[z]):
                            self.my_homebase.my_belief_target.merge(z, self.my_robot[x].my_belief_target.position_log_estimate[z], self.my_robot[x].my_belief_target.observation_log[z])

                        # Update homebase about position
                        self.my_robot[x].my_belief_position.surface()
                        self.my_robot[x].update_estimate_robot()
                        self.my_homebase.my_belief_position.belief_state[x] = copy.deepcopy(self.my_robot[x].my_belief_position.belief_state[x])

            for x in range(len(self.my_robot)):
                if self.my_robot[x].id_contact[-1][0] == self.my_robot[x].my_decision.diving_depth:
                    for z in range(len(self.my_robot)):

                        # Update robots about target
                        if len(self.my_robot[x].my_belief_target.position_log_estimate[z]) < len(self.my_homebase.my_belief_target.position_log_estimate[z]):
                            self.my_robot[x].my_belief_target.merge(z, self.my_homebase.my_belief_target.position_log_estimate[z], self.my_homebase.my_belief_target.observation_log[z])

                    # Update the position estimate of my neighbour about me
                    for y in range(len(self.my_robot)):
                        if x != y:
                            self.my_robot[y].my_belief_position.initialize_neighbour(x, self.my_robot[x].my_belief_position.belief_state[x])

            # Increase runtime counter
            self.i = self.i + 1

            if (self.d >= self.yaml_parameters['deciding_rate']) & (self.yaml_parameters['visual'] == 'on'):
                # Safe picture of the beliefs (target & position) only everytime i take a picture
                self.my_result.picture_save()

        # Turn saved pictures into video and then delete the pictures
        if self.yaml_parameters['visual'] == 'on':
            self.my_result.video_save()

        # Update cicle dependent parameters
        self.cicle = self.cicle + 1

        arg_x = np.argmax(self.my_robot[self.performance_id_leader].my_belief_target.belief_state) % self.size_world[0]
        arg_y = np.floor(np.argmax(self.my_robot[self.performance_id_leader].my_belief_target.belief_state) / self.size_world[1])

        if self.i >= max_runtime:
            self.performance_position_target = self.performance_position_target + [[self.position_target[0], self.position_target[1], 'max_runtime']]

        elif np.sqrt((arg_x - self.position_target[0]) ** 2 + (arg_y - self.position_target[1]) ** 2) > self.yaml_parameters['max_error']:
            self.performance_position_target = self.performance_position_target + [[self.position_target[0], self.position_target[1], 'max_error']]

        else:
            self.performance_position_target = self.performance_position_target + [[self.position_target[0], self.position_target[1], 'normal']]

        self.performance_number_of_iteration = self.performance_number_of_iteration + [(self.i - 1) * self.yaml_parameters['step_distance']]

        # Get the time measurements (regardless of the depth layer) from the decision module
        for x in self.my_robot:
            self.performance_time_computation = self.performance_time_computation + x.my_decision.performance_time_computation


    def performance_target_position(self):
        # Filling performance_map
        performance_map = [[0 for i in range(self.size_world[0])] for j in range(self.size_world[1])]
        for i in range(len(self.performance_position_target)):
            performance_map[self.performance_position_target[i][1]][self.performance_position_target[i][0]] = self.performance_number_of_iteration[i]

        # data coordinates and values
        x = []
        y = []
        for i in range(len(self.performance_position_target)):
            x = x + [self.performance_position_target[i][0] / self.scaling]
            y = y + [self.performance_position_target[i][1] / self.scaling]

        z = self.performance_number_of_iteration

        # Change into numpy array (otherwise I get an error)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        # Target grid to interpolate to
        xi = np.linspace(0, self.size_world_real[0], self.size_world[0])
        yi = np.linspace(0, self.size_world_real[1], self.size_world[1])
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate
        zi = griddata((x, y), z, (xi, yi), method='nearest')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(zi, extent=[0,self.size_world_real[0],self.size_world_real[1],0])

        # Add patches
        for i in range(len(self.performance_position_target)):
            if self.performance_position_target[i][2] == 'max_error':
                tar = patches.Circle(np.divide([self.performance_position_target[i][0], self.performance_position_target[i][1]], self.scaling), radius=self.my_result.size_point, color='salmon', fill=True)
                ax.add_patch(tar)

            if self.performance_position_target[i][2] == 'max_runtime':
                tar = patches.Circle(np.divide([self.performance_position_target[i][0], self.performance_position_target[i][1]], self.scaling), radius=self.my_result.size_point, color='firebrick', fill=True)
                ax.add_patch(tar)

            if self.performance_position_target[i][2] == 'normal':
                tar = patches.Circle(np.divide([self.performance_position_target[i][0], self.performance_position_target[i][1]], self.scaling), radius=self.my_result.size_point, color='black', fill=True)
                ax.add_patch(tar)

        # Save figure
        plt.colorbar(im)
        plt.gca().set_aspect('equal', adjustable='box')
        if len(self.my_robot) > 1:
            ax.set_title('Performance analysis ' + '(' + str(len(self.position_initial)) + ' robots)' + '\n' + 'Average: ' + str(int(np.sum(self.performance_number_of_iteration) / self.cicle)) + ' over ' + str(self.cicle) + ' cicles')
        else:
            ax.set_title('Performance analysis ' + '(' + str(len(self.position_initial)) + ' robot)' + '\n' + 'Average distance: ' + str(int(np.sum(self.performance_number_of_iteration) / self.cicle)) + ' over ' + str(self.cicle) + ' cicles')
        fig.savefig(self.path + '/performance/' + self.path + '_performance_target_position.png')
        plt.close(fig)


    def performance_time(self):
        # Creating visual representation of time results as a function of their level
        average = [0] * len(self.my_robot) * self.yaml_parameters['path_depth']
        i_count = [0] * len(self.my_robot) * self.yaml_parameters['path_depth']

        ax = plt.subplot(1, 1, 1)

        for x in self.performance_time_computation:
            ax.scatter(x[0], x[1], color='blue')
            average[x[0]] = average[x[0]] + x[1]
            i_count[x[0]] = i_count[x[0]] + 1

        for j in range(len(average)):
            if average[j] != 0:
                average[j] = average[j] / i_count[j]
            else:
                average[j] = 0

        ax.set_xlim(0,len(average) + 1)
        ax.set_ylim(0,)

        subtitle = ''
        for i_count in range(1, len(average)):
            if average[i_count] != 0:
                subtitle = subtitle + '   Average ' + str(i_count) + ': ' + str(np.round(average[i_count], 8)) + '\n'
        plt.title('Performance analysis ' + '(' + str(len(self.position_initial)) + ' robots)')
        plt.text(0, 0, subtitle, bbox=dict(facecolor='white', alpha=0.5))

        # Save picture in main folder
        plt.savefig(self.path + '/performance/' + self.path + '_performance_time.png')
        plt.close()

# Initialize a simulation
my_simulation = simulation()

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()

with open(args.yaml_file, 'rt') as fh:
    yaml_parameters = yaml.safe_load(fh)

if yaml_parameters['number_of_cicles'] == '':
    for i in range(len(yaml_parameters['position_target'])):
        # Everytime I set a new random position for the target
        my_simulation.run()

        # Get information of performance over the total of all my simulations
        my_simulation.performance_target_position()
        my_simulation.performance_time()

else:
    # Warn user to use less cicles
    if yaml_parameters['number_of_cicles'] > len(yaml_parameters['position_target']):
        print('Use smaller number_of_cicles or more position_targets')

    else:
        for i in range(yaml_parameters['number_of_cicles']):
            # Everytime I set a new random position for the target
            my_simulation.run()

            # Get information of performance over the total of all my simulations
            my_simulation.performance_target_position()
            my_simulation.performance_time()