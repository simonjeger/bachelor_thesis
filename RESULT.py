import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os


class result:

    def __init__(self, path, name_of_simulation, size_world, position_target, my_robot):
        self.path = path
        self.name_of_simulation = name_of_simulation
        self.size_world = size_world
        self.position_target = position_target
        self.my_robot = my_robot
        self.picture_id = 0
        self.distance_max = np.sqrt(self.size_world[0] ** 2 + self.size_world[1] ** 2)

        self.size_point = 1 / 95 * np.sqrt(self.size_world[0]**2 + self.size_world[1]**2) # circa 1.5

        # Generate new folder
        os.mkdir(self.path + '/construction')


    def build_xy(self, belief_state):
        distance_x = np.linspace(0, self.size_world[0] - 1, self.size_world[0])
        distance_y = np.linspace(0, self.size_world[1] - 1, self.size_world[1])
        x = self.gaussian(distance_x, belief_state[0])
        y = self.gaussian(distance_y, belief_state[1])

        result = [[0 for i in range(self.size_world[0])] for j in range(self.size_world[1])]
        for j in range(self.size_world[1]):
            for i in range(self.size_world[0]):
                result[j][i] = x[i] * y[j]
        return result / np.sum(result)


    def build_rphi(self, position, belief_state):
        result = [[0 for i in range(self.size_world[0])] for j in range(self.size_world[1])]

        for j in range(self.size_world[1]):
            for i in range(self.size_world[0]):
                i_phi = np.arctan2(j - position[1], i - position[0])
                i_r = np.sqrt((i - position[0]) ** 2 + (j - position[1]) ** 2)
                phi = self.gaussian_phi(i_phi, belief_state[0])
                r = self.gaussian(i_r, belief_state[1])
                result[j][i] = r * phi
        return result / np.sum(result)


    def gaussian(self, x, mean_std):
        mean = mean_std[0]
        std = mean_std[1]
        normal_distr = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.square(np.subtract(x, mean)) / (2 * std ** 2))
        return normal_distr


    def gaussian_phi(self, x, mean_std):
        mean = mean_std[0]
        std = mean_std[1]
        difference = np.min([abs(x - mean), abs(x - 2 * np.pi - mean), abs(x + 2 * np.pi - mean)])
        normal_distr = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(- np.square(difference) / (2 * std ** 2))
        return normal_distr


    def picture_save(self):
        color_robot = 'red'
        color_neighbour = 'white'
        color_target = 'black'

        # Save picture of each step
        fig, (ax) = plt.subplots(len(self.my_robot), len(self.my_robot) + 1, figsize = (50 / 10 * (len(self.my_robot) + 1), 50 * len(self.my_robot) / 10))
        for x in range(len(self.my_robot)):

            # Actually generating the belief_position
            my_belief_position = [0] * len(self.my_robot)
            for y in range(len(self.my_robot)):
                if y == self.my_robot[x].id_robot:
                    my_belief_position[y] = self.build_xy(self.my_robot[x].my_belief_position.belief_state[y])
                else:
                    my_belief_position[y] = self.build_rphi(self.my_robot[x].position_robot_estimate[x], self.my_robot[x].my_belief_position.belief_state[y])

            # Belief_state_target
            ax[x, 0].imshow(self.my_robot[x].my_belief_target.belief_state)
            pos = patches.Circle(self.my_robot[x].position_robot_estimate[x], radius=self.size_point, color=color_robot, fill=True)
            tar = patches.Circle(self.position_target, radius=self.size_point, color=color_target, fill=True)
            ax[x, 0].add_patch(pos)
            ax[x, 0].add_patch(tar)
            ax[x, 0].set_title('Belief of robot ' + str(x) + ' about target')

            # Make connecting line when contact between robots
            for y in range(len(self.my_robot[x].position_robot_estimate)):
                contact = self.my_robot[x].id_contact[y]
                if (contact == 1) & (x != y):
                    nei_x = self.my_robot[x].position_robot_estimate[x][0] + self.my_robot[x].my_belief_position.belief_state[y][1][0] * np.cos(self.my_robot[x].my_belief_position.belief_state[y][0][0])
                    nei_y = self.my_robot[x].position_robot_estimate[x][1] + self.my_robot[x].my_belief_position.belief_state[y][1][0] * np.sin(self.my_robot[x].my_belief_position.belief_state[y][0][0])
                    lin = patches.ConnectionPatch(self.my_robot[x].position_robot_estimate[x],[nei_x, nei_y], "data", color=color_neighbour)
                    nei = patches.Circle([nei_x, nei_y], radius=self.size_point, color=color_neighbour, fill=True)
                    ax[x, 0].add_patch(lin)
                    ax[x, 0].add_patch(nei)

            # Belief_state_position
            for y in range(len(self.my_robot[x].position_robot_estimate)):
                ax[x, y + 1].imshow(my_belief_position[y])
                #ax[x, y + 1].imshow(self.my_robot[x].my_belief_position_cheap.belief_state[y])
                nei = patches.Circle((self.my_robot[x].position_robot_exact[y]), radius=self.size_point, color=color_neighbour, fill=True)
                nei_b = patches.Circle((self.my_robot[x].position_robot_exact[y]), radius=self.size_point / 5, color='black', fill=True)
                pos = patches.Circle((self.my_robot[x].position_robot_estimate[x]), radius=self.size_point, color=color_robot, fill=True)
                ax[x, y + 1].add_patch(nei)
                ax[x, y + 1].add_patch(nei_b)
                ax[x, y + 1].add_patch(pos)
                ax[x, y + 1].set_title('Belief of robot ' + str(x) + ' about position of robot ' + str(y))

        plt.savefig(self.path + '/construction/' + str(self.picture_id) + '.png')
        plt.close()
        self.picture_id = self.picture_id + 1


    def video_save(self):
        # Make video out of all the pictures, previously taken
        img_array = []

        for x in range(0,self.picture_id):
            img = cv2.imread(('./' + self.path + '/construction/' + '{0}.png').format(x))
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

            out = cv2.VideoWriter(self.path + '/video/' + self.name_of_simulation + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

        # Delete folder of pictures
        for i in range(self.picture_id):
            os.remove('./' + self.path + '/construction/' + str(i) + '.png')
        os.rmdir(self.path + '/construction')