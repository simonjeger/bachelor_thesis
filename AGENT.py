import numpy as np
import copy

import SENSOR
import BELIEF
import DECISION


class auv:

    def __init__(self, path, size_world, size_world_real, position_target, number_of_robots, id_robot, position_initial):

        # Initialize
        self.path = path
        self.size_world = size_world
        self.size_world_real = size_world_real
        self.scaling = size_world[0] / size_world_real[0]
        self.position_target = position_target
        self.number_of_robots = number_of_robots
        self.id_robot = id_robot
        self.id_contact = [[0, 0] for i in range(number_of_robots + 1)] # [Do I have contact to him, does he has contact to me]
        self.id_contact[self.id_robot] = [1, 1]
        self.position_robot_exact = copy.deepcopy(position_initial)
        self.position_robot_estimate = copy.deepcopy(position_initial)

        # Initialize parameters for motion of robot
        #self.step_distance = 1.4 * self.scaling
        self.step_distance = 1000 * self.scaling
        self.number_of_directions = 8
        self.path_depth = 1
        self.communication_range_observation = 5000 * self.scaling
        self.communication_range_neighbour = 9500 * self.scaling

        #self.my_sensor_target = SENSOR.sensor_target_boolean(self.path, self.size_world, self.size_world_real, self.position_target)
        self.my_sensor_target = SENSOR.sensor_target_angle(self.path, self.size_world, self.size_world_real, self.position_target)
        self.my_sensor_motion = SENSOR.sensor_motion(self.path, self.size_world, self.size_world_real, self.step_distance)
        self.my_sensor_distance = SENSOR.sensor_distance(self.path, self.size_world, self.size_world_real, self.communication_range_neighbour, self.id_robot, self.position_robot_exact)

        #self.my_belief_target = BELIEF.belief_target_boolean(self.size_world, self.my_sensor_target, self.number_of_robots, self.id_robot, self.id_contact)
        self.my_belief_target = BELIEF.belief_target_angle(self.size_world, self.my_sensor_target, self.number_of_robots, self.id_robot, self.id_contact)
        self.my_belief_position = BELIEF.belief_position(self.id_robot, self.position_robot_exact, self.position_robot_estimate, self.my_sensor_distance, self.my_sensor_motion, self.number_of_robots)

        self.my_decision = DECISION.decision(self.size_world, self.my_belief_target, self.my_belief_position, self.id_robot, self.id_contact, self.position_robot_estimate, self.my_sensor_target, self.path_depth, self.number_of_directions, self.step_distance)


    def update_exact(self, angle_step_distance):
        # position_exact
        self.position_robot_exact[self.id_robot][0] = self.position_robot_exact[self.id_robot][0] + angle_step_distance[1] * np.cos(angle_step_distance[0])
        self.position_robot_exact[self.id_robot][1] = self.position_robot_exact[self.id_robot][1] + angle_step_distance[1] * np.sin(angle_step_distance[0])


    def update_estimate_robot(self):
        # position_estimate
        i_x = self.my_belief_position.belief_state[self.id_robot][0][0]
        i_y = self.my_belief_position.belief_state[self.id_robot][1][0]
        self.position_robot_estimate[self.id_robot] = [i_x, i_y]


    def update_estimate_neighbour(self):
        # position_estimate
        for x in range(self.number_of_robots):
            if x != self.id_robot:
                i_phi = self.my_belief_position.belief_state[x][0][0]
                i_r = self.my_belief_position.belief_state[x][1][0]
                self.position_robot_estimate[x] = [i_phi, i_r]


class homebase:
    def __init__(self, path, size_world, size_world_real, position_target, number_of_robots, position_initial):

        # Initialize
        self.path = path
        self.size_world = size_world
        self.size_world_real = size_world_real
        self.scaling = size_world[0] / size_world_real[0]
        self.position_target = position_target
        self.number_of_robots = number_of_robots
        self.id_contact = [[0, 0] for i in range(number_of_robots + 1)] # [Do I have contact to him, does he has contact to me]
        self.position_robot_exact = copy.deepcopy(position_initial)

        self.step_distance = 1000 * self.scaling

        self.my_sensor_target = SENSOR.sensor_target_angle(self.path, self.size_world, self.size_world_real, self.position_target)
        self.my_sensor_motion = SENSOR.sensor_motion(self.path, self.size_world, self.size_world_real, self.step_distance)

        self.my_belief_target = BELIEF.hb_belief_target(self.size_world, self.my_sensor_target, self.number_of_robots)
        self.my_belief_position = BELIEF.hb_belief_position(self.my_sensor_motion, self.position_robot_exact, self.number_of_robots)