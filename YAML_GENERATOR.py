import numpy as np
import os

path = 'config'
os.makedirs(path, exist_ok=True)

def write(max_runtime, position_initial, step_distance, number_of_directions, path_depth, decision, rise_gain, diving_depth):

    name = str(len(position_initial)) + 'rob_' + str(number_of_directions) + 'dir_' + str(path_depth) + 'pat_' + str(decision[0:3]) + '_' + str(rise_gain) + '_' + str(diving_depth)
    bsub = 'bsub -W 24:00 -R "rusage[mem=10000]" python SIMULATION.py'

    # Write submit command
    file = open('submit.txt', "a")
    file.write(bsub + ' config/' + name + '.yaml' + '\n')
    file.close()

    # Clear file
    file = open(path + '/' + name + '.yaml', "w")
    file.close()
    os.remove(path + '/' + name + '.yaml')

    # Write file
    file = open(path + '/' + name + '.yaml', "w")
    text = ''

    text = text + "# parameter_simulation" + '\n'
    text = text + "name_of_simulation: " + "'" + 'R_' + name + "'" + '\n'
    text = text + "size_world: [50000, 50000]" + '\n'
    text = text + "resolution: [250, 250]" + '\n'
    text = text + "position_initial: " + str(position_initial) + '\n'
    text = text + "position_target: 'random'" + '\n'
    text = text + "max_belief: 0.99" + '\n'
    text = text + "max_runtime: " + str(max_runtime) + '\n'
    text = text + "max_error: 0" + '\n'
    text = text + "" + '\n'
    text = text + "# parameter_robot" + '\n'
    text = text + "deciding_rate: " + str(int(2250 / step_distance)) + " #steps/decision" + '\n'
    text = text + "step_distance: " + str(step_distance) + '\n'
    text = text + "number_of_directions: " + str(number_of_directions) + '\n'
    text = text + "path_depth: " + str(path_depth) + '\n'
    text = text + "communication_range_observation: 5000" + '\n'
    text = text + "communication_range_neighbour: 9500" + '\n'
    text = text + "choice_sensor_target: 'angle'" + '\n'
    text = text + "" + '\n'
    text = text + "# parameter_sensor_target" + '\n'
    text = text + "cross_over: 3500" + '\n'
    text = text + "width: 180" + '\n'
    text = text + "smoothness: 6" + '\n'
    text = text + "max_pos: 0.9" + '\n'
    text = text + "max_neg: 0.1" + '\n'
    text = text + "std_angle: 3" + '\n'
    text = text + "" + '\n'
    text = text + "# parameter_sensor_motion" + '\n'
    text = text + "std_v: 0.001" + '\n'
    text = text + "" + '\n'
    text = text + "# parameter_sensor_distance" + '\n'
    text = text + "std_const: 100" + '\n'
    text = text + "std_mean: 0.1" + '\n'
    text = text + "" + '\n'
    text = text + "# parameter_belief_position" + '\n'
    text = text + "std_x: 100" + '\n'
    text = text + "std_y: 100" + '\n'
    text = text + "" + '\n'
    text = text + "# parameter belief" + '\n'
    text = text + "lower_bound: ''" + '\n'
    text = text + "" + '\n'
    text = text + "# parameter_decision" + '\n'
    text = text + "decision: " + str(decision) + '\n'
    text = text + "rise_gain: " + str(rise_gain) + '\n'
    text = text + "rise_time: ''" + '\n'
    text = text + "rise_n: ''" + '\n'
    text = text + "diving_depth: " + str(diving_depth) + '\n'

    file.write(text)
    file.close()

max_runtime = [9*50000, 5*50000, 4*50000, 3*50000]
position_0 = [[[0, 0]], [[0, 0], [50000, 50000]], [[0, 0], [50000, 0], [50000, 50000]]]
position_1 = [[[0, 0]], [[7000, 0], [0, 0]], [[0, 0], [7000, 0], [14000, 0]]]
number_of_directions_0 = [8, 16]
path_depth_0 = [1, 3]
decision_0 = ['expensive', 'cheap']
decision_1 = ['lawnmower']
rise_gain_0 = ['on', 'off']
diving_depth_0 = [2]

# 0
for a in position_0:
    for b in number_of_directions_0:
        for c in path_depth_0:
            if len(a) != 1:
                for d in decision_0:
                    for e in rise_gain_0:
                        if (e == 'on'):
                            for f in diving_depth_0:
                                write(max_runtime[len(a)-1], a, 1.5, b, c, d, e, f)
                        elif (e == 'off'):
                            write(max_runtime[len(a)-1], a, 1.5, b, c, d, e, 1)
            else:
                write(max_runtime[len(a) - 1], a, 1.5, b, c, 'expensive', 'off', 1)

# 1
for a in position_1:
    write(max_runtime[len(a)-1], a, 1, 4, 1, decision_1[0], 'off', 1)