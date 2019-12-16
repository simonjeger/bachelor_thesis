# This is a script that generates the .yaml files that are used by SIMULATION.py

import numpy as np
import os

path = 'config'
os.makedirs(path, exist_ok=True)

def write(visual, max_runtime, position_initial, position_target, step_distance, number_of_directions, path_depth, decision, rise_gain, diving_depth):

    name = str(len(position_initial)) + 'rob_' + str(number_of_directions) + 'dir_' + str(path_depth) + 'pat_' + str(decision[0:3]) + '_' + str(rise_gain) + '_' + str(diving_depth)
    if len(position_initial) == 1:
        bsub = 'bsub -W 50:00 -R "rusage[mem=10000]" python SIMULATION.py'
    if len(position_initial) == 2:
        bsub = 'bsub -W 100:00 -R "rusage[mem=10000]" python SIMULATION.py'
    if len(position_initial) == 3:
        bsub = 'bsub -W 240:00 -R "rusage[mem=10000]" python SIMULATION.py'

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
    text = text + "resolution: [40, 40]" + '\n'
    text = text + "number_of_cicles: ''" + '\n'
    text = text + "visual: '" + str(visual) + "'" + '\n'
    text = text + "position_initial: " + str(position_initial) + '\n'
    text = text + "position_target: " + str(position_target) + '\n'
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
    text = text + "choice_sensor_target: 'boolean'" + '\n'
    text = text + "" + '\n'
    text = text + "# parameter_sensor_target" + '\n'
    text = text + "cross_over: 3500" + '\n'
    text = text + "alpha: 0.9" + '\n'
    text = text + "beta: 0.9" + '\n'
    text = text + "gamma: 1.95" + '\n'
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
    text = text + "decision: '" + str(decision) + "'"+ '\n'
    text = text + "rise_gain: '" + str(rise_gain) + "'" + '\n'
    text = text + "rise_time: ''" + '\n'
    text = text + "rise_n: ''" + '\n'
    text = text + "diving_depth: " + str(diving_depth) + '\n'

    file.write(text)
    file.close()

max_runtime = [9*50000, 5*50000, 4*50000, 3*50000]

position_initial_0 = [[[0, 0]], [[0, 0], [50000, 50000]], [[0, 0], [50000, 0], [0, 50000]]]
#position_initial_0 = [[[0, 0], [50000, 50000]], [[0, 0], [50000, 0], [0, 50000]]]
position_initial_1 = [[[0, 0]], [[7000, 0], [0, 0]], [[0, 0], [7000, 0], [14000, 0]]]

position_target_x = np.linspace(2500, 47500, 10)
position_target_y = np.linspace(2500, 47500, 10)
position_target = []
n = 5
for j in range(len(position_target_y)):
    for i in range(len(position_target_x)):
        position_target = position_target + [[position_target_x[i], position_target_y[j]]] * n

number_of_directions_0 = [8]
path_depth_0 = [1, 2, 3]
#path_depth_0 = [1]
decision_0 = ['expensive', 'cheap']
#decision_0 = ['cheap']
#decision_1 = ['lawnmower']
#rise_gain_0 = ['on', 'off']
#rise_gain_0 = ['on']
rise_gain_0 = ['off']
diving_depth_0 = [1, 2]
visual = 'off'


# 0: probabilistic algorithm
for a in position_initial_0:
    for b in number_of_directions_0:
        for c in path_depth_0:
            for d in decision_0:
                if ((c > 1) & (d == 'expensive') & (len(a) > 1)) | ((c > 2) & (d == 'cheap') & (len(a) > 1)):
                    print('')
                else:
                    if len(a) != 1:
                        for e in rise_gain_0:
                            if (e == 'on'):
                                for f in diving_depth_0:
                                    write(visual, max_runtime[len(a)-1], a, position_target, 1.5, b, c, d, e, f)
                            elif (e == 'off'):
                                write(visual, max_runtime[len(a)-1], a, position_target, 1.5, b, c, d, e, 1)
                    else:
                        write(visual, max_runtime[len(a) - 1], a, position_target, 1.5, b, c, d, 'off', 1)

# 1: lawnmower algorithm
for a in position_initial_1:
    write(visual, max_runtime[len(a)-1], a, position_target, 0.75, 4, 1, decision_1[0], 'off', 1)