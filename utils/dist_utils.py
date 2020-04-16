#########################
# Purpose: Useful functions for calculating the distance between different weight vectors
########################
import numpy as np
import os
import argparse
import tensorflow as tf

import global_vars as gv

from io_utils import file_write

def collate_weights(delta_curr):
    for l in range(len(delta_curr)):
        flat_layer = delta_curr[l].flatten()
        if l == 0:
            delta_curr_w = flat_layer
        elif l == 1:
            delta_curr_b = flat_layer
        elif l % 2 == 0:
            delta_curr_w = np.concatenate(
                (delta_curr_w, flat_layer))
        elif (l + 1) % 2 == 0:
            delta_curr_b = np.concatenate(
                (delta_curr_b, flat_layer))
    return delta_curr_w, delta_curr_b

def model_shape_size(delta_curr):
    shape_w = []
    shape_b = []
    size_w = []
    size_b = []
    for l in range(len(delta_curr)):
        layer_shape = delta_curr[l].shape
        size = 1
        for item in layer_shape:
            size *= item
        if l % 2 == 0:
            size_w.append(size)
            shape_w.append(layer_shape)
        elif (l + 1) % 2 == 0:
            size_b.append(size)
            shape_b.append(layer_shape)
    return [shape_w, shape_b, size_w, size_b]

def est_accuracy(mal_visible, t):
    args = gv.args

    delta_other_prev = None
    # If the malicious agent has been chosen in a previous epoch
    if len(mal_visible) >= 1:
        # Choose the latest epoch when the adv was chosen
        mal_prev_t = mal_visible[-1]
        print('Loading from previous iteration %s' % mal_prev_t)

        delta_other_prev = np.load(
            gv.dir_name + 'ben_delta_t%s.npy' % mal_prev_t, allow_pickle=True)
        delta_other_prev = delta_other_prev / (t - mal_prev_t)
        print('Divisor: %s' % (t - mal_prev_t))

    # Check the accuracy of estimate after time step 2
    if len(mal_visible) >= 3:
        mal_prev_prev_t = mal_visible[-2]
        if mal_prev_prev_t >= args.mal_delay:
            delta_other_prev_prev = np.load(
                gv.dir_name + 'ben_delta_t%s.npy' % mal_prev_prev_t, allow_pickle=True)
            # Subtract the penultimate benign delta from the last delta
            ben_delta_diff = delta_other_prev - delta_other_prev_prev
            est_accuracy_l2 = 0.0
            # For each weight layer, add the norm of the delta differences
            for i in range(len(ben_delta_diff)):
                est_accuracy_l2 += np.linalg.norm(ben_delta_diff[i])
            print('Accuracy of estimate on round %s: %s' %
                  (mal_prev_prev_t, est_accuracy_l2))
            write_dict = {}
            write_dict['t'] = mal_prev_prev_t
            write_dict['est_accuracy_l2'] = est_accuracy_l2
            file_write(write_dict, purpose='est_accuracy_log')

    # Return the estimated previous iteration with adversary's benign weight delta
    return delta_other_prev

def weight_constrain(loss1,mal_loss1,agent_model,constrain_weights,t):
    args = gv.args
    # Add weight based regularization
    loss2 = tf.constant(0.0)
    layer_count = 0
    if 'dist_oth' in args.mal_strat and t<1:
        rho = 0.0
    else:
        rho = 1e-4
    for layer in agent_model.layers:
        counter = 0
        for weight in layer.weights:
            constrain_weight_curr = tf.convert_to_tensor(
                constrain_weights[layer_count], dtype=tf.float32)
            delta_constrain = (weight - constrain_weight_curr)
            if 'wt_o' in args.mal_strat:
                if counter % 2 == 0:
                    loss2 += tf.nn.l2_loss(delta_constrain)
            else:
                loss2 += tf.nn.l2_loss(delta_constrain)
            layer_count += 1
            counter += 1
    loss = loss1 + rho * loss2
    mal_loss = mal_loss1
    
    return loss, loss2, mal_loss
