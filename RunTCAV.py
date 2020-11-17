import cav as cav
import model as model
import tcav as tcav
import utils as utils
import utils_plot as utils_plot  # utils_plot requires matplotlib
import os
import torch
import activation_generator as act_gen
import tensorflow as tf

working_dir = './tcav_class_test'
activation_dir = working_dir + '/activations/'
cav_dir = working_dir + '/cavs/'
source_dir = "./data/"
bottlenecks = ['conv2']

utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(cav_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs.
alphas = [0.1]

target = 'cat'
concepts = ["dotted", "striped", "zigzagged"]
random_counterpart = 'random500_1'

LABEL_PATH = './data/imagenet_comp_graph_label_strings.txt'

mymodel = model.CNNWrapper(LABEL_PATH)

act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=100)

tf.compat.v1.logging.set_verbosity(0)

num_random_exp = 30  # folders (random500_0, random500_1)

mytcav = tcav.TCAV(target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp)

results = mytcav.run()

utils_plot.plot_results(results, num_random_exp=num_random_exp)