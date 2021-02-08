from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import logging
import numpy as np
import random as rd
import tensorflow as tf
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from cleverhans.compat import flags
from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.dataset import CIFAR10
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from cleverhans.utils_tf import train as train_2d
from cleverhans.attacks import CarliniWagnerL2 
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.model_zoo.fully_connected import ModelMLP, ModelMLP_dyn

from global_defs import CONFIG
from functions   import normalize_reshape_inputs_2d, add_noise_and_QR, plot_data,\
                        compute_polytopes_a, compute_epsilons_balls_alpha,\
                        test_balls, compute_returnees, modify_D, plot_violins,\
                        threshold_evaluation, compute_auroc, plot_results_models

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_filters', CONFIG.NB_FILTERS,
                     'Model size multiplier')
flags.DEFINE_integer('nb_epochs', CONFIG.NB_EPOCHS,
                     'Number of epochs to train model')
flags.DEFINE_integer('batch_size', CONFIG.BATCH_SIZE,
                     'Size of training batches')
flags.DEFINE_float('learning_rate', CONFIG.LEARNING_RATE,
                   'Learning rate for training')


  
def main(argv=None):
  
  from cleverhans_tutorials import check_installation
  check_installation(__file__)
  
  if not os.path.exists( CONFIG.SAVE_PATH ):
    os.makedirs( CONFIG.SAVE_PATH )
  save_path_data = CONFIG.SAVE_PATH + 'data/'
  if not os.path.exists( save_path_data ):
    os.makedirs( save_path_data )
  model_path = CONFIG.SAVE_PATH + '../all/' +  CONFIG.DATASET + '/'
  if not os.path.exists( model_path ):
    os.makedirs( model_path )
    os.makedirs( model_path + 'data/' )
  
  nb_epochs = FLAGS.nb_epochs
  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate
  nb_filters = FLAGS.nb_filters
  len_x = int(CONFIG.NUM_TEST/2)
  
  start = time.time()

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set seeds to improve reproducibility
  if CONFIG.DATASET == 'mnist' or CONFIG.DATASET == 'cifar10':
    tf.set_random_seed(1234)
    np.random.seed(1234)
    rd.seed(1234)
  elif CONFIG.DATASET == 'moon' or CONFIG.DATASET == 'dims':
    tf.set_random_seed(13)
    np.random.seed(1234)
    rd.seed(0)          
  
  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  tf_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
  tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2 
  sess = tf.Session(config=tf_config)   
  
  if CONFIG.DATASET == 'mnist':
    # Get MNIST data
    mnist = MNIST(train_start=0, train_end=CONFIG.NUM_TRAIN,
                  test_start=0, test_end=CONFIG.NUM_TEST)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')
  elif CONFIG.DATASET == 'cifar10':
    # Get CIFAR10 data
    data = CIFAR10(train_start=0, train_end=CONFIG.NUM_TRAIN,
                  test_start=0, test_end=CONFIG.NUM_TEST)
    dataset_size = data.x_train.shape[0]
    dataset_train = data.to_tensorflow()[0]
    dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(16)
    x_train, y_train = data.get_set('train')
    x_test, y_test = data.get_set('test')                             
  elif CONFIG.DATASET == 'moon':
    # Create a two moon example
    X, y = make_moons(n_samples=(CONFIG.NUM_TRAIN+CONFIG.NUM_TEST), noise=0.2,
                      random_state=0)
    X = StandardScaler().fit_transform(X)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(X, y,
                                            test_size=(CONFIG.NUM_TEST/(CONFIG.NUM_TRAIN
                                            +CONFIG.NUM_TEST)), random_state=0)                          
    x_train, y_train, x_test, y_test = normalize_reshape_inputs_2d(model_path, x_train1,
                                                                   y_train1, x_test1,
                                                                   y_test1)
  elif CONFIG.DATASET == 'dims':
    X, y = make_moons(n_samples=(CONFIG.NUM_TRAIN+CONFIG.NUM_TEST), noise=0.2,
                      random_state=0)
    X = StandardScaler().fit_transform(X)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(X, y,
                                            test_size=(CONFIG.NUM_TEST/(CONFIG.NUM_TRAIN
                                            +CONFIG.NUM_TEST)), random_state=0)                          
    x_train2, y_train, x_test2, y_test = normalize_reshape_inputs_2d(model_path, x_train1,
                                                                     y_train1,x_test1,
                                                                     y_test1)
    x_train, x_test = add_noise_and_QR(x_train2, x_test2, CONFIG.NUM_DIMS)

  np.save(os.path.join(save_path_data, 'x_test'), x_test)
  np.save(os.path.join(save_path_data, 'y_test'), y_test)

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Train an model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  eval_params = {'batch_size': 1}
  rng = np.random.RandomState([2017, 8, 30])
  
  with open(CONFIG.SAVE_PATH + 'acc_param.txt', 'a') as fi:

    def do_eval(adv_x, preds, x_set, y_set, report_key):
      acc, pred_np, adv_x_np = model_eval(sess, x, y, preds, adv_x, nb_classes, x_set,
                                          y_set, args=eval_params)
      setattr(report, report_key, acc)
      if report_key:
        print('Accuracy on %s examples: %0.4f' % (report_key, acc), file=fi)
      return pred_np, adv_x_np
    
    if CONFIG.DATASET == 'mnist':
      trained_model_path = model_path + 'data/trained_model'
      model = ModelBasicCNN('model1', nb_classes, nb_filters)
    elif CONFIG.DATASET == 'cifar10':
      trained_model_path = model_path + 'data/trained_model'
      model = ModelAllConvolutional('model1', nb_classes, nb_filters,
                                    input_shape=[32, 32, 3])
    elif CONFIG.DATASET == 'moon':
      trained_model_path = model_path + 'data/trained_model'
      model = ModelMLP('model1', nb_classes)
    elif CONFIG.DATASET == 'dims':
      trained_model_path = save_path_data + 'trained_model'
      model = ModelMLP_dyn('model1', nb_classes, CONFIG.NUM_DIMS)
      
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    
    def evaluate():
      _, _ = do_eval(x, preds, x_test, y_test, 'test during train')
    
    if os.path.isfile( trained_model_path + '.index' ):
      tf_model_load(sess, trained_model_path)
    else:
      if CONFIG.DATASET == 'mnist':
        train(sess, loss, x_train, y_train, evaluate=evaluate,
              args=train_params, rng=rng, var_list=model.get_params())
      elif CONFIG.DATASET == 'cifar10':
        train(sess, loss, None, None,
              dataset_train=dataset_train, dataset_size=dataset_size,
              evaluate=evaluate, args=train_params, rng=rng,
              var_list=model.get_params())
      elif CONFIG.DATASET == 'moon':
        train_2d(sess, loss, x, y, x_train, y_train, save=False, evaluate=evaluate,
                args=train_params, rng=rng, var_list=model.get_params())
      elif CONFIG.DATASET == 'dims':
        train_2d(sess, loss, x, y, x_train, y_train, evaluate=evaluate,
                args=train_params, rng=rng, var_list=model.get_params())
      saver = tf.train.Saver()
      saver.save(sess, trained_model_path)
    
    # Evaluate the accuracy on test examples
    if os.path.isfile( save_path_data + 'logits_zero_attacked.npy' ):
      logits_0 = np.load(save_path_data + 'logits_zero_attacked.npy')
    else:
      _, _ = do_eval(x, preds, x_train, y_train, 'train')
      logits_0, _ = do_eval(x, preds, x_test, y_test, 'test')
      np.save(os.path.join(save_path_data, 'logits_zero_attacked'), logits_0) 
    
    if CONFIG.DATASET == 'moon':
      num_grid_points = 5000
      if os.path.isfile( model_path + 'data/images_mesh' + str(num_grid_points) + '.npy' ):
        x_mesh = np.load(model_path + 'data/images_mesh' + str(num_grid_points) + '.npy')
        logits_mesh = np.load(model_path + 'data/logits_mesh' + str(num_grid_points) + '.npy')
      else:
        xx, yy = np.meshgrid(np.linspace(0, 1, num_grid_points), np.linspace(0, 1, num_grid_points)) 
        x_mesh1 = np.stack([np.ravel(xx), np.ravel(yy)]).T
        y_mesh1 = np.ones((x_mesh1.shape[0]),dtype='int64')
        x_mesh, y_mesh, _, _ = normalize_reshape_inputs_2d(model_path, x_mesh1, y_mesh1)
        logits_mesh, _ = do_eval(x, preds, x_mesh, y_mesh, 'mesh')
        x_mesh = np.squeeze(x_mesh)
        np.save(os.path.join(model_path, 'data/images_mesh'+str(num_grid_points)), x_mesh)
        np.save(os.path.join(model_path, 'data/logits_mesh'+str(num_grid_points)), logits_mesh)
        
    points_x = x_test[:len_x]
    points_y = y_test[:len_x]
    points_x_bar = x_test[len_x:]
    points_y_bar = y_test[len_x:] 
     
    # Initialize the CW attack object and graph
    cw = CarliniWagnerL2(model, sess=sess) 
    
    # first attack
    attack_params = {
        'learning_rate': CONFIG.CW_LEARNING_RATE,
        'max_iterations': CONFIG.CW_MAX_ITERATIONS
      }
    
    if CONFIG.DATASET == 'moon':
     
      out_a = compute_polytopes_a(x_mesh, logits_mesh, model_path)
      attack_params['const_a_min'] = out_a
      attack_params['const_a_max'] = 100
    
    adv_x = cw.generate(x, **attack_params) 
      
    if os.path.isfile( save_path_data + 'images_once_attacked.npy' ):
      adv_img_1 = np.load(save_path_data + 'images_once_attacked.npy')
      logits_1 = np.load(save_path_data + 'logits_once_attacked.npy')
    else:
      #Evaluate the accuracy on adversarial examples
      preds_adv = model.get_logits(adv_x)
      logits_1, adv_img_1 = do_eval(adv_x, preds_adv, points_x_bar, points_y_bar,
                                    'test once attacked')
      np.save(os.path.join(save_path_data, 'images_once_attacked'), adv_img_1)
      np.save(os.path.join(save_path_data, 'logits_once_attacked'), logits_1)
      
    # counter attack 
    attack_params['max_iterations'] = 1024
      
    if CONFIG.DATASET == 'moon':  
      
      out_alpha2 = compute_epsilons_balls_alpha(x_mesh, np.squeeze(x_test),
                                                np.squeeze(adv_img_1), model_path,
                                                CONFIG.SAVE_PATH)
      attack_params['learning_rate'] = out_alpha2
      attack_params['const_a_min'] = -1
      attack_params['max_iterations'] = 2048
      
      plot_data(np.squeeze(adv_img_1), logits_1, CONFIG.SAVE_PATH+'data_pred1.png', x_mesh,
                logits_mesh)
      
    adv_adv_x = cw.generate(x, **attack_params) 
      
    x_k = np.concatenate((points_x, adv_img_1), axis=0)
    y_k = np.concatenate((points_y, logits_1), axis=0)
    
    if os.path.isfile( save_path_data + 'images_twice_attacked.npy' ):
      adv_img_2 = np.load(save_path_data + 'images_twice_attacked.npy')
      logits_2 = np.load(save_path_data + 'logits_twice_attacked.npy')
    else:
      # Evaluate the accuracy on adversarial examples
      preds_adv_adv = model.get_logits(adv_adv_x)
      logits_2, adv_img_2 = do_eval(adv_adv_x, preds_adv_adv, x_k, y_k,
                                    'test twice attacked')   
      
      np.save(os.path.join(save_path_data, 'images_twice_attacked'), adv_img_2)
      np.save(os.path.join(save_path_data, 'logits_twice_attacked'), logits_2)
    
    if CONFIG.DATASET == 'moon':  
      plot_data(np.squeeze(adv_img_2[:len_x]), logits_2[:len_x],
                CONFIG.SAVE_PATH+'data_pred2.png', x_mesh, logits_mesh)
      plot_data(np.squeeze(adv_img_2[len_x:]), logits_2[len_x:],
                CONFIG.SAVE_PATH+'data_pred12.png', x_mesh, logits_mesh)
      test_balls(np.squeeze(x_k), np.squeeze(adv_img_2), logits_0, logits_1, logits_2,
                 CONFIG.SAVE_PATH)
 
  compute_returnees(logits_0[len_x:], logits_1, logits_2[len_x:], logits_0[:len_x],
                    logits_2[:len_x], CONFIG.SAVE_PATH) 
  
  if x_test.shape[-1] > 1:
    num_axis=(1,2,3)
  else:
    num_axis=(1,2)
    
  D_p = np.squeeze(np.sqrt(np.sum(np.square(points_x-adv_img_2[:len_x]), axis=num_axis)))
  D_p_p = np.squeeze(np.sqrt(np.sum(np.square(adv_img_1-adv_img_2[len_x:]),
                                    axis=num_axis)))
  D_p_mod, D_p_p_mod = modify_D(D_p, D_p_p, logits_0[len_x:], logits_1, logits_2[len_x:],
                                logits_0[:len_x], logits_2[:len_x])
      
  if D_p_mod != [] and D_p_p_mod != []:
    plot_violins(D_p_mod, D_p_p_mod, CONFIG.SAVE_PATH)
    threshold_evaluation(D_p_mod, D_p_p_mod, CONFIG.SAVE_PATH)
    _ = compute_auroc(D_p_mod, D_p_p_mod, CONFIG.SAVE_PATH)
      
  plot_results_models(len_x, CONFIG.DATASET, CONFIG.SAVE_PATH)
  
  print('Time needed:', time.time()-start)

  return report


if __name__ == '__main__':
  
  print( '===== START =====' )
  tf.app.run()
  print( '===== DONE! =====' )
