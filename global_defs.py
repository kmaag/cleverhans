#!/usr/bin/env python3
'''
script including
class object with global settings
'''

class CONFIG:
  
  #------------------#
  # select or define #
  #------------------#
  
  datasets = ['mnist','cifar10','moon','dims'] 
  DATASET = datasets[1] 
  
  #---------------------#
  # set necessary path  #
  #---------------------#
  
  my_io_path  = '/home/user/'
  
  #-----------#
  # optionals #
  #-----------#

  # training parameters 
  if DATASET == 'mnist' or DATASET == 'cifar10':
    NUM_TRAIN = 60000 
    NUM_TEST = 10000 
    NB_EPOCHS = 6
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NB_FILTERS = 64
    
  elif DATASET == 'moon':
    NUM_TRAIN = 2000 
    NUM_TEST = 300
    NB_EPOCHS = 50 
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1 
    NB_FILTERS = 0
    
  elif DATASET == 'dims':
    NUM_TRAIN = 10000 
    NUM_TEST = 600
    NB_EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01 
    NB_FILTERS = 0
    NUM_DIMS = 256
  
  # Carlini-Wagner parameters
  CW_MAX_ITERATIONS = 1024
  CW_LEARNING_RATE = 1e-2
  
  if DATASET == 'dims':
    SAVE_PATH = my_io_path + DATASET + '_' + str(CW_MAX_ITERATIONS) + '_' + str(NUM_DIMS) + '/'
  else:
    SAVE_PATH = my_io_path + DATASET + '_' + str(CW_MAX_ITERATIONS) + '/'

    
    


