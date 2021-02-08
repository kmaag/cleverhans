"""The CarliniWagnerL2 attack
"""
# pylint: disable=missing-docstring
import logging

import os
import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_sum, reduce_max
from cleverhans.model import CallableModelWrapper, Model, wrapper_warning_logits
from cleverhans import utils

from global_defs import CONFIG

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

_logger = utils.create_logger("cleverhans.attacks.carlini_wagner_l2")
_logger.setLevel(logging.INFO)


class CarliniWagnerL2(Attack):
  """
  This attack was originally proposed by Carlini and Wagner. It is an
  iterative attack that finds adversarial examples on many defenses that
  are robust to other attacks.
  Paper link: https://arxiv.org/abs/1608.04644

  At a high level, this attack is an iterative attack using Adam and
  a specially-chosen loss function to find adversarial examples with
  lower distortion than other attacks. This comes at the cost of speed,
  as this attack is often much slower than others.

  :param model: cleverhans.model.Model
  :param sess: tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess, dtypestr='float32', **kwargs):
    """
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """
    if not isinstance(model, Model):
      wrapper_warning_logits()
      model = CallableModelWrapper(model, 'logits')

    super(CarliniWagnerL2, self).__init__(model, sess, dtypestr, **kwargs)

    self.feedable_kwargs = ('y', 'y_target')

    self.structural_kwargs = [
        'batch_size', 'confidence', 'targeted', 'learning_rate', 'const_a_min',
        'const_a_max', 'max_iterations', 'clip_min', 'clip_max'
    ]

  def generate(self, x, **kwargs):
    """
    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.

    :param x: A tensor with the inputs.
    :param kwargs: See `parse_params`
    """
    assert self.sess is not None, \
        'Cannot use `generate` when no `sess` was provided'
    self.parse_params(**kwargs)

    labels, nb_classes = self.get_or_guess_labels(x, kwargs)

    attack = CWL2(self.sess, self.model, self.batch_size, self.confidence,
                  'y_target' in kwargs, self.learning_rate, self.const_a_min,
                  self.const_a_max, self.max_iterations, self.clip_min, self.clip_max,
                  nb_classes, x.get_shape().as_list()[1:])
    
    self.counter_x = -1 
    def cw_wrap(x_val, y_val):
      self.counter_x += 1
      return np.array(attack.attack(x_val, y_val, self.counter_x), dtype=self.np_dtype)

    wrap = tf.py_func(cw_wrap, [x, labels], self.tf_dtype)
    wrap.set_shape(x.get_shape())

    return wrap

  def parse_params(self,
                   y=None,
                   y_target=None,
                   batch_size=1,
                   confidence=0,
                   learning_rate=5e-3,
                   const_a_min=1e-2,
                   const_a_max=1e10,
                   max_iterations=1000,
                   clip_min=0,
                   clip_max=1):
    """
    :param y: (optional) A tensor with the true labels for an untargeted
              attack. If None (and y_target is None) then use the
              original labels the classifier assigns.
    :param y_target: (optional) A tensor with the target labels for a
              targeted attack.
    :param confidence: Confidence of adversarial examples: higher produces
                       examples with larger l2 distortion, but more
                       strongly classified as adversarial.
    :param batch_size: Number of attacks to run simultaneously.
    :param learning_rate: The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
    :param const_a_min: The constant value for parameter a (min).
    :param const_a_max: The constant value for parameter a (max).
    :param max_iterations: The maximum number of iterations. Setting this
                           to a larger value will produce lower distortion
                           results. Using only a few iterations requires
                           a larger learning rate, and will produce larger
                           distortion results.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # ignore the y and y_target argument
    self.batch_size = batch_size
    self.confidence = confidence
    self.learning_rate = learning_rate
    self.const_a_min = np.ones(batch_size)*const_a_min
    self.const_a_max = np.ones(batch_size)*const_a_max
    self.max_iterations = max_iterations
    self.clip_min = clip_min
    self.clip_max = clip_max


def ZERO():
  return np.asarray(0., dtype=np_dtype)


class CWL2(object):
  def __init__(self, sess, model, batch_size, confidence, targeted,
               learning_rate, const_a_min, const_a_max, max_iterations, 
               clip_min, clip_max, num_labels, shape):
               
    """
    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.

    :param sess: a TF session.
    :param model: a cleverhans.model.Model object.
    :param batch_size: Number of attacks to run simultaneously.
    :param confidence: Confidence of adversarial examples: higher produces
                       examples with larger l2 distortion, but more
                       strongly classified as adversarial.
    :param targeted: boolean controlling the behavior of the adversarial
                     examples produced. If set to False, they will be
                     misclassified in any wrong class. If set to True,
                     they will be misclassified in a chosen target class.
    :param learning_rate: The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
    :param const_a_min: The constant value for parameter a (min).
    :param const_a_max: The constant value for parameter a (max).
    :param max_iterations: The maximum number of iterations. Setting this
                           to a larger value will produce lower distortion
                           results. Using only a few iterations requires
                           a larger learning rate, and will produce larger
                           distortion results.
    :param clip_min: (optional float) Minimum input component value.
    :param clip_max: (optional float) Maximum input component value.
    :param num_labels: the number of classes in the model's output.
    :param shape: the shape of the model's input tensor.
    """

    self.sess = sess
    self.TARGETED = targeted
    self.LEARNING_RATE = learning_rate
    self.MAX_ITERATIONS = max_iterations
    self.CONST_A_MIN = const_a_min
    self.CONST_A_MAX = const_a_max
    self.CONFIDENCE = confidence
    self.batch_size = batch_size
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.model = model

    self.shape = shape = tuple([batch_size] + list(shape))

    # the variable we're going to optimize over
    modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))

    # these are variables to be more efficient in sending data to tf
    self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')
    self.tlab = tf.Variable(
        np.zeros((batch_size, num_labels)), dtype=tf_dtype, name='tlab')
    self.const = tf.Variable(
        np.zeros(batch_size), dtype=tf_dtype, name='const')

    # and here's what we use to assign them
    self.assign_timg = tf.placeholder(tf_dtype, shape, name='assign_timg')
    self.assign_tlab = tf.placeholder(
        tf_dtype, (batch_size, num_labels), name='assign_tlab')
    self.assign_const = tf.placeholder(
        tf_dtype, [batch_size], name='assign_const')

    # the resulting instance, tanh'd to keep bounded from clip_min
    # to clip_max
    self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
    self.newimg = self.newimg * (clip_max - clip_min) + clip_min

    # prediction BEFORE-SOFTMAX of the model
    self.output = model.get_logits(self.newimg)

    # distance to the input data
    self.other = (tf.tanh(self.timg) + 1) / \
        2 * (clip_max - clip_min) + clip_min
    self.l2dist = reduce_sum(
        tf.square(self.newimg - self.other), list(range(1, len(shape))))

    # compute the probability of the label class versus the maximum other
    real = reduce_sum((self.tlab) * self.output, 1)
    other = reduce_max((1 - self.tlab) * self.output - self.tlab * 10000,
                       1)

    if self.TARGETED:
      # if targeted, optimize for making the other class most likely
      loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
    else:
      # if untargeted, optimize for making this class least likely.
      loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

    # sum up the losses
    self.loss2 = reduce_sum(self.l2dist)
    self.loss1 = reduce_sum(self.const * loss1)
    self.loss = self.loss1 + self.loss2
    
    # Setup the adam optimizer and keep track of variables we're creating
    start_vars = set(x.name for x in tf.global_variables())
    batch_step = tf.Variable(99, trainable=False)
    learn_rate = tf.train.inverse_time_decay(learning_rate=self.LEARNING_RATE*100,
                                             global_step=batch_step * batch_size,
                                             decay_steps=1.0, decay_rate=1.0)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.0,
                                           use_nesterov=False)
    # Passing batch_step to minimize() will increment it at each step
    self.train = optimizer.minimize(self.loss, var_list=[modifier], global_step=batch_step)
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]

    # these are the variables to initialize when we run
    self.setup = []
    self.setup.append(self.timg.assign(self.assign_timg))
    self.setup.append(self.tlab.assign(self.assign_tlab))
    self.setup.append(self.const.assign(self.assign_const))

    self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

  def attack(self, imgs, targets, counter_x):
    """
    Perform the L_2 attack on the given instance for the given targets.

    If self.targeted is true, then the targets represents the target labels
    If self.targeted is false, then targets are the original class labels
    """
    self.counter_x = counter_x
    r = []
    for i in range(0, len(imgs), self.batch_size):
      _logger.debug(
          ("Running CWL2 attack on instance", i, " of ", len(imgs)))
      r.extend(
          self.attack_batch(imgs[i:i + self.batch_size],
                            targets[i:i + self.batch_size]))
    return np.array(r)

  def attack_batch(self, imgs, labs):
    """
    Run the attack on a batch of instance and labels.
    """

    def compare(x, y):
      if not isinstance(x, (float, int, np.int64)):
        x = np.copy(x)
        if self.TARGETED:
          x[y] -= self.CONFIDENCE
        else:
          x[y] += self.CONFIDENCE
        x = np.argmax(x)
      if self.TARGETED:
        return x == y
      else:
        return x != y
    
    batch_size = self.batch_size

    oimgs = np.clip(imgs, self.clip_min, self.clip_max)

    # re-scale instances to be within range [0, 1]
    imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
    imgs = np.clip(imgs, 0, 1)
    # now convert to [-1, 1]
    imgs = (imgs * 2) - 1
    # convert to tanh-space
    imgs = np.arctanh(imgs * .999999)
    
    # placeholders for the best l2, score, and instance attack found so far
    o_bestl2 = [1e10] * batch_size
    o_bestattack = np.copy(oimgs)

    # set the lower and upper bounds accordingly
    if CONFIG.DATASET == 'moon':
      if self.CONST_A_MIN == -1:
        param_b = np.load(CONFIG.SAVE_PATH + 'data/param_b.npy')
        CONST = np.ones(batch_size) * param_b[int((self.counter_x)/2)]
      else:
        lower_bound = np.ones(batch_size) * self.CONST_A_MIN
        CONST = np.ones(batch_size) * self.CONST_A_MIN
        upper_bound = np.ones(batch_size) * self.CONST_A_MAX
    else:
      lower_bound = np.zeros(batch_size)
      CONST = np.ones(batch_size) * 1e-2
      upper_bound = np.ones(batch_size) * 1e10
    
    o_bestconst = CONST.copy()
    
    if CONFIG.DATASET == 'moon' and self.CONST_A_MIN == -1:
      # completely reset adam's internal state.
      self.sess.run(self.init)
      batch = imgs[:batch_size]
      batchlab = labs[:batch_size]

      # set the variables so that we don't have to send them over again
      self.sess.run(
          self.setup, {
              self.assign_timg: batch,
              self.assign_tlab: batchlab,
              self.assign_const: CONST
          })
      
      flag_break = 0
      for iteration in range(self.MAX_ITERATIONS):
        
        if flag_break == 1:
          break
        
        _, l, l2s, scores, nimg = self.sess.run([
            self.train, self.loss, self.l2dist, self.output,
            self.newimg
        ]) 

        if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
          _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                          "l2={:.3g} f={:.3g}").format(
                              iteration, self.MAX_ITERATIONS, l,
                              np.mean(l2s), np.mean(scores)))

        # adjust the best result found so far
        for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
          lab = np.argmax(batchlab[e])
          o_bestattack[e] = ii
          if compare(sc, lab):
            o_bestl2[e] = l2
            flag_break = 1
            break
            
      _logger.debug("  Successfully generated adversarial examples " +
                    " of {} instances.".format(batch_size))
      o_bestl2 = np.array(o_bestl2)
      mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
      _logger.debug("   Mean successful distortion: {:.4g}".format(mean))
    
    else:   
        
      flag_compare = 0
      self.BINARY_SEARCH_STEPS = 5
      self.repeat = self.BINARY_SEARCH_STEPS >= 10
    
      for outer_step in range(self.BINARY_SEARCH_STEPS):
        # completely reset adam's internal state.
        self.sess.run(self.init)
        batch = imgs[:batch_size]
        batchlab = labs[:batch_size]

        bestl2 = [1e10] * batch_size
        bestscore = [-1] * batch_size
        _logger.debug("  Binary search step %s of %s",
                      outer_step, self.BINARY_SEARCH_STEPS)

        # The last iteration (if we run many steps) repeat the search once.
        if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
          CONST = upper_bound

        # set the variables so that we don't have to send them over again
        self.sess.run(
            self.setup, {
                self.assign_timg: batch,
                self.assign_tlab: batchlab,
                self.assign_const: CONST
            })

        for iteration in range(self.MAX_ITERATIONS):
          # perform the attack
          _, l, l2s, scores, nimg = self.sess.run([
              self.train, self.loss, self.l2dist, self.output,
              self.newimg
          ])

          if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
            _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                          "l2={:.3g} f={:.3g}").format(
                              iteration, self.MAX_ITERATIONS, l,
                              np.mean(l2s), np.mean(scores)))

          # adjust the best result found so far
          for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
            lab = np.argmax(batchlab[e])
            if l2 < bestl2[e] and compare(sc, lab):
              bestl2[e] = l2
              bestscore[e] = np.argmax(sc)
            if compare(sc, lab):
              o_bestl2[e] = l2
              o_bestattack[e] = ii
              o_bestconst[e] = CONST[e]
              flag_compare = 1
            if flag_compare == 0:
              o_bestattack[e] = ii

        # adjust the constant as needed
        for e in range(batch_size):
          if compare(bestscore[e], np.argmax(batchlab[e])) and \
            bestscore[e] != -1:
            # success, divide const by two
            upper_bound[e] = min(upper_bound[e], CONST[e])
            if upper_bound[e] < 1e9:
              CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
          else:
            # failure, either multiply by 10 if no solution found yet
            #          or do binary search with the known upper bound
            lower_bound[e] = max(lower_bound[e], CONST[e])
            if upper_bound[e] < 1e9:
              CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
            else:
              CONST[e] *= 10
        _logger.debug("  Successfully generated adversarial examples " +
                      "on {} of {} instances.".format(
                          sum(upper_bound < 1e9), batch_size))
        o_bestl2 = np.array(o_bestl2)
        mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
        _logger.debug("   Mean successful distortion: {:.4g}".format(mean))
        
    # return the best solution found
    return o_bestattack
  
  
  
  
