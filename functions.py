import os
import numpy as np
import random as rd
import seaborn as sns
from PIL import Image
from pandas import DataFrame
from skimage import measure as ms
import matplotlib.colors as colors
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid.axislines import SubplotZero

from global_defs import CONFIG


def colorFader(c1,c2,mix=0): 
  c1=np.array(matplotlib.colors.to_rgb(c1))
  c2=np.array(matplotlib.colors.to_rgb(c2))
  return ((1-mix)*c1 + mix*c2) * 255


def get_neighbors(mat, row, col, radius=2):
  rows, cols = len(mat), len(mat[0])
  out = []
  for i in range(row-radius, row+radius+1):
    row = []
    for j in range(col-radius, col+radius+1):
      if 0 <= i < rows and 0 <= j < cols:
        row.append(mat[i][j])
      else:
        row.append(0)
    out.append(row)
  return np.asarray(out)


def normalize_reshape_inputs_2d(model_path, x_train1, y_train1, x_test1=[], y_test1=[]):
  # shapes x:(num examples, 2), y:(num examples)
  x_train1 -= x_train1.min()
  x_train1 /= x_train1.max()
  x_train = np.expand_dims(x_train1, axis=1)
  x_train = np.expand_dims(x_train, axis=1)
  y_train = np.zeros((y_train1.size, y_train1.max()+1))
  y_train[np.arange(y_train1.size),y_train1] = 1
  if x_test1 != []:
    x_test1 -= x_test1.min()
    x_test1 /= x_test1.max()
    x_test = np.expand_dims(x_test1, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    y_test = np.zeros((y_test1.size, y_test1.max()+1))
    y_test[np.arange(y_test1.size),y_test1] = 1
    plot_data(x_train1, y_train, model_path+'data_train.png')
    plot_data(x_test1, y_test, model_path+'data_test.png')
  else:
    x_test = x_test1.copy()
    y_test = y_test1.copy()
  # shapes x:(num examples, 1, 1, 2), y:(num examples, 2)
  return x_train, y_train, x_test, y_test


def add_noise_and_QR(x_train1, x_test1, num_dims):
  
  x_train1 = np.squeeze(x_train1)
  x_test1 = np.squeeze(x_test1)
  add_dims = int(num_dims-2)

  noise_train = np.clip(np.random.normal(0.5,0.1,(x_train1.shape[0],add_dims)), 0,1)
  noise_test = np.clip(np.random.normal(0.5,0.1,(x_test1.shape[0],add_dims)), 0,1)
  
  noise_train = np.concatenate((x_train1, noise_train), axis=-1)
  noise_test = np.concatenate((x_test1, noise_test), axis=-1)
  
  noise_train = np.expand_dims(noise_train, axis=1)
  x_train = np.expand_dims(noise_train, axis=1)
  
  noise_test = np.expand_dims(noise_test, axis=1)
  x_test = np.expand_dims(noise_test, axis=1)

  return x_train, x_test
  

def plot_data(x_test, y_test, save_path, x_mesh=[], y_mesh=[] ):
  print('plot data')
  
  colors_mesh = {0:'aliceblue', 1:'lavenderblush'}
  colors = {0:'cornflowerblue', 1:'hotpink'} 
  
  font_size = 16

  fig, ax = plt.subplots()
  
  if x_mesh != []:
    x_mesh = x_mesh[::50,:]
    y_mesh = y_mesh[::50,:]
    y_mesh = np.argmax(y_mesh,axis=-1)
    df_mesh = DataFrame(dict(x=x_mesh[:,0], y=x_mesh[:,1], label=y_mesh))
    grouped = df_mesh.groupby('label')
    for key, group in grouped:
      group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors_mesh[key])
  
  y_test = np.argmax(y_test,axis=-1)
  df = DataFrame(dict(x=x_test[:,0], y=x_test[:,1], label=y_test))
  grouped = df.groupby('label')
  for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
  
  ax.set_xlabel(' ')
  ax.set_ylabel(' ')
  plt.xticks(fontsize=font_size)
  plt.yticks(fontsize=font_size)
  plt.legend(fontsize=16)
    
  plt.savefig(save_path, dpi=400)


def compute_polytopes_a(x_mesh, logits_mesh, save_path, flag_polytopes=False,
                        flag_surface=False, flag_gradients=False):
  print('compute parameters')
  
  h = np.abs(x_mesh[0,0] - x_mesh[1,0]) # grid distance
  sqrt_xy = int(np.sqrt(x_mesh.shape[0])) # grid length
  
  x_grid = x_mesh[:,0].reshape(sqrt_xy,sqrt_xy)
  y_grid = x_mesh[:,1].reshape(sqrt_xy,sqrt_xy)
  argmax_mesh = np.argmax(logits_mesh, axis=1)
  
  if not os.path.isfile( save_path + 'data/grid_polytopes' + str(sqrt_xy) + '.npy' ):
    print('compute polytopes')
    
    # calculate f values, gradients and matrix if a point is an inner or boundary point
    f_values_grid = (logits_mesh[:,0]-logits_mesh[:,1]).reshape(sqrt_xy,sqrt_xy)
    f_us = np.zeros((sqrt_xy,sqrt_xy))
    grad = np.zeros((sqrt_xy,sqrt_xy,4))
    # 0: boundary, 1:inner
    in_bd = np.zeros((sqrt_xy,sqrt_xy))
    for x in range(1,sqrt_xy-1):
      for y in range(1,sqrt_xy-1):
        f_us[x,y] = f_values_grid[x,y]
        grad[x,y,0] = (f_values_grid[x+1,y]-f_us[x,y]) / h
        grad[x,y,1] = (f_values_grid[x,y+1]-f_us[x,y]) / h
        grad[x,y,2] = (f_us[x,y]-f_values_grid[x-1,y]) / h
        grad[x,y,3] = (f_us[x,y]-f_values_grid[x,y-1]) / h
        
        if sqrt_xy==1000:
          if np.round(grad[x,y,0],2)==np.round(grad[x,y,2],2) and np.round(grad[x,y,1],1)==np.round(grad[x,y,3],1): 
            in_bd[x,y] = 1
        elif sqrt_xy==5000:
          if np.round(grad[x,y,0],1)==np.round(grad[x,y,2],1) and np.round(grad[x,y,1],0)==np.round(grad[x,y,3],0):
            in_bd[x,y] = 1
    
    # matrix for inner/boundary and for class decision as grid
    grid_in_bd = in_bd.reshape(sqrt_xy,sqrt_xy)
    grid_polytopes = np.asarray(ms.label(grid_in_bd, background=0), dtype='int64')
    print(np.unique(grid_polytopes))
    
    # smooth wrong small pixel segments
    if sqrt_xy==1000:
      list_smalls = []
      for i in np.unique(grid_polytopes):
        if np.count_nonzero(grid_polytopes==i) < 50:
          list_smalls.append(i)
      for x in range(sqrt_xy):
        for y in range(sqrt_xy):
          if grid_polytopes[x,y] == 0:
            nb = get_neighbors(grid_polytopes, x, y, 2)
            (values,counts) = np.unique(nb,return_counts=True)
            if len(values) == 2:
              grid_polytopes[x,y] = values[np.argmax(counts)]
          if grid_polytopes[x,y] in list_smalls:
            nb = get_neighbors(grid_polytopes, x, y, 2)
            (values,counts) = np.unique(nb,return_counts=True)
            counts[values==0] = -1
            counts[values==grid_polytopes[x,y]] = -1
            if len(values) > 2:
              grid_polytopes[x,y] = values[np.argmax(counts)]
      for x in range(sqrt_xy):
        for y in range(sqrt_xy):
          if grid_polytopes[x,y] == 0: 
            nb = get_neighbors(grid_polytopes, x, y, 2)
            (values,counts) = np.unique(nb,return_counts=True)
            if len(values) == 2:
              grid_polytopes[x,y] = values[np.argmax(counts)]
              
    elif sqrt_xy==5000:
      for x in range(sqrt_xy):
        for y in range(sqrt_xy):
          if grid_polytopes[x,y] == 0:
            nb = get_neighbors(grid_polytopes, x, y, 2)
            (values,counts) = np.unique(nb,return_counts=True)
            if len(values) == 2:
              grid_polytopes[x,y] = values[np.argmax(counts)]
      list_smalls = []
      for i in np.unique(grid_polytopes):
        if np.count_nonzero(grid_polytopes==i) < 500:
          list_smalls.append(i)
      for x in range(sqrt_xy):
        for y in range(sqrt_xy):        
          if grid_polytopes[x,y] in list_smalls:
            if y > 0.542*sqrt_xy:
              grid_polytopes[x,y] = 6
            elif y > 0.25*sqrt_xy:
              grid_polytopes[x,y] = 7
            else:
              grid_polytopes[x,y] = 0
      for x in range(sqrt_xy):
        for y in range(sqrt_xy):     
          if grid_polytopes[x,y] == 0: 
            nb = get_neighbors(grid_polytopes, x, y, 2)
            (values,counts) = np.unique(nb,return_counts=True)
            if len(values) == 2:
              grid_polytopes[x,y] = values[1] 
              
    np.save(os.path.join(save_path, 'data/f_us' + str(sqrt_xy)), f_us)
    np.save(os.path.join(save_path, 'data/grad' + str(sqrt_xy)), grad)
    np.save(os.path.join(save_path, 'data/grid_polytopes' + str(sqrt_xy)), grid_polytopes)
  
  grad = np.load(save_path + 'data/grad' + str(sqrt_xy) + '.npy')  
  grid_polytopes = np.load(save_path + 'data/grid_polytopes' + str(sqrt_xy) + '.npy')  
  print('unique poytopes', np.unique(grid_polytopes))

  if not os.path.isfile( save_path + 'data/min_max_grad' + str(sqrt_xy) + '.npy' ):
    print('compute a and alpha 2')
    
    # detection of class decision boundary
    def border_detection(CM):  
      CM_out = np.copy(CM)
      np.place(CM,CM>0,1)
      CM2 = CM[1:sqrt_xy-1,1:sqrt_xy-1] + CM[0:sqrt_xy-2,1:sqrt_xy-1] + CM[2:sqrt_xy,1:sqrt_xy-1]+ CM[1:sqrt_xy-1,2:sqrt_xy] + CM[1:sqrt_xy-1,0:sqrt_xy-2] + CM[0:sqrt_xy-2,0:sqrt_xy-2] + CM[2:sqrt_xy,0:sqrt_xy-2] + CM[0:sqrt_xy-2,2:sqrt_xy] + CM[2:sqrt_xy,2:sqrt_xy]
      CM2 = 0.5* ( ( CM2 * (CM2 == 9) ) / 9 )
      # 1: boundary, 1.5: inner, 0: else
      CM_out[1:sqrt_xy-1,1:sqrt_xy-1] += CM2 
      CM_out[:,0] = 0
      CM_out[:,sqrt_xy-1] = 0
      CM_out[0,:] = 0
      CM_out[sqrt_xy-1,:] = 0
      return CM_out
    
    # detection for class 1
    grid_decision = np.asarray(argmax_mesh.reshape(sqrt_xy,sqrt_xy), dtype='float64')
    tmp_dec_bd = border_detection(grid_decision)
    tmp_dec_bd[tmp_dec_bd!=1]=0
    # detection for class 0
    grid_decision[grid_decision==1]=-1
    grid_decision[grid_decision==0]=1
    grid_decision[grid_decision==-1]=0
    grid_dec_bd = border_detection(grid_decision)
    grid_dec_bd[grid_dec_bd!=1]=0
    # 1: boundary, 0: else
    grid_dec_bd[tmp_dec_bd==1]=1
    
    # list of polytopes with decision boundary
    list_db_polys = np.unique( grid_polytopes[grid_dec_bd==1] )[1:]
    print('polytopes with dec bd', list_db_polys)
    grad_top = grad[:,:,0]
    grad_right = grad[:,:,1]
    min_max_grad = [10000,-10000] 
    
    # calculate min gradint of polytopes with decision boundary
    for i in list_db_polys:
      tmp_indicies = np.where(grid_polytopes==i)
      tmp_grad_top = np.sum(grad_top[tmp_indicies]) / np.asarray(tmp_indicies).shape[1]
      tmp_grad_right = np.sum(grad_right[tmp_indicies]) / np.asarray(tmp_indicies).shape[1]
      min_max_grad[0] = min(min_max_grad[0], np.sqrt(tmp_grad_top**2 + tmp_grad_right**2) )
      min_max_grad[1] = max(min_max_grad[1], np.sqrt(tmp_grad_top**2 + tmp_grad_right**2) )
      
    np.save(os.path.join(save_path, 'data/grid_dec_bd' + str(sqrt_xy)), grid_dec_bd)
    np.save(os.path.join(save_path, 'data/min_max_grad' + str(sqrt_xy)), min_max_grad)

  grid_dec_bd = np.load(save_path + 'data/grid_dec_bd' + str(sqrt_xy) + '.npy')
  min_max_grad = np.load(save_path + 'data/min_max_grad' + str(sqrt_xy) + '.npy')
  
  param_a = (2 * np.sqrt(2)) / min_max_grad[0]
  
  print('min/max gradient', min_max_grad)
  print('value a', param_a)

  # plot poytopes
  if flag_polytopes:
    print('plot polytopes')
  
    cl1 = ['lavenderblush', 'thistle', 'plum', 'violet', 'orchid', 'hotpink', 'pink', 'lightpink', 'mediumpurple']
    cl2 = ['azure', 'lightcyan', 'paleturquoise', 'turquoise', 'cyan', 'deepskyblue', 'lightskyblue', 'cornflowerblue'] 
    colors_list = []
    for i in range(grid_polytopes.max()+1):
      colors_list.append(colorFader(rd.choice(cl1),rd.choice(cl2),rd.uniform(0, 0.5)))

    image_poly = np.zeros((sqrt_xy, sqrt_xy, 3))
    for x in range(sqrt_xy):
      new_x = sqrt_xy-1-x
      for y in range(sqrt_xy):
        if grid_polytopes[x,y] == 0:
          image_poly[new_x,y,:] = (255,255,255)
        else:
          image_poly[new_x,y,:] = colors_list[grid_polytopes[x,y]]
        if grid_dec_bd[x,y] == 1:
          image_poly[new_x,y,:] = (0,0,0)
        nb = get_neighbors(grid_dec_bd, x, y, 2)
        (values,counts) = np.unique(nb,return_counts=True)
        if 1 in values:
          image_poly[new_x,y,:] = (0,0,0)
    image_poly /= 255
    font_size = 16
    plt.clf()
    xy_ticks = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    x_ticks = np.arange(0,sqrt_xy+sqrt_xy/5,sqrt_xy/5)
    y_ticks = np.arange(sqrt_xy,0-sqrt_xy/5,-sqrt_xy/5)
    print(x_ticks, y_ticks)
    plt.xticks(x_ticks, (xy_ticks), fontsize = font_size)
    plt.yticks(y_ticks, (xy_ticks), fontsize = font_size)
    plt.imshow(image_poly, alpha=0.7) 
    plt.savefig(save_path + 'polytopes' + str(sqrt_xy) + '.png', dpi=400)
    plt.close()
    
  # plot f values as surface 
  if flag_surface:
    print('plot surfaces')
    f_us = np.load(save_path + 'data/f_us' + str(sqrt_xy) + '.npy')  
    
    if not os.path.exists( save_path + 'plot_surface' + str(sqrt_xy) + '/' ):
      os.makedirs( save_path + 'plot_surface' + str(sqrt_xy) + '/' )

    def plot_surface(x, y, z, name_plot):
      fig = plt.figure()
      ax = fig.gca(projection='3d') 
      ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap=sns.cubehelix_palette(start=.5, rot=-.6, dark=0.3, as_cmap=True, reverse=True), antialiased=False, alpha=0.6) 
      for ii in range(0,360,10):
        ax.view_init(elev=10., azim=ii)
        plt.savefig(save_path + 'plot_surface' + str(sqrt_xy) + '/' + name_plot + str(ii) + '.png', dpi=400)
      plt.close()
    
    surf_z0 = f_us.copy()
    surf_z0[surf_z0<0] = 0
    plot_surface(x_grid, y_grid, surf_z0, 'surface0_')
    surf_z1 = f_us.copy()
    surf_z1[surf_z1>0] = 0
    surf_z1 *= -1
    plot_surface(x_grid, y_grid, surf_z1, 'surface1_')
  
  # plot gradients as heatmap
  if flag_gradients:
    print('plot gradients')
    
    grad_top = grad[:,:,0]
    grad_right = grad[:,:,1]
    grads = np.sqrt(grad_top**2 + grad_right**2)
    grads[grid_polytopes==0] = 0
    grads = np.flipud(grads)
  
    plt.clf()
    plt.imshow(grads, cmap='gist_heat', interpolation='nearest')
    plt.colorbar()
    plt.savefig(save_path + 'gradients' + str(sqrt_xy) + '.png', dpi=400)
    plt.close()
    
  return param_a


def compute_epsilons_balls_alpha(x_mesh, x_test, adv_img_1, model_path, save_path, flag_balls=False, flag_eps=False):
  print('compute parameters 2')

  sqrt_xy = int(np.sqrt(x_mesh.shape[0])) 
  x_grid = x_mesh[:,0].reshape(sqrt_xy,sqrt_xy)
  y_grid = x_mesh[:,1].reshape(sqrt_xy,sqrt_xy)
  x_k = np.concatenate((x_test[:int(x_test.shape[0]/2)], adv_img_1), axis=0)
  
  min_max_grad = np.load(model_path + 'data/min_max_grad' + str(sqrt_xy) + '.npy')
  grid_polytopes = np.load(model_path + 'data/grid_polytopes' + str(sqrt_xy) + '.npy') 
  grid_dec_bd = np.load(model_path + 'data/grid_dec_bd' + str(sqrt_xy) + '.npy')
  
  param_alpha2 = 1 / (16* (1 + min_max_grad[1]/min_max_grad[0] )**2 )
  print('min/max gradient', min_max_grad)
  print('value alpha2', param_alpha2)
  
  if not os.path.isfile( model_path + 'data/dec_bd_points_parts' + str(sqrt_xy) + '.npy' ):
    print('compute decision boundary points')
    
    dec_bd_parts = grid_dec_bd.copy()
    dec_bd_parts[grid_polytopes==0] = 0
    
    dec_bd_parts = np.asarray(ms.label(dec_bd_parts, background=0), dtype='int64')

    # num points, num dec bd parts, xy mesh & xy
    dec_bd_points_parts = np.zeros((x_test.shape[0], int(len(np.unique(dec_bd_parts))-1), 4))

    for k in np.unique(dec_bd_parts)[1:]:

      # indizes decision boundary
      tmp_ind_bd = np.where(dec_bd_parts==k)
      tmp_ind_bd_array = np.asarray(tmp_ind_bd)
      xy_bd = np.zeros((tmp_ind_bd_array.shape[1],2))
      xy_bd[:,0] = x_grid[tmp_ind_bd]
      xy_bd[:,1] = y_grid[tmp_ind_bd]
      
      xy_tmp = np.zeros(xy_bd.shape)
      
      # calculate min distance points of data points and decision boundary
      for i in range(x_test.shape[0]):
        xy_tmp[:,0] = x_test[i,0]
        xy_tmp[:,1] = x_test[i,1]
        tmp_dist = np.sqrt( (xy_tmp[:,0]-xy_bd[:,0])**2 + (xy_tmp[:,1]-xy_bd[:,1])**2 )
        dec_bd_points_parts[i,k-1,0:2] = xy_bd[np.argmin(tmp_dist)]
        dec_bd_points_parts[i,k-1,2:4] = tmp_ind_bd_array[:,np.argmin(tmp_dist)]
        
    print(x_test[0:10], dec_bd_points_parts[0:10])
    
    np.save(os.path.join(model_path, 'data/dec_bd_points_parts' + str(sqrt_xy)), dec_bd_points_parts)
    
  dec_bd_points_parts = np.load(model_path + 'data/dec_bd_points_parts' + str(sqrt_xy) + '.npy')
  
  if not os.path.isfile( save_path + 'data/dec_bd_points.npy' ):
    print('compute epsilons')
  
    dec_bd_points = np.zeros((x_test.shape[0], 4))
    xy_tmp = np.zeros((dec_bd_points_parts.shape[1], 2))

    for i in range(x_test.shape[0]):
      
      xy_tmp[:,0] = x_k[i,0]
      xy_tmp[:,1] = x_k[i,1]
      
      dist_tmp = np.sqrt( (xy_tmp[:,0]-dec_bd_points_parts[i,:,0])**2 + (xy_tmp[:,1]-dec_bd_points_parts[i,:,1])**2 )
      
      dec_bd_points[i,0:2] = dec_bd_points_parts[i,np.argmin(dist_tmp),0:2]
      dec_bd_points[i,2:4] = dec_bd_points_parts[i,np.argmin(dist_tmp),2:4]
  
    np.save(os.path.join(save_path, 'data/dec_bd_points'), dec_bd_points)
      
  dec_bd_points = np.load(save_path + 'data/dec_bd_points.npy')  
    
  epsilons = np.sqrt( (x_k[:,0]-dec_bd_points[:,0])**2 + (x_k[:,1]-dec_bd_points[:,1])**2 )
  epsilons += 0.0002
  
  param_b = (8*epsilons) / min_max_grad[0]
  np.save(os.path.join(save_path, 'data/param_b'), param_b)
  np.save(os.path.join(save_path, 'data/epsilons_calc'), epsilons)
  
  print('epsilons (min/max/mean)', epsilons.min(), epsilons.max(), np.mean(epsilons))
  print('param_b (min/max/mean)', param_b.min(), param_b.max(), np.mean(param_b))
  
  if not os.path.isfile( save_path + 'data/eps3_ball.npy' ):
    print('compute 3 epsilons balls')

    eps3_ball = np.zeros((x_test.shape[0]))
    polytopes_points = np.zeros((x_test.shape[0],2))
    # calculate min distance of decision boundary and polytopes without
    for i in range(x_test.shape[0]):
      nb = get_neighbors(grid_polytopes, int(dec_bd_points[i,2]), int(dec_bd_points[i,3]), 3)
      (values,counts) = np.unique(nb,return_counts=True)
      if len(values) == 1 and values[0] == 0:
        print(i, dec_bd_points[i,0:2])
        print('Fehler')
        exit()
        
      # polytopes without Q's
      grid_polytopes_tmp = grid_polytopes.copy()
      for j in range(len(values)):
        grid_polytopes_tmp[grid_polytopes==values[j]] = 0
      
      # indizes polytopes without Q's
      tmp_ind_poly = np.where(grid_polytopes_tmp>0)
      xy_non_bd_poly = np.zeros((np.asarray(tmp_ind_poly).shape[1],2))
      xy_non_bd_poly[:,0] = x_grid[tmp_ind_poly]
      xy_non_bd_poly[:,1] = y_grid[tmp_ind_poly]
      
      xy_db = np.zeros(xy_non_bd_poly.shape)
      
      xy_db[:,0] = dec_bd_points[i,0]
      xy_db[:,1] = dec_bd_points[i,1]
      tmp_dist = np.sqrt( (xy_db[:,0]-xy_non_bd_poly[:,0])**2 + (xy_db[:,1]-xy_non_bd_poly[:,1])**2 )
      polytopes_points[i,0:2] = xy_non_bd_poly[np.argmin(tmp_dist)]
      eps3_ball[i] = tmp_dist.min()
      
    print('dist decision bd and polys without (min/max/mean):', eps3_ball.min(), eps3_ball.max(), np.mean(eps3_ball))
    
    np.save(os.path.join(save_path, 'data/eps3_ball'), eps3_ball)
    np.save(os.path.join(save_path, 'data/polytopes_points'), polytopes_points)
  
  # plot balls
  if flag_balls:
    print('plot balls')
    
    dec_bd_points = np.load(save_path + 'data/dec_bd_points.npy')
    polytopes_points = np.load(save_path + 'data/polytopes_points.npy')
  
    tmp_ind_poly_bd = np.where(grid_polytopes==0)
    poly_bd = np.zeros((np.asarray(tmp_ind_poly_bd).shape[1],2))
    poly_bd[:,0] = x_grid[tmp_ind_poly_bd]
    poly_bd[:,1] = y_grid[tmp_ind_poly_bd]
    
    tmp_ind_dec_bd = np.where(grid_dec_bd==1)
    poly_dec_bd = np.zeros((np.asarray(tmp_ind_dec_bd).shape[1],2))
    poly_dec_bd[:,0] = x_grid[tmp_ind_dec_bd]
    poly_dec_bd[:,1] = y_grid[tmp_ind_dec_bd]
    
    colors_mesh = {0:'aliceblue', 1:'lavenderblush'}
    def plot_balls(x_data, dec_bd_points_data, polytopes_points_data, name):
      fig, ax = plt.subplots()
      plt.scatter(poly_dec_bd[:,0], poly_dec_bd[:,1], color='gray', alpha=0.2, marker='.')
      plt.scatter(poly_bd[:,0], poly_bd[:,1], color='black', alpha=0.2, marker='.')
      plt.scatter(x_data[:,0], x_data[:,1], color='blue', alpha=0.5, marker='.') 
      plt.scatter(dec_bd_points_data[:,0], dec_bd_points_data[:,1], color='green', alpha=0.3, marker='.')
      plt.scatter(polytopes_points_data[:,0], polytopes_points_data[:,1], color='orange', alpha=0.3, marker='.')
      plt.savefig(save_path + 'balls_' + name + '.png', dpi=400) 
    
    plot_balls(x_k[:int(x_test.shape[0]/2)], dec_bd_points[:int(x_test.shape[0]/2)], polytopes_points[:int(x_test.shape[0]/2)], 'X')
    
    plot_balls(x_k[int(x_test.shape[0]/2):], dec_bd_points[int(x_test.shape[0]/2):], polytopes_points[int(x_test.shape[0]/2):], 'Xbar')
    
  # plot epsilons
  if flag_eps:
    print('plot epsilons')
    
    font_size = 16
  
    f, ax1 = plt.subplots(figsize=(7,4))
    sns.violinplot(data=[epsilons[:int(x_test.shape[0]/2)],epsilons[int(x_test.shape[0]/2):]], palette='Set3', inner='box', bw=0.3, cut=0, linewidth=1.0, scale='width', orient='h')
    sns.despine(left=True)
    ax1.set_yticklabels([ '$\\mathcal{X}$', '$D_p(\\bar{\\mathcal{X}})$' ])
    plt.xscale('log')
    plt.xlabel('$\\epsilon$', size = 16)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    plt.savefig(save_path + 'vio_stat_log.png', bbox_inches='tight')
    plt.close()
    
    f, ax1 = plt.subplots(figsize=(7,4))
    sns.violinplot(data=[epsilons[:int(x_test.shape[0]/2)],epsilons[int(x_test.shape[0]/2):]], palette='Set3', inner='box', bw=0.3, cut=0, linewidth=1.0, scale='width', orient='h')
    sns.despine(left=True)
    ax1.set_yticklabels([ '$\\mathcal{X}$', '$D_p(\\bar{\\mathcal{X}})$' ])
    plt.xlabel('$\\epsilon$', size = 16)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    plt.savefig(save_path + 'vio_stat.png', bbox_inches='tight')
    plt.close()
      
  return param_alpha2


def test_balls(x_k, adv_img_2, logits_0, logits_1, logits_2, save_path):
  print('compute successful attacks')
  
  logits_0 = np.argmax(logits_0, axis=1)
  logits_1 = np.argmax(logits_1, axis=1)
  logits_2 = np.argmax(logits_2, axis=1)
  len_x = int(x_k.shape[0]/2)
  
  dec_bd_points = np.load(save_path + 'data/dec_bd_points.npy')
  eps3_ball_1 = np.load(save_path + 'data/eps3_ball.npy') 
  eps_2 = np.load(save_path + 'data/epsilons_calc.npy') 
  
  eps_1 = eps3_ball_1/3 + 0.0002
  eps3_ball_2 = (eps_2 - 0.0002)*3
  
  dist_x0k = np.sqrt( (x_k[:,0]-dec_bd_points[:,0])**2 + (x_k[:,1]-dec_bd_points[:,1])**2 )
  ind_x0_1 = dist_x0k[:len_x] < eps_1[:len_x]
  ind_xk_1 = dist_x0k[len_x:] < eps_1[len_x:]
  ind_x0_2 = dist_x0k[:len_x] >= eps_1[:len_x]
  ind_xk_2 = dist_x0k[len_x:] >= eps_1[len_x:]
  
  dist_xj = np.sqrt( (adv_img_2[:,0]-dec_bd_points[:,0])**2 + (adv_img_2[:,1]-dec_bd_points[:,1])**2 )
  ind_x0j_1 = dist_xj[:len_x] < eps3_ball_1[:len_x]
  ind_xkj_1 = dist_xj[len_x:] < eps3_ball_1[len_x:]
  ind_x0j_2 = dist_xj[:len_x] < eps3_ball_2[:len_x]
  ind_xkj_2 = dist_xj[len_x:] < eps3_ball_2[len_x:]
  
  ind_change_0 = logits_0[:len_x] != logits_2[:len_x]
  ind_change_k1 = logits_0[len_x:] != logits_1
  ind_change_k2 = logits_1 != logits_2[len_x:]
  ind_change_k = np.logical_and(ind_change_k1, ind_change_k2)

  result_path = os.path.join(save_path, 'results_balls.txt')
  with open(result_path, 'wt') as fi:
    print('eps_1 = calculated by theorem 1, 3eps_1 = correspondig 3 epsilon ball', file=fi)
    print('eps_2 = calculated by distance to stationary point, 3eps_2 = correspondig 3 epsilon ball', file=fi)
    print(' ', file=fi)
    print('number of:', file=fi)
    print(' ', file=fi)
    print('x0 in eps_1: ', np.sum(np.logical_and(ind_x0_1,ind_change_0)), file=fi)
    print('and x0j in 3eps_1: ', np.sum( np.logical_and(np.logical_and(ind_x0_1, ind_x0j_1), ind_change_0) ), file=fi)
    print(' ', file=fi)
    print('x0 in eps_2 (not in eps_1): ', np.sum(np.logical_and(ind_x0_2,ind_change_0)), file=fi)
    print('and x0j in 3eps_1: ', np.sum( np.logical_and(np.logical_and(ind_x0_2, ind_x0j_1), ind_change_0) ), file=fi)
    print('and x0j in 3eps_2: ', np.sum( np.logical_and(np.logical_and(ind_x0_2, ind_x0j_2), ind_change_0) ), file=fi)
    print(' ', file=fi)
    print('xk in eps_1: ', np.sum(np.logical_and(ind_xk_1,ind_change_k2)), file=fi)
    print('and x0j in 3eps_1: ', np.sum( np.logical_and(np.logical_and(ind_xk_1, ind_xkj_1), ind_change_k2) ), file=fi)
    print('xk in eps_1 (successful first attack): ', np.sum( np.logical_and(ind_xk_1, ind_change_k) ), file=fi)
    print('and x0j in 3eps_1: ', np.sum( np.logical_and(np.logical_and(ind_xk_1, ind_xkj_1), ind_change_k) ), file=fi)
    print(' ', file=fi)
    print('xk in eps_2 (not in eps_1): ', np.sum(np.logical_and(ind_xk_2,ind_change_k2)), file=fi)
    print('and x0j in 3eps_1: ', np.sum( np.logical_and(np.logical_and(ind_xk_2, ind_xkj_1), ind_change_k2) ), file=fi)
    print('and x0j in 3eps_2: ', np.sum( np.logical_and(np.logical_and(ind_xk_2, ind_xkj_2), ind_change_k) ), file=fi)
    print('xk in eps_2 (successful first attack): ', np.sum( np.logical_and(ind_xk_2, ind_change_k) ), file=fi)
    print('and x0j in 3eps_1: ', np.sum( np.logical_and(np.logical_and(ind_xk_2, ind_xkj_1), ind_change_k) ), file=fi)
    print('and x0j in 3eps_2: ', np.sum( np.logical_and(np.logical_and(ind_xk_2, ind_xkj_2), ind_change_k) ), file=fi)


def compute_returnees(logits0, logits1, logits2, logits0x, logits2x, save_path):
  print('compute returnees')
  
  logits0 = np.argmax(logits0, axis=1)
  logits1 = np.argmax(logits1, axis=1)
  logits2 = np.argmax(logits2, axis=1)
  logits0x = np.argmax(logits0x, axis=1)
  logits2x = np.argmax(logits2x, axis=1)
  
  returnees = logits2==logits0
  
  result_path = os.path.join(save_path, 'results_returnees.txt')
  with open(result_path, 'wt') as fi:
    print('returnees', np.sum(returnees), 'of', logits0.shape[0], file=fi)
    print('returnees rate', np.sum(returnees)/logits0.shape[0], file=fi)
    print(' ', file=fi)
    print('class change predicted label to once attacked:', np.sum(logits1!=logits0), file=fi)
    print('class change once attacked to counter attacked:', np.sum(logits1!=logits2), file=fi)
    print(' ', file=fi)
    print('class change predicted label to counter attacked:', np.sum(logits0x!=logits2x), file=fi)
    
  
def modify_D(D_p, D_p_p, logits0, logits1, logits2, logits0x, logits2x):
  
  print('shape Dp and Dpp:', np.shape(D_p), np.shape(D_p_p))
  
  max_both = max(D_p.max(), D_p_p.max())
  D_p /= max_both
  D_p_p /= max_both
  
  logits0 = np.argmax(logits0, axis=1)
  logits1 = np.argmax(logits1, axis=1)
  logits2 = np.argmax(logits2, axis=1)
  logits0x = np.argmax(logits0x, axis=1)
  logits2x = np.argmax(logits2x, axis=1)
  
  D_p_mod = D_p[logits0x != logits2x]
  D_p_p_mod = D_p_p[np.logical_and(logits2==logits0, logits1!=logits0)]
  
  print('shape Dp and Dpp success:', np.shape(D_p_mod), np.shape(D_p_p_mod))
  
  return D_p_mod, D_p_p_mod
    

def plot_violins(D_p, D_p_p, save_path):
  print('plot violins')
  
  font_size = 16
  
  f, ax1 = plt.subplots(figsize=(7,4))
  sns.violinplot(data=[D_p,D_p_p], palette='Set3', inner='box', bw=0.3, cut=0, linewidth=1.0, scale='width', orient='h')
  sns.despine(left=True)
  ax1.set_yticklabels([ '$D_p$ (original)', '$\\bar{D}_{p,p}$ (attacked)' ])
  plt.xlabel('normalized distances', size = 16)
  plt.xticks(fontsize=font_size)
  plt.yticks(fontsize=font_size)
  plt.tight_layout()
  plt.savefig(save_path + 'vio.png', bbox_inches='tight')
  plt.close()
  

def threshold_evaluation(D_p, D_p_p, save_path):
  print('compute metrics for D')
  
  # threshold value to be maximized, threshold, tp, fn, tn, fp, accuracy, recall, precision
  t1_array = np.zeros((9))
  t2_array = np.zeros((9))
  
  for t in np.arange(0,1,0.001):
    
    tp = np.sum(D_p_p < t)
    fn = np.sum(D_p_p >= t)
    tn = np.sum(D_p >= t)
    fp = np.sum(D_p < t)
    acc = (tp+tn)/(tp+fn+tn+fp)
    rec = tp/(tp+fn)
    if tp+fp > 0:
      pre = tp/(tp+fp)
    else:
      pre = 0.0
    
    t1 = 0.5 * rec + 0.25 * acc + 0.25 * pre
    if t1 > t1_array[0]:
      t1_array[0] = t1
      t1_array[1] = t
      t1_array[2] = tp
      t1_array[3] = fn
      t1_array[4] = tn
      t1_array[5] = fp
      t1_array[6] = acc
      t1_array[7] = rec
      t1_array[8] = pre
      
    if acc > t2_array[0]:
      t2_array[0] = acc
      t2_array[1] = t
      t2_array[2] = tp
      t2_array[3] = fn
      t2_array[4] = tn
      t2_array[5] = fp
      t2_array[6] = acc
      t2_array[7] = rec
      t2_array[8] = pre
      
  result_path = os.path.join(save_path, 'results_thresholding.txt')
  with open(result_path, 'wt') as fi:
    print('results threshold 1', file=fi)
    print('threshold value to be maximized, threshold, tp, fn, tn, fp, accuracy, recall, precision', file=fi)
    print(t1_array, file=fi)
    print(' ', file=fi)
    print('results threshold 2', file=fi)
    print('threshold value to be maximized, threshold, tp, fn, tn, fp, accuracy, recall, precision', file=fi)
    print(t2_array, file=fi)
    

def compute_auroc(D_p, D_p_p, save_path=[], name='results_thresholding.txt'):
  print('compute auroc value')
  
  labels = np.zeros((len(D_p)+len(D_p_p)))
  labels[:len(D_p)] = 1
  
  predictions = np.zeros((len(D_p)+len(D_p_p)))
  predictions[:len(D_p)] = D_p
  predictions[len(D_p):] = D_p_p

  fpr, tpr, _ = roc_curve(labels, predictions)
  auc_val = auc(fpr, tpr)
  
  if save_path != []:
    result_path = os.path.join(save_path, name)
    with open(result_path, 'a') as fi:
      print(' ', file=fi)
      print('AUROC: ', auc_val, file=fi)
  
  return auc_val
      

def plot_results_models(num_samples, dataset, model_path):
  print('plot results of all models')
  
  save_path = model_path + '../all/' +  dataset + '/'
  if not os.path.exists( save_path ):
    os.makedirs( save_path )
  
  models = sorted(os.listdir( model_path + '../' ))
  
  # auroc, successful first attack, successful second attack, returnees, num iterations
  data_array = np.zeros((len(models),5))
  if dataset == 'dims':
    # auroc, successful first attack, successful second attack, returnees, num iterations, num dims
    data_array = np.zeros((len(models),6))
    
  # 5,25,50,75,95% quantils, num iterations
  quantile_array = np.zeros((len(models)+1,6))
  def fill_quantiles(eps, count, num):
    quantile_array[count,0] = np.percentile(eps, 5)
    quantile_array[count,1] = np.percentile(eps, 25)
    quantile_array[count,2] = np.percentile(eps, 50)
    quantile_array[count,3] = np.percentile(eps, 75)
    quantile_array[count,4] = np.percentile(eps, 95)
    quantile_array[count,5] = num
  
  counter = 0
  for model in models:
    if dataset in model:
      
      if os.path.isfile(model_path + '../' + model + '/results_thresholding.txt'):
        with open(model_path + '../' + model + '/results_thresholding.txt', 'r') as f:
          lines = f.read().strip().split('\n')
          data_array[counter,0] = float(lines[-1].split(':')[1])
      else:
        data_array[counter,0] = np.nan
        
      with open(model_path + '../' + model + '/results_returnees.txt', 'r') as f:
        lines = f.read().strip().split('\n')
        data_array[counter,1] = float(lines[3].split(':')[1]) / num_samples
        data_array[counter,2] = float(lines[4].split(':')[1]) / num_samples
        data_array[counter,3] = float(lines[0].split(' ')[1]) / num_samples
        
      data_array[counter,4] = int(model.split('_')[1])
      if dataset == 'dims':
        data_array[counter,5] = int(model.split('_')[-1])
      
      if dataset == 'moon':
        eps = np.load(model_path + '../' + model + '/data/epsilons_calc.npy') 
        if counter == 0:
          fill_quantiles(eps, 0, 0)
        fill_quantiles(eps, counter+1, int(model.split('_')[-1]))

      counter += 1
  
  def selection_sort(x, index):
    for i in range(x.shape[0]):
      swap = i + np.argmin(x[i:,index])
      x_tmp = x[i].copy()
      x[i] = x[swap].copy()
      x[swap] = x_tmp
    return x
  
  size_font = 20
  
  if dataset == 'dims':
    
    data_array = data_array[:counter,:]
    num_dims_diff = np.unique(data_array[:,5])
    num_it = np.unique(data_array[:,4])
    
    num_steps = np.arange(1, len(num_dims_diff)+1)
    color_map = ['tab:cyan', 'teal', 'turquoise']
    f1 = plt.figure(figsize=(6,4))
    plt.clf()
    for i,n in zip(num_it, range(len(num_it))):
      tmp_data_array = data_array[data_array[:,4]==i,:]
      tmp_data_array = selection_sort(tmp_data_array, 5)
      plt.plot(num_steps, tmp_data_array[:,0], color=color_map[n], marker='o', label=str(int(tmp_data_array[0,4])))  
    plt.xticks(num_steps[::2], np.asarray(tmp_data_array[:,5][::2],dtype='int64'), fontsize = size_font)
    plt.yticks(fontsize = size_font)
    plt.xlabel('dimension', fontsize=size_font)
    plt.ylabel('AUROC', fontsize=size_font, labelpad=-1) 
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode='expand', borderaxespad=0., prop={'size': size_font}, numpoints=1)
    plt.grid()
    f1.savefig(save_path + 'aurocs.png', bbox_inches='tight', dpi=400)
    plt.close()
    
    color_map = ['purple', 'hotpink', 'mediumvioletred']
    f1 = plt.figure(figsize=(6,4))
    plt.clf()
    for i,n in zip(num_it, range(len(num_it))):
      tmp_data_array = data_array[data_array[:,4]==i,:]
      tmp_data_array = selection_sort(tmp_data_array, 5)
      plt.plot(num_steps, tmp_data_array[:,3], color=color_map[n], marker='o', label=str(int(tmp_data_array[0,4])), alpha=0.4)  
    plt.xticks(num_steps[::2], np.asarray(tmp_data_array[:,5][::2],dtype='int64'), fontsize = size_font)
    plt.yticks(fontsize = size_font)
    plt.xlabel('dimension', fontsize=size_font)
    plt.ylabel('return rate', fontsize=size_font, labelpad=-1) 
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode='expand', borderaxespad=0., prop={'size': size_font}, numpoints=1)
    plt.grid()
    f1.savefig(save_path + 'returns.png', bbox_inches='tight', dpi=400)
    plt.close()
    
    font_size = 50
    for i in num_it:
      tmp_data_array = data_array[data_array[:,4]==i,:]
      tmp_data_array = selection_sort(tmp_data_array, 5)
      num_dim = tmp_data_array.shape[0]

      t = np.arange(0.0, 0.7, 0.01)
      f = plt.subplots(figsize=(num_dim*4,10))
      for j in range(num_dim):
        path_tmp = CONFIG.SAVE_PATH.split('_')[0] + '_' + str(int(i)) + '_' + str(int(tmp_data_array[j,5])) + '/data/'
        x_test = np.load(path_tmp + 'x_test.npy')
        x_adv = np.load(path_tmp + 'images_once_attacked.npy')
        x_adv_adv = np.load(path_tmp + 'images_twice_attacked.npy')
        dist1 = np.squeeze(np.sqrt(np.sum(np.square(x_test[:num_samples]-x_adv_adv[:num_samples]), axis=(1,2,3))))
        dist2 = np.squeeze(np.sqrt(np.sum(np.square(x_adv-x_adv_adv[num_samples:]), axis=(1,2,3))))
        labels = np.zeros((num_samples*2),dtype='int32')
        labels[num_samples:] = 1
        
        df_mesh = DataFrame(dict(y=np.concatenate((np.log10(dist1),np.log10(dist2)),axis=-1), l=labels))
        df_mesh["x"] = ""
  
        ax = plt.subplot(1,num_dim+1,j+1)
        g = sns.violinplot(data=df_mesh, x='x', y='y', hue='l', split=True, scale='width', palette={0: "paleturquoise", 1: "bisque"}, inner='quart', bw=0.3, linewidth=4)
        sns.despine(left=True)
        ax.get_legend().remove()
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ax.yaxis.set_ticks([np.log10(x) for p in range(-6,1) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
        ax.set_title(label=str(int(tmp_data_array[j,5])), size=font_size)
        ax.set_ylabel('')
        if j > 0:
          ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
      plt.tight_layout()
      plt.savefig(save_path + 'vio' + str(int(i)) + '.png', bbox_inches='tight')
      plt.close()
    
  else:
    
    data_array = data_array[:counter,:]
    data_array = selection_sort(data_array, 4)
    
    num_steps = np.arange(1, len(data_array[:,4])+1)
    color_map = ['tab:cyan', 'orchid', 'steelblue','purple']
    
    f1 = plt.figure(figsize=(6,4))
    plt.clf()
    plt.plot(num_steps, data_array[:,0], color=color_map[0], marker='o')#, label=)   
    plt.xticks(num_steps[::2], np.asarray(data_array[:,4][::2],dtype='int64'), fontsize = size_font)
    plt.yticks(fontsize = size_font)
    plt.xlabel('number of iterations', fontsize=size_font)
    plt.ylabel('AUROC', fontsize=size_font, labelpad=-1) 
    plt.grid()
    save_path = model_path + '../all/' +  dataset + '/'
    f1.savefig(save_path + 'auroc.png', bbox_inches='tight', dpi=400)
    plt.close()
    
    f1 = plt.figure(figsize=(6,4))
    plt.clf()
    plt.plot(num_steps, data_array[:,3], color=color_map[1], marker='o')#, label=)   
    plt.xticks(num_steps[::2], np.asarray(data_array[:,4][::2],dtype='int64'), fontsize = size_font)
    plt.yticks(fontsize = size_font)
    plt.xlabel('number of iterations', fontsize=size_font)
    plt.ylabel('return rate', fontsize=size_font, labelpad=-1) 
    plt.grid()
    save_path = model_path + '../all/' +  dataset + '/'
    f1.savefig(save_path + 'returnees.png', bbox_inches='tight', dpi=400)
    plt.close()
    
    f1 = plt.figure(figsize=(6,4))
    plt.clf()
    plt.plot(num_steps, data_array[:,1], color=color_map[2], marker='o', label='first attack') 
    plt.plot(num_steps, data_array[:,2], color=color_map[3], marker='o', label='second attack') 
    plt.xticks(num_steps[::2], np.asarray(data_array[:,4][::2],dtype='int64'), fontsize = size_font)
    plt.yticks(fontsize = size_font)
    plt.xlabel('number of iterations', fontsize=size_font)
    plt.ylabel('rate of successful attacks', fontsize=size_font, labelpad=-1) 
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode='expand', borderaxespad=0., prop={'size': 16}, numpoints=1)
    plt.grid()
    save_path = model_path + '../all/' +  dataset + '/'
    f1.savefig(save_path + 'success.png', bbox_inches='tight', dpi=400)
    plt.close()
  
  if dataset == 'moon':
    
    quantile_array = quantile_array[:counter+1,:]
    quantile_array = selection_sort(quantile_array, 5)
    
    num_steps = np.arange(1, len(quantile_array[:,5])+1)
    color_map = ['lightskyblue', 'dodgerblue', 'navy', 'cornflowerblue','lightsteelblue']

    f1 = plt.figure(figsize=(8,5))
    plt.clf()
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.handlelength'] = 0
    plt.fill_between(num_steps, quantile_array[:,0], quantile_array[:,4], color='aliceblue', alpha=0.9 ) 
    plt.fill_between(num_steps, quantile_array[:,1], quantile_array[:,3], color='lightskyblue', alpha=0.4 ) 
    plt.plot(num_steps, quantile_array[:,0], color=color_map[0], marker='o', label='$5\\%$') 
    plt.plot(num_steps, quantile_array[:,1], color=color_map[1], marker='o', label='$25\\%$') 
    plt.plot(num_steps, quantile_array[:,2], color=color_map[2], marker='o', label='$50\\%$') 
    plt.plot(num_steps, quantile_array[:,3], color=color_map[3], marker='o', label='$75\\%$') 
    plt.plot(num_steps, quantile_array[:,4], color=color_map[4], marker='o', label='$95\\%$') 
    plt.xticks(num_steps[::2], np.asarray(quantile_array[:,5][::2],dtype='int64'), fontsize = size_font)
    plt.yticks(fontsize = size_font)
    plt.xlabel('number of iterations', fontsize=size_font)
    plt.ylabel('$\\epsilon$', fontsize=size_font, labelpad=-1) 
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode='expand', borderaxespad=0., prop={'size': size_font}, numpoints=1)
    plt.grid()
    save_path = model_path + '../all/' +  dataset + '/'
    f1.savefig(save_path + 'quantiles.png', bbox_inches='tight', dpi=400)
    plt.close()
    
  
  
