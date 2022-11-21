import numpy as np
import quaternion
import tensorflow as tf
from tqdm import tqdm
import pywt
from scipy.spatial.transform import Rotation as R


def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))


def generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi):
    cur_l = np.array(init_l)
    cur_theta = np.array(init_theta)
    cur_psi = np.array(init_psi)
    pred_l = []
    pred_l.append(np.array(cur_l))

    for [delta_l, delta_theta, delta_psi] in zip(y_delta_l, y_delta_theta, y_delta_psi):
        cur_theta = cur_theta + delta_theta
        cur_psi = cur_psi + delta_psi
        cur_l[0] = cur_l[0] + delta_l * np.sin(cur_theta) * np.cos(cur_psi)
        cur_l[1] = cur_l[1] + delta_l * np.sin(cur_theta) * np.sin(cur_psi)
        cur_l[2] = cur_l[2] + delta_l * np.cos(cur_theta)
        pred_l.append(np.array(cur_l))

    return np.reshape(pred_l, (len(pred_l), 3))


def get_quat_from_rot_vec(a,b,c):
  return R.from_rotvec(np.array([a,b,c])).as_quat()

def causal_mask(batch, size):
    x, y = tf.expand_dims(tf.range(size),1), tf.range(size)
    mask = x>=y
    mask = tf.reshape(mask, (1, size, size))
    mask = tf.tile(mask, (batch, 1, 1))
    return mask

def create_cwt_images(X, scale_range, wavelet_name = 'morl', rescale=True, upsample=False, rescale_steps=30, rescale_scales=30):
    samples = X.shape[0]
    steps = X.shape[1] 
    sensors = X.shape[2] 

    x_dim = steps
    y_dim = len(scale_range)
    
    if upsample:
        x_dim = x_dim * 2

    if rescale:
        x_dim = rescale_steps
        y_dim = rescale_scales
    
    
    # prepare the output array
    X_cwt = np.ndarray(shape=(samples, y_dim, x_dim, sensors), dtype = 'float32')
    
    for sample in tqdm(range(samples),desc = 'Creating CWT transformed vectors: '):
        
        for sensor in range(sensors):
            series = X[sample, :, sensor]
            # upsample
            if upsample:
                x = np.linspace(1,len(series),len(series)*2)
                xp = np.arange(1,len(series)+1,dtype='float32')
                series = np.interp(x, xp, series)
            # continuous wavelet transform 
            coeffs, _ = pywt.cwt(series, scale_range, wavelet_name)
            # resize the 2D cwt coeffs
            if rescale:
                coeffs = resize(coeffs, (y_dim, x_dim), mode = 'constant')

            X_cwt[sample,:,:,sensor] = coeffs
            
    return X_cwt
