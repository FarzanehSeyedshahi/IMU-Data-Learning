import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate
import glob, os

from util import *
from tensorflow.keras.utils import Sequence


def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def load_euroc_mav_dataset(imu_data_filename, gt_data_filename):
    gt_data = pd.read_csv(gt_data_filename).values    
    imu_data = pd.read_csv(imu_data_filename).values

    gyro_data = interpolate_3dvector_linear(imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    acc_data = interpolate_3dvector_linear(imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    pos_data = gt_data[:, 1:4]
    ori_data = gt_data[:, 4:8]

    return gyro_data, acc_data, pos_data, ori_data


def load_oxiod_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]
    
    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return gyro_data, acc_data, pos_data, ori_data


def load_dataset_9d_quat(gyro_data, acc_data, mag_data, pos_data, ori_data, window_size, stride):

    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, :]

    x_gyro = []
    x_acc = []
    x_mag = []
    y_delta_p = []
    y_delta_q = []

    # print(np.array(pos_data).shape, pos_data[0])
    # print(np.array(ori_data).shape, ori_data[0])
    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])
        x_mag.append(mag_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))

    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    x_mag = np.reshape(x_mag, (len(x_mag), x_mag[0].shape[0], x_mag[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))
    return [x_gyro, x_acc, x_mag], [y_delta_p, y_delta_q], init_p, init_q

def load_dataset_9d_rm(gyro_data, acc_data, mag_data, pos_data, ori_data, window_size, stride):
    if np.isnan(ori_data).any():print("***********")


    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, :]

    x_gyro = []
    x_acc = []
    x_mag = []
    y_delta_p = []
    y_delta_rm = []
    # print(np.array(pos_data).shape, pos_data[0])
    # print(np.array(ori_data).shape, ori_data[0])
    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])
        x_mag.append(mag_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        rm_a = ori_data[idx + window_size//2 - stride//2, :]
        rm_b = ori_data[idx + window_size//2 + stride//2, :]

        # Just Considering the exact position
        delta_p = p_b.T - p_a.T

        # delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)
        delta_rm = rm_a.T * rm_b

        y_delta_p.append(delta_p)
        y_delta_rm.append(delta_rm)

    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    x_mag = np.reshape(x_mag, (len(x_mag), x_mag[0].shape[0], x_mag[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_rm = np.reshape(y_delta_rm, (len(y_delta_rm), y_delta_rm[0].shape[0]))

    return [x_gyro, x_acc, x_mag], [y_delta_p, y_delta_rm], init_p, init_q


def load_oxiod_9D_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]
    mag_data = imu_data[:, 13:16]
    
    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return gyro_data, acc_data, mag_data, pos_data, ori_data

def force_quaternion_uniqueness(q):

    q_data = quaternion.as_float_array(q)

    if np.absolute(q_data[0]) > 1e-05:
        if q_data[0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[1]) > 1e-05:
        if q_data[1] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[2]) > 1e-05:
        if q_data[2] < 0:
            return -q
        else:
            return q
    else:
        if q_data[3] < 0:
            return -q
        else:
            return q

# def force_rotation_matrix_uniqueness(rm):

#     if np.absolute(rm[1]) > 1e-05:
#         if rm[0] < 0:
#             return -rm
#         else:
#             return rm
#     elif np.absolute(rm[2]) > 1e-05:
#         if rm[1] < 0:
#             return -rm
#         else:
#             return rm
#     else:
#         if rm[2] < 0:
#             return -rm
#         else:
#             return rm


def cartesian_to_spherical_coordinates(point_cartesian):
    delta_l = np.linalg.norm(point_cartesian)

    if np.absolute(delta_l) > 1e-05:
        theta = np.arccos(point_cartesian[2] / delta_l)
        psi = np.arctan2(point_cartesian[1], point_cartesian[0])
        return delta_l, theta, psi
    else:
        return 0, 0, 0


def load_dataset_6d_rvec(imu_data_filename, gt_data_filename, window_size=200, stride=10):

    #imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    #gt_data = np.genfromtxt(gt_data_filename, delimiter=',')
    
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    #imu_data = imu_data[1200:-300]
    #gt_data = gt_data[1200:-300]
    
    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    init_q = quaternion.from_float_array(ori_data[window_size//2 - stride//2, :])
    
    init_rvec = np.empty((3, 1))
    cv2.Rodrigues(quaternion.as_rotation_matrix(init_q), init_rvec)

    init_tvec = pos_data[window_size//2 - stride//2, :]

    x = []
    y_delta_rvec = []
    y_delta_tvec = []

    for idx in range(0, gyro_acc_data.shape[0] - window_size - 1, stride):
        x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])

        tvec_a = pos_data[idx + window_size//2 - stride//2, :]
        tvec_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        rmat_a = quaternion.as_rotation_matrix(q_a)
        rmat_b = quaternion.as_rotation_matrix(q_b)

        delta_rmat = np.matmul(rmat_b, rmat_a.T)

        delta_rvec = np.empty((3, 1))
        cv2.Rodrigues(delta_rmat, delta_rvec)

        delta_tvec = tvec_b - np.matmul(delta_rmat, tvec_a.T).T

        y_delta_rvec.append(delta_rvec)
        y_delta_tvec.append(delta_tvec)


    x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    y_delta_rvec = np.reshape(y_delta_rvec, (len(y_delta_rvec), y_delta_rvec[0].shape[0]))
    y_delta_tvec = np.reshape(y_delta_tvec, (len(y_delta_tvec), y_delta_tvec[0].shape[0]))

    return x, [y_delta_rvec, y_delta_tvec], init_rvec, init_tvec


def load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    init_p = pos_data[window_size//2 - stride//2, :]
    init_q = ori_data[window_size//2 - stride//2, :]

    #x = []
    x_gyro = []
    x_acc = []
    y_delta_p = []
    y_delta_q = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size//2 - stride//2, :]
        p_b = pos_data[idx + window_size//2 + stride//2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))


    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    #return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q


def load_dataset_3d(gyro_data, acc_data, loc_data, window_size=200, stride=10):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    l0 = loc_data[window_size//2 - stride//2 - stride, :]
    l1 = loc_data[window_size//2 - stride//2, :]
    init_l = l1
    delta_l, init_theta, init_psi = cartesian_to_spherical_coordinates(l1 - l0)

    #x = []
    x_gyro = []
    x_acc = []
    y_delta_l = []
    y_delta_theta = []
    y_delta_psi = []

    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        delta_l0, theta0, psi0 = cartesian_to_spherical_coordinates(loc_data[idx + window_size//2 - stride//2, :] - loc_data[idx + window_size//2 - stride//2 - stride, :])

        l0 = loc_data[idx + window_size//2 - stride//2, :]
        l1 = loc_data[idx + window_size//2 + stride//2, :]

        delta_l, theta1, psi1 = cartesian_to_spherical_coordinates(l1 - l0)

        delta_theta = theta1 - theta0
        delta_psi = psi1 - psi0

        if delta_theta < -np.pi:
            delta_theta += 2 * np.pi
        elif delta_theta > np.pi:
            delta_theta -= 2 * np.pi

        if delta_psi < -np.pi:
            delta_psi += 2 * np.pi
        elif delta_psi > np.pi:
            delta_psi -= 2 * np.pi

        y_delta_l.append(np.array([delta_l]))
        y_delta_theta.append(np.array([delta_theta]))
        y_delta_psi.append(np.array([delta_psi]))


    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_delta_l = np.reshape(y_delta_l, (len(y_delta_l), y_delta_l[0].shape[0]))
    y_delta_theta = np.reshape(y_delta_theta, (len(y_delta_theta), y_delta_theta[0].shape[0]))
    y_delta_psi = np.reshape(y_delta_psi, (len(y_delta_psi), y_delta_psi[0].shape[0]))

    #return x, [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi
    return [x_gyro, x_acc], [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi


def load_dataset_2d(imu_data_filename, gt_data_filename, window_size=200, stride=10):

    #imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    #gt_data = np.genfromtxt(gt_data_filename, delimiter=',')
    
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    #imu_data = imu_data[1200:-300]
    #gt_data = gt_data[1200:-300]
    
    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
    loc_data = gt_data[:, 2:4]

    #l0 = loc_data[0, :]
    #l1 = loc_data[window_size, :]

    #l0 = loc_data[window_size - stride - stride, :]
    #l1 = loc_data[window_size - stride, :]

    l0 = loc_data[window_size//2 - stride//2 - stride, :]
    l1 = loc_data[window_size//2 - stride//2, :]
    
    l_diff = l1 - l0
    psi0 = np.arctan2(l_diff[1], l_diff[0])
    init_l = l1
    init_psi = psi0

    x = []
    y_delta_l = []
    y_delta_psi = []

    #for idx in range(stride, gyro_acc_data.shape[0] - window_size - 1, stride):
    #for idx in range(window_size, gyro_acc_data.shape[0] - window_size - 1, stride):
    for idx in range(0, gyro_acc_data.shape[0] - window_size - 1, stride):
        x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])

        #l0_diff = loc_data[idx, :] - loc_data[idx - stride, :]
        #l0_diff = loc_data[idx, :] - loc_data[idx - window_size, :]
        #l0_diff = loc_data[idx + window_size - stride, :] - loc_data[idx + window_size - stride - stride, :]
        l0_diff = loc_data[idx + window_size//2 - stride//2, :] - loc_data[idx + window_size//2 - stride//2 - stride, :]
        psi0 = np.arctan2(l0_diff[1], l0_diff[0])

        #l0 = loc_data[idx, :]
        #l0 = loc_data[idx + window_size - stride, :]
        #l1 = loc_data[idx + window_size, :]

        #l0 = loc_data[idx, :]
        #l1 = loc_data[idx + stride, :]

        l0 = loc_data[idx + window_size//2 - stride//2, :]
        l1 = loc_data[idx + window_size//2 + stride//2, :]

        l_diff = l1 - l0
        psi1 = np.arctan2(l_diff[1], l_diff[0])
        delta_l = np.linalg.norm(l_diff)
        delta_psi = psi1 - psi0

        #psi0 = psi1

        if delta_psi < -np.pi:
            delta_psi += 2 * np.pi
        elif delta_psi > np.pi:
            delta_psi -= 2 * np.pi

        y_delta_l.append(np.array([delta_l]))
        y_delta_psi.append(np.array([delta_psi]))

        #y_delta_l.append(np.array([delta_l / (window_size / 100)]))
        #y_delta_psi.append(np.array([delta_psi / (window_size / 100)]))


    x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    y_delta_l = np.reshape(y_delta_l, (len(y_delta_l), y_delta_l[0].shape[0]))
    y_delta_psi = np.reshape(y_delta_psi, (len(y_delta_psi), y_delta_psi[0].shape[0]))

    return x, [y_delta_l, y_delta_psi], init_l, init_psi        

def load_dataset_IMUData(dataset_path, file_names):
    ImuDataR, ImuDataL, gtRmL, gtRmR, gtQL, gtQR = np.empty((0,17)), np.empty((0,17)), np.empty((0,8)), np.empty((0,8)), np.empty((0,4)), np.empty((0,4))
    # Reading file by file in the folder: Changing to np.array
    os.chdir(dataset_path)
    i = 0
    for file in glob.glob(file_names):
        i = i+1
        ImuDataRtmp, ImuDataLtmp, gtRmLtmp, gtRmRtmp, gtQLtmp, gtQRtmp = getCSVData(dataset_path + file)

        ImuDataR = np.append(ImuDataR, ImuDataRtmp.values, axis=0)
        ImuDataL = np.append(ImuDataL, ImuDataLtmp.values, axis=0)

        gtRmR = np.append(gtRmR, gtRmRtmp.values, axis=0)
        gtRmL = np.append(gtRmL, gtRmLtmp.values, axis=0)

        gtQR = np.append(gtQR, gtQRtmp.values, axis=0)
        gtQL = np.append(gtQL, gtQLtmp.values, axis=0)
    # print(i)

    #IMUData
    x_accR, x_accL = ImuDataR[:,2:5], ImuDataL[:,2:5]
    x_gyroR, x_gyroL = ImuDataR[:,5:8], ImuDataL[:,5:8]
    x_magR, x_magL = ImuDataR[:,8:11], ImuDataL[:,8:11]

    #Groundtruth
    y_poseR, y_poseL = gtRmR[:,5:8], gtRmL[:,5:8]
    y_orientationQR, y_orientationQL = gtQR[:,:], gtQL[:,:]
    y_orientationRmR, y_orientationRmL = gtRmR[:,2:5], gtRmL[:,2:5]

    return x_accR, x_accL, x_gyroR, x_gyroL, x_magR, x_magL, y_poseR, y_poseL, y_orientationRmR, y_orientationRmL, y_orientationQR, y_orientationQL

def getCSVData(filename):
  skipA = 5
  skipB = 6
  mycolumns1 = ["Frame", "Subframe", "AccX-L", "AccY-L", "AccZ-L", "GyroX-L", "GyroY-L", "GyroZ-L", "MagX-L", "MagY-L", "MagZ-L","GlobalAngX-L","GlobalAngY-L", "GlobalAngZ-L", "HighGX-L", "HighGY-L", "HighGZ-L", "AccX-R","AccY-R", "AccZ-R", "GyroX-R", "GyroY-R", "GyroZ-R", "MagX-R", "MagY-R", "MagZ-R","GlobalAngX-R","GlobalAngY-R", "GlobalAngZ-R", "HighGX-R", "HighGY-R", "HighGZ-R"]
  dftest = pd.read_csv(filename,skiprows=skipA, skip_blank_lines=False, names= mycolumns1, engine='python')
  segment_row_number = dftest[dftest['Frame'] == 'Frame'].index[0]+1

  #Reading the IMUData
  df = pd.read_csv(filename, skiprows= skipA, header=None, nrows=segment_row_number-skipA)
  df.columns = ["Frame", "Subframe", "AccX-L", "AccY-L", "AccZ-L", "GyroX-L", "GyroY-L", "GyroZ-L", "MagX-L", "MagY-L", "MagZ-L","GlobalAngX-L","GlobalAngY-L", "GlobalAngZ-L", "HighGX-L", "HighGY-L", "HighGZ-L", "AccX-R","AccY-R", "AccZ-R", "GyroX-R", "GyroY-R", "GyroZ-R", "MagX-R", "MagY-R", "MagZ-R","GlobalAngX-R","GlobalAngY-R", "GlobalAngZ-R", "HighGX-R", "HighGY-R", "HighGZ-R"]
  
  #put a filter on a subframe
  ImuDataL = df.query('Subframe == 0')[["Frame", "Subframe", "AccX-L", "AccY-L", "AccZ-L", "GyroX-L", "GyroY-L", "GyroZ-L", "MagX-L", "MagY-L", "MagZ-L","GlobalAngX-L","GlobalAngY-L", "GlobalAngZ-L", "HighGX-L", "HighGY-L", "HighGZ-L"]]
  ImuDataR = df.query('Subframe == 0')[["Frame", "Subframe", "AccX-R","AccY-R", "AccZ-R", "GyroX-R", "GyroY-R", "GyroZ-R", "MagX-R", "MagY-R", "MagZ-R","GlobalAngX-R","GlobalAngY-R", "GlobalAngZ-R", "HighGX-R", "HighGY-R", "HighGZ-R"]]
  
  #Reading Ground Truth(Rotation Matrix)
  marker = pd.read_csv(filename, skiprows = segment_row_number+skipB, header=None)
  marker.columns = ["Frame", "Subframe", "RX-L", "RY-L", "RZ-L", "TX-L", "TY-L", "TZ-L", "RX-R","RY-R", "RZ-R", "TX-R", "TY-R", "TZ-R"]
  gtRmL = marker[["Frame", "Subframe", "RX-L", "RY-L", "RZ-L", "TX-L", "TY-L", "TZ-L"]]
  gtRmR = marker[["Frame", "Subframe","RX-R","RY-R", "RZ-R", "TX-R", "TY-R", "TZ-R"]]
  
  #Reading Ground Truth(Quaternion)
  tempL = pd.DataFrame(marker.apply(lambda row : get_quat_from_rot_vec(marker['RX-L'].iloc[0],marker['RY-L'].iloc[0], marker['RZ-L'].iloc[0]), axis = 1))
  gtQL = pd.DataFrame(tempL.iloc[:,0].tolist(), columns=["AngvW", "Angvi", "Angvj", "Angvk"])
  tempR = pd.DataFrame(marker.apply(lambda row : get_quat_from_rot_vec(marker['RX-R'].iloc[0],marker['RY-R'].iloc[0], marker['RZ-R'].iloc[0]), axis = 1))
  gtQR = pd.DataFrame(tempR.iloc[:,0].tolist(), columns=["AngvW", "Angvi", "Angvj", "Angvk"])
  return ImuDataR, ImuDataL, gtRmL, gtRmR, gtQL, gtQR

