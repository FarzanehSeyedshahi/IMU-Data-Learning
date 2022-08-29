import argparse
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

from sklearn.metrics import r2_score

from dataset import *
from util import *
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc', 'synthetic', 'synthetic9d', 'oxiod9d'], help='Training dataset name (\'oxiod\' or \'euroc\')')
    parser.add_argument('model', help='Model path')
    args = parser.parse_args()

    model = load_model(args.model)

    window_size = 200
    stride = 10

    if args.dataset == 'synthetic':
        gyro_data, acc_data, pos_data, ori_data = load_synthetic_dataset("dataset/evaluation")
        [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
        
        [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro[0:200, :, :], x_acc[0:200, :, :]], batch_size=1, verbose=0)
        

        gt_trajectory, gt_quaternion = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
        pred_trajectory, pred_quaternion = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

        # for i in range(0, len(pred_trajectory-1), 200):
        pred_trajectory = pred_trajectory[0:200, :]
        gt_trajectory = gt_trajectory[0:200, :]

        trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))

        print('Trajectory RMSE: %f' % (trajectory_rmse))

    else:
        imu_data_filenames = []
        gt_data_filenames = []

        dataset_path = '/home/work/fshahi/Oxford Inertial Odometry Dataset/handheld/'

        if args.dataset in ['oxiod', 'oxiod9d']:
            imu_data_filenames.append(dataset_path + 'data1/syn/imu2.csv')
            imu_data_filenames.append(dataset_path + 'data1/syn/imu5.csv')
            imu_data_filenames.append(dataset_path + 'data1/syn/imu6.csv')
            imu_data_filenames.append(dataset_path + 'data3/syn/imu1.csv')
            imu_data_filenames.append(dataset_path + 'data4/syn/imu1.csv')
            imu_data_filenames.append(dataset_path + 'data4/syn/imu3.csv')
            imu_data_filenames.append(dataset_path + 'data5/syn/imu1.csv')

            gt_data_filenames.append(dataset_path + 'data1/syn/vi2.csv')
            gt_data_filenames.append(dataset_path + 'data1/syn/vi5.csv')
            gt_data_filenames.append(dataset_path + 'data1/syn/vi6.csv')
            gt_data_filenames.append(dataset_path + 'data3/syn/vi1.csv')
            gt_data_filenames.append(dataset_path + 'data4/syn/vi1.csv')
            gt_data_filenames.append(dataset_path + 'data4/syn/vi3.csv')
            gt_data_filenames.append(dataset_path + 'data5/syn/vi1.csv')

        elif args.dataset == 'euroc':
            imu_data_filenames.append('MH_02_easy/mav0/imu0/data.csv')
            imu_data_filenames.append('MH_04_difficult/mav0/imu0/data.csv')
            imu_data_filenames.append('V1_03_difficult/mav0/imu0/data.csv')
            imu_data_filenames.append('V2_02_medium/mav0/imu0/data.csv')
            imu_data_filenames.append('V1_01_easy/mav0/imu0/data.csv')

            gt_data_filenames.append('MH_02_easy/mav0/state_groundtruth_estimate0/data.csv')
            gt_data_filenames.append('MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv')
            gt_data_filenames.append('V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv')
            gt_data_filenames.append('V2_02_medium/mav0/state_groundtruth_estimate0/data.csv')
            gt_data_filenames.append('V1_01_easy/mav0/state_groundtruth_estimate0/data.csv')

        for (cur_imu_data_filename, cur_gt_data_filename) in zip(imu_data_filenames, gt_data_filenames):
            if args.dataset == 'oxiod':
                gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
                [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
                [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro[0:200, :, :], x_acc[0:200, :, :]], batch_size=1, verbose=0)

            
            elif args.dataset == 'euroc':
                gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)
                [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
                [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=0)

            
            elif args.dataset == 'oxiod9d':
                gyro_data, acc_data, mag_data, pos_data, ori_data = load_oxiod_9D_dataset(cur_imu_data_filename, cur_gt_data_filename)
                [x_gyro, x_acc, x_mag], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_9d_quat( gyro_data, acc_data, mag_data, pos_data, ori_data, window_size,stride)
                [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc, x_mag], batch_size=1, verbose=0)




            gt_trajectory, gt_quaternion = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
            pred_trajectory, prerd_quaternion = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

            if args.dataset in ['oxiod', 'oxiod9d']:
                pred_trajectory = pred_trajectory[0:200, :]
                gt_trajectory = gt_trajectory[0:200, :]

            trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))

            
            print('Trajectory RMSE (%s), sequence %s: %f' % (args.model[:-5], cur_imu_data_filename[-8:], trajectory_rmse))
            # print(pred_trajectory.shape)
            # from sklearn.ensemble import RandomForestRegressor
            # clf = RandomForestRegressor(n_estimators=10)
            print("Accuracy Score Poz_X: ",r2_score(pred_trajectory[:,0], gt_trajectory[:,0]))
            print("Accuracy Score Poz_Y: ",r2_score(pred_trajectory[:,1], gt_trajectory[:,1]))
            print("Accuracy Score Poz_Z: ",r2_score(pred_trajectory[:,2], gt_trajectory[:,2]))




if __name__ == '__main__':
    main()
