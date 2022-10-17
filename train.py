import numpy as np
import matplotlib.pyplot as plt
import argparse

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from tensorflow.keras.optimizers import Adam

from sklearn.utils import shuffle

from time import time

from dataset import *
from model import *
from util import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc'], help='Training dataset name (\'oxiod\' or \'euroc\')')
    parser.add_argument('output', help='Model output name')
    args = parser.parse_args()

    np.random.seed(0)

    window_size_transformer = 200
    window_size = window_size_transformer
    stride = 10

    x_gyro = []
    x_acc = []
    x_mag = []

    y_delta_p = []
    y_delta_q = []

    imu_data_filenames = []
    gt_data_filenames = []

    if args.dataset == 'oxiod':
        dataset_path = \
        '/home/work/fshahi/Oxford Inertial Odometry Dataset/handheld/'
        imu_data_filenames.append(dataset_path
                + 'data5/syn/imu3.csv')
        imu_data_filenames.append(dataset_path
                + 'data2/syn/imu1.csv')
        imu_data_filenames.append(dataset_path
                + 'data2/syn/imu2.csv')
        imu_data_filenames.append(dataset_path
                + 'data5/syn/imu2.csv')
        imu_data_filenames.append(dataset_path
                + 'data3/syn/imu4.csv')
        imu_data_filenames.append(dataset_path
                + 'data4/syn/imu4.csv')
        imu_data_filenames.append(dataset_path
                + 'data4/syn/imu2.csv')
        imu_data_filenames.append(dataset_path
                + 'data1/syn/imu7.csv')
        imu_data_filenames.append(dataset_path
                + 'data5/syn/imu4.csv')
        imu_data_filenames.append(dataset_path
                + 'data4/syn/imu5.csv')
        imu_data_filenames.append(dataset_path
                + 'data1/syn/imu3.csv')
        imu_data_filenames.append(dataset_path
                + 'data3/syn/imu2.csv')
        imu_data_filenames.append(dataset_path
                + 'data2/syn/imu3.csv')
        imu_data_filenames.append(dataset_path
                + 'data1/syn/imu1.csv')
        imu_data_filenames.append(dataset_path
                + 'data3/syn/imu3.csv')
        imu_data_filenames.append(dataset_path
                + 'data3/syn/imu5.csv')
        imu_data_filenames.append(dataset_path
                + 'data1/syn/imu4.csv')

        gt_data_filenames.append(dataset_path
                + 'data5/syn/vi3.csv')
        gt_data_filenames.append(dataset_path
                + 'data2/syn/vi1.csv')
        gt_data_filenames.append(dataset_path
                + 'data2/syn/vi2.csv')
        gt_data_filenames.append(dataset_path
                + 'data5/syn/vi2.csv')
        gt_data_filenames.append(dataset_path
                + 'data3/syn/vi4.csv')
        gt_data_filenames.append(dataset_path
                + 'data4/syn/vi4.csv')
        gt_data_filenames.append(dataset_path
                + 'data4/syn/vi2.csv')
        gt_data_filenames.append(dataset_path
                + 'data1/syn/vi7.csv')
        gt_data_filenames.append(dataset_path
                + 'data5/syn/vi4.csv')
        gt_data_filenames.append(dataset_path
                + 'data4/syn/vi5.csv')
        gt_data_filenames.append(dataset_path
                + 'data1/syn/vi3.csv')
        gt_data_filenames.append(dataset_path
                + 'data3/syn/vi2.csv')
        gt_data_filenames.append(dataset_path
                + 'data2/syn/vi3.csv')
        gt_data_filenames.append(dataset_path
                + 'data1/syn/vi1.csv')
        gt_data_filenames.append(dataset_path
                + 'data3/syn/vi3.csv')
        gt_data_filenames.append(dataset_path
                + 'data3/syn/vi5.csv')
        gt_data_filenames.append(dataset_path
                + 'data1/syn/vi4.csv')
    
    elif args.dataset == 'euroc':
        imu_data_filenames.append('MH_01_easy/mav0/imu0/data.csv')
        imu_data_filenames.append('MH_03_medium/mav0/imu0/data.csv')
        imu_data_filenames.append('MH_05_difficult/mav0/imu0/data.csv')
        imu_data_filenames.append('V1_02_medium/mav0/imu0/data.csv')
        imu_data_filenames.append('V2_01_easy/mav0/imu0/data.csv')
        imu_data_filenames.append('V2_03_difficult/mav0/imu0/data.csv')

        gt_data_filenames.append('MH_01_easy/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('MH_03_medium/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V1_02_medium/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V2_01_easy/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv')

    for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):
        if args.dataset == 'oxiod':
        #     cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
            # 9D input
              cur_gyro_data, cur_acc_data, cur_mag_data, cur_pos_data, cur_ori_data = load_oxiod_9D_dataset(cur_imu_data_filename, cur_gt_data_filename)




        elif args.dataset == 'euroc':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)

        # [cur_x_gyro, cur_x_acc], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)
#       9D input
        [cur_x_gyro, cur_x_acc, cur_x_mag], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_9d_quat(cur_gyro_data, cur_acc_data, cur_mag_data, cur_pos_data, cur_ori_data, window_size, stride)

        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)
        #       9D input
        x_mag.append(cur_x_mag)


        y_delta_p.append(cur_y_delta_p)
        y_delta_q.append(cur_y_delta_q)
        

    x_gyro = np.vstack(x_gyro)
    x_acc = np.vstack(x_acc)
    #       9D input
    x_mag = np.vstack(x_mag)


    y_delta_p = np.vstack(y_delta_p)
    y_delta_q = np.vstack(y_delta_q)

#     x_gyro, x_acc, y_delta_p, y_delta_q = shuffle(x_gyro, x_acc, y_delta_p, y_delta_q)
#       9D input
    x_gyro, x_acc, x_mag, y_delta_p, y_delta_q = shuffle(x_gyro, x_acc, x_mag, y_delta_p, y_delta_q)


#     input_scales = np.arange(2,22, dtype='float32')
#     x_acc_cwt = create_cwt_images(x_acc, input_scales, wavelet_name = "morl", rescale=False, upsample=False)
#     x_gyro_cwt = create_cwt_images(x_gyro, input_scales, wavelet_name = "morl", rescale=False, upsample=False)
#     x_mag_cwt = create_cwt_images(x_mag, input_scales, wavelet_name = "morl", rescale=False, upsample=False)

##original
#     pred_model = create_pred_model_6d_quat(window_size)
#     train_model = create_train_model_6d_quat(pred_model, window_size)

## resnet 6D
#     pred_model = create_pred_resnet_model_6d_quat(window_size)
#     train_model = create_train_model_6d_quat(pred_model, window_size)

## resnet 9D
#     pred_model = create_resnet_pred_model_9d_quat(window_size)
#     train_model = create_train_resnet_or_without_model_9d_quat(pred_model, window_size)


## resnet cwt
#     sample_size = 20
#     pred_model = create_pred_resnet_cwt(sample_size, window_size)
#     train_model = create_train_resnet_cwt(pred_model, sample_size, window_size)
##################################################################

# VIT
#     pred_model = create_vit_classifier((200, 9,1), (200, 9), (20,3), 64, 8)
#     train_model = create_VIT_model_9d(pred_model, window_size= window_size_transformer)

# Combined VIT
    pred_model = create_combined_transformer_pred_model_9d(window_size_transformer)
    train_model = create_train_resnet_or_without_model_9d_quat(pred_model, window_size)


##################################################################


## Transformer network 9D
#     pred_model = create_transformer_pred_model_9d(
#     head_size=256,
#     window_size= window_size_transformer,
#     num_heads=4,
#     ff_dim=4,
#     num_transformer_blocks=12,
#     mlp_units=[128],
#     dropout=0.25,
#     mlp_dropout=0.25)
    pred_model.summary()
#     train_model = create_train_resnet_or_without_model_9d_quat(pred_model, window_size= window_size_transformer)


    ##### Adding decay learning rate ########
    # define SGD optimizer
    epochs = 500

    train_model.compile(optimizer=Adam(learning_rate=0.0001, decay=0.0001/epochs), loss=None)

    model_checkpoint = ModelCheckpoint('{}_checkpoint.hdf5'.format(args.output), monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir="logs/{}_{}".format(args.output, time()))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # 9D input
#     history = train_model.fit([x_gyro, x_acc, x_mag, y_delta_p, y_delta_q], epochs=1, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard, early_stopping], validation_split=0.1)
#     train_model = load_model('{}_checkpoint.hdf5'.format(args.output), custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer}, compile=False)


    # cwt input
#     history = train_model.fit([x_gyro_cwt, x_acc_cwt, x_mag_cwt, y_delta_p, y_delta_q], epochs=epochs, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard, early_stopping], validation_split=0.1)
#     train_model = load_model('{}_checkpoint.hdf5'.format(args.output), custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer}, compile=False)

#     VIT transformer
#     history = train_model.fit([np.c_[x_gyro, x_acc, x_mag], y_delta_p, y_delta_q], epochs=epochs, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard, early_stopping], validation_split=0.1)
#     train_model = load_model('{}_checkpoint.hdf5'.format(args.output), custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer, 'Patches':Patches, 'PatchEncoder':PatchEncoder}, compile=False)


#     CNV-VIT transformer
    history = train_model.fit([x_gyro, x_acc, x_mag, y_delta_p, y_delta_q], epochs=epochs, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard, early_stopping], validation_split=0.1)
    train_model = load_model('{}_checkpoint.hdf5'.format(args.output), custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer, 'Patches':Patches, 'PatchEncoder':PatchEncoder}, compile=False)


## original
#     pred_model = create_pred_model_6d_quat(window_size)

## resnet 9D
#     pred_model = create_resnet_pred_model_9d_quat(window_size)

## resnet cwt
#     pred_model = create_pred_resnet_cwt(sample_size, window_size)

## resnet 6D
#     pred_model = create_pred_resnet_model_6d_quat(window_size)
    
# Transformer9D
#     pred_model = create_transformer_pred_model_9d(
#     head_size=256,
#     window_size= window_size_transformer,
#     num_heads=4,
#     ff_dim=4,
#     num_transformer_blocks=12,
#     mlp_units=[128],
#     dropout=0.25,
#     mlp_dropout=0.25)

# vit 9D
#     pred_model = create_vit_classifier()

# Combined VIT
    pred_model = create_combined_transformer_pred_model_9d(window_size_transformer)

    pred_model.set_weights(train_model.get_weights()[:-2])
    pred_model.save('%s.hdf5' % args.output)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig("loss_results/loss_per_epoch_{}_{}_{}.png".format(args.output, time(), args.dataset))

if __name__ == '__main__':
    main()
