
'''
Usage:


'''
import numpy as np
import torch
from scipy import signal
from scipy.signal import resample
from scipy.signal import hilbert
import matplotlib.pyplot as plt



def filter_EEG(model_Sel, data_Sel, input_data, fs, win_sel, win_classify):
    if data_Sel in ['SEED']:
        win_len = win_classify['win_len']
        win_step = win_len - win_classify['win_overlap']
    else:
        win_start = int(win_classify[0] * fs - win_sel[0] * fs)
        win_end = int(win_classify[1] * fs - win_sel[0] * fs)
        win = np.arange(win_start, win_end, 1)

    output_data = {'fs': fs}
    X_train = input_data['X']
    label_train = input_data['Y']



    if model_Sel in ['EEGNet', 'EEGConformer', 'LDMANet', 'TSception', 'ShallowNet', 'MSVTNet',  'TransNet', 'EDPNet',
                     'IFNetV2', 'MSVTNet',
                     'MSSTNet']:
        
        if data_Sel in ['BNCI2014_001', 'BCI4_2b', 'Lee2019_MI', 'PhysionetMI', 'WBCIC_MI_3C','WBCIC_MI_2C']:
            b, a = signal.butter(4, [4 / (fs / 2), 40 / (fs / 2)], 'bandpass') # motor imagery 数据集
            X_train_filtered = signal.filtfilt(b, a, X_train, axis=2)
            X_train_filtered = X_train_filtered[:, np.newaxis, :, :]
            output_data['X'] = torch.from_numpy(X_train_filtered[:, :, :, win]).float()
            output_data['Y'] = torch.from_numpy(input_data['Y']).to(torch.long)


    if model_Sel in ['FBCNet', 'TSFCNet', 'LMDANet', 'LightConvNet']:
        if data_Sel in ['BNCI2014_001', 'BCI4_2b', 'Lee2019_MI', 'PhysionetMI', 'WBCIC_MI_3C', 'WBCIC_MI_2C']:
            cutoff_range = np.arange(4, 40, 4)
            X_train_filtered = []
            for frei in range(len(cutoff_range)):
                b, a = signal.butter(5, [cutoff_range[frei] / (fs / 2), (cutoff_range[frei] + 4) / (fs / 2)], 'bandpass')
                X_train_filtered.append(signal.filtfilt(b, a, X_train, axis=2))

            X_train_filtered = np.array(X_train_filtered)
            X_train_filtered = X_train_filtered.transpose((1, 0, 2, 3))
            output_data['X'] = torch.from_numpy(X_train_filtered[:, :, :, win]).float()
            output_data['Y'] = torch.from_numpy(input_data['Y']).to(torch.long)

    if model_Sel in ['IFNet', 'IFNet_SimSiam']:
        if data_Sel in ['BNCI2014_001', 'BNCI2014_002', 'BNCI2014_004', 'BNCI2015_001', 'BCI4_2b',
                        'BNCI2015_004', 'Cho2017', 'Lee2019_MI', 'PhysionetMI', 'WBCIC_MI_3C', 'WBCIC_MI_2C']:
            X_train_filtered = []
            b, a = signal.butter(5, [4 / (fs / 2), 16 / (fs / 2)], 'bandpass')
            X_train_filtered.append(signal.filtfilt(b, a, X_train, axis=2))
            b, a = signal.butter(5, [16 / (fs / 2), 40 / (fs / 2)], 'bandpass')
            X_train_filtered.append(signal.filtfilt(b, a, X_train, axis=2))

            X_train_filtered = np.array(X_train_filtered)
            X_train_filtered = X_train_filtered.transpose((1, 0, 2, 3))   # batch, frequency, channel, time
            output_data['X'] = torch.from_numpy(X_train_filtered[:, :, :, win]).float()
            output_data['Y'] = torch.from_numpy(input_data['Y']).to(torch.long)

    return output_data


if __name__ == "__main__":
    run_code = 0
