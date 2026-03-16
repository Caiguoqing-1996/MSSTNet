
import numpy as np
from scipy.io import loadmat
import mne
import os
from scipy.signal import resample
import re

#############################################################################
########################  Motor Imagery   ###################################
#############################################################################
def load_BNCI2014_001(file_path, subject, win_sel):
    '''
    This four class motor imagery data set was originally released as data set 2a of the BCI Competition IV.
    :param file_path:
    :param subject:
    :param win_sel:
    :return:
    '''
    data_subject = {}
    stimcodes = ('769', '770', '771', '772')  # 注意读数据集说明，‘768’才是一个trial的开始，也就意味着对应0时刻
    channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
    tmin = win_sel[0]
    tmax = win_sel[1]

    file_to_load = file_path + '\A0' + str(subject) + 'T.gdf'
    raw_data = mne.io.read_raw_gdf(file_to_load)
    fs = raw_data.info.get('sfreq')
    events, event_ids = mne.events_from_annotations(raw_data)
    stims = [value for key, value in event_ids.items() if key in stimcodes]
    epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                        baseline=None, preload=True, proj=False, reject_by_annotation=False)
    epochs = epochs.drop_channels(channels_to_remove)
    class_return = epochs.events[:, -1] - min(epochs.events[:, -1])
    data_return = epochs.get_data() * 1e6
    data_return = data_return - np.mean(data_return, axis=1, keepdims=True)


    # channel_index = np.array([2,4,6,8,10,12,14,16,18]) - 1
    channel_index = np.arange(0, 22)

    data_subject['train'] = {'X': data_return[:, channel_index, :],
                             'Y': class_return}

    ################################# 测试集数据读取  ####################################################
    file_to_load = file_path + '\A0' + str(subject) + 'E.gdf'
    raw_data = mne.io.read_raw_gdf(file_to_load)
    fs = raw_data.info.get('sfreq')
    labels_mat = loadmat(file_path + '/Data sets 2a_true_labels/' + 'A0' + str(subject) + 'E.mat')
    class_all = labels_mat['classlabel'][:, 0]
    events, event_ids = mne.events_from_annotations(raw_data)
    index_type = [value for key, value in event_ids.items() if key in '783']
    events_index = np.where(events[:, 2] == np.array(index_type))[0]
    events = events[events_index, :]
    events[:, 2] = class_all
    stims = list(np.array(np.unique(class_all)))
    epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                        baseline=None, preload=True, proj=False, reject_by_annotation=False)
    epochs = epochs.drop_channels(channels_to_remove)
    class_return = epochs.events[:, -1] - min(epochs.events[:, -1])
    data_return = epochs.get_data() * 1e6
    data_return = data_return - np.mean(data_return, axis=1, keepdims=True)

    data_subject['test'] = {'X': data_return[:, channel_index, :],
                            'Y': class_return}
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]
    data_subject['fs'] = 250
    data_subject['win_sel'] = win_sel
    return data_subject


def load_data_BCI4_2b(data_path, subject, time_extract, training):


    stimcodes = ('769', '770')  # 注意读数据集说明，‘768’才是一个trial的开始，也就意味着对应0时刻
    channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
    tmin = time_extract[0]
    tmax = time_extract[1]


    if training:
        for sessioni in range(3):

            sessioni= 2

            file_to_load = data_path + '\B0' + str(subject) + '0{}T.gdf'.format(sessioni+1)
            raw_data = mne.io.read_raw_gdf(file_to_load)
            fs = raw_data.info.get('sfreq')
            events, event_ids = mne.events_from_annotations(raw_data)
            stims = [value for key, value in event_ids.items() if key in stimcodes]

            epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                                baseline=None, preload=True, proj=False, reject_by_annotation=False)
            epochs = epochs.drop_channels(channels_to_remove)
            if sessioni == 0:
                class_return = epochs.events[:, -1] - min(epochs.events[:, -1]) + 1
                data_return = epochs.get_data() * 1e6
            else:
                class_return_ada = epochs.events[:, -1] - min(epochs.events[:, -1]) + 1
                data_return_ada = epochs.get_data() * 1e6

                class_return = np.concatenate((class_return, class_return_ada), axis=0)
                data_return = np.concatenate((data_return, data_return_ada), axis=0)

    else:
        for sessioni in range(2):

            file_to_load = data_path + '\B0' + str(subject) + '0{}E.gdf'.format(sessioni+4)
            labels_mat = loadmat(data_path + '/Data sets 2b_true_labels/' + '\B0' + str(subject) + '0{}E.mat'.format(sessioni+4))

            raw_data = mne.io.read_raw_gdf(file_to_load)
            fs = raw_data.info.get('sfreq')
            events, event_ids = mne.events_from_annotations(raw_data)
            stims = [value for key, value in event_ids.items() if key in ['783']]

            epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                                baseline=None, preload=True, proj=False, reject_by_annotation=False)
            epochs = epochs.drop_channels(channels_to_remove)

            if sessioni == 0:
                class_all = labels_mat['classlabel'][:, 0]
                class_return = class_all - min(class_all) + 1
                data_return = epochs.get_data() * 1e6
            else:
                class_all = labels_mat['classlabel'][:, 0]
                class_return_ada = class_all - min(class_all) + 1
                data_return_ada = epochs.get_data() * 1e6

                class_return = np.concatenate((class_return, class_return_ada), axis=0)
                data_return = np.concatenate((data_return, data_return_ada), axis=0)



    class_return = class_return - np.min(class_return)
    data_return = data_return - np.mean(data_return, axis=1, keepdims=True)

    return data_return, class_return

def load_BCI4_2b(file_path, subject, win_sel):
    X_test, y_test = load_data_BCI4_2b(file_path, subject, win_sel, False)
    X_train, y_train = load_data_BCI4_2b(file_path, subject, win_sel, True)

    data_subject = {}
    data_subject['train'] = {'X': X_train,
                             'Y': y_train}
    data_subject['test'] = {'X': X_test,
                            'Y': y_test}
    data_subject['fs'] = 250
    data_subject['win_sel'] = win_sel

    return data_subject


def load_data_KU(data_path, subject, training):

    if training:
        if subject < 10:
            file_to_load = data_path + '\sess01_subj0' + str(subject) + '_EEG_MI.mat'
        else:
            file_to_load = data_path + '\sess01_subj' + str(subject) + '_EEG_MI.mat'


        raw_eeg_subject = loadmat(file_to_load)

        x_data1 = raw_eeg_subject['EEG_MI_train']['smt'][0, 0]
        labels1 = raw_eeg_subject['EEG_MI_train']['y_dec'][0, 0][0]
        ch_names = raw_eeg_subject['EEG_MI_train']['chan'][0, 0][0]
        ch_names_list = [str(x[0]) for x in ch_names]

        x_data2 = raw_eeg_subject['EEG_MI_test']['smt'][0, 0]
        labels2 = raw_eeg_subject['EEG_MI_test']['y_dec'][0, 0][0]
        x_data = np.concatenate((x_data1, x_data2), axis=1)
        labels = np.concatenate((labels1, labels2), axis=0)

        class_return = labels - np.min(labels)
        data_return = np.transpose(x_data, axes=[1, 2, 0])

    else:
        if subject < 10:
            file_to_load = data_path + '\sess02_subj0' + str(subject) + '_EEG_MI.mat'
        else:
            file_to_load = data_path + '\sess02_subj' + str(subject) + '_EEG_MI.mat'

        raw_eeg_subject = loadmat(file_to_load)

        x_data1 = raw_eeg_subject['EEG_MI_train']['smt'][0, 0]
        labels1 = raw_eeg_subject['EEG_MI_train']['y_dec'][0, 0][0]
        ch_names = raw_eeg_subject['EEG_MI_train']['chan'][0, 0][0]
        ch_names_list = [str(x[0]) for x in ch_names]

        x_data2 = raw_eeg_subject['EEG_MI_test']['smt'][0, 0]
        labels2 = raw_eeg_subject['EEG_MI_test']['y_dec'][0, 0][0]
        x_data = np.concatenate((x_data1, x_data2), axis=1)
        labels = np.concatenate((labels1, labels2), axis=0)

        class_return = labels - np.min(labels)
        data_return = np.transpose(x_data, axes=[1, 2, 0])
    return data_return, class_return


def load_Lee2019_MI(file_path, subject, win_sel):
    import pickle

    if subject < 10:
        file_to_load = file_path + '\subj0' + str(subject) + '_EEG_MI.pkl'
    else:
        file_to_load = file_path + '\subj' + str(subject) + '_EEG_MI.pkl'
    with open(file_to_load, 'rb') as f:
        data_subject = pickle.load(f)

    # X_train, y_train = load_data_KU(file_path, subject, True)
    # X_test, y_test = load_data_KU(file_path, subject, False)
    #
    # # channel_Sel = np.array([33, 34,
    # #                         13, 14, 15,
    # #                         39,  40, 41]) - 1
    # channel_Sel = np.array([4, 5, 6,
    #                         8, 33, 9, 10, 34, 11,
    #                         35, 13, 36, 14, 37, 15, 38,
    #                         18, 39, 19, 40, 20, 41, 21,
    #                         24, 42, 25, 43, 26]) - 1  # 28个通道
    #
    # from scipy.signal import resample
    # data_subject = {}
    # X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
    # X_test = X_test - np.mean(X_test, axis=1, keepdims=True)
    #
    # data_subject['train'] = {'X': X_train,
    #                          'Y': y_train}
    # data_subject['test'] = {'X': X_test,
    #                         'Y': y_test}
    # data_subject['train']['X'] = data_subject['train']['X'][:, channel_Sel, :]
    # data_subject['test']['X'] = data_subject['test']['X'][:, channel_Sel, :]
    # data_subject['train']['X'] = resample(data_subject['train']['X'], 1000, axis=-1)
    # data_subject['test']['X'] = resample(data_subject['test']['X'], 1000, axis=-1)
    # data_subject['fs'] = 250
    # data_subject['win_sel'] = win_sel

    return data_subject



def load_WBCIC_MI_2C(file_path, subject, win_sel):
    '''

    数据总通道是64，去除眼电等，还剩余59个通道，
    其中PZ作为参考电极，进行重参考，因此余下58个通道

    :param file_path:
    :param subject:
    :param win_sel:
    :return:
    '''


    data_subject = {}
    labels = []
    all_EEG = []
    for sessioni in range(1, 4):
        # 构建subject和session的文件夹名称
        if subject<11:
            mat_file = f"sub-{subject:03d}_ses-{sessioni:02d}_task-motorimagery_eeg.mat"
            mat_file_path = os.path.join(file_path, f"sub-{subject:03d}", f"ses-{sessioni:02d}", 'eeg', mat_file)
        else:
            mat_file = f"sub-{subject+1:03d}_ses-{sessioni:02d}_task-motorimagery_eeg.mat"
            mat_file_path = os.path.join(file_path, f"sub-{subject:03d}", f"ses-{sessioni:02d}", 'eeg', mat_file)

        mat_data = loadmat(mat_file_path)
        si_labels = mat_data['labels'] - np.min(np.unique(mat_data['labels']))
        labels.append(np.squeeze(si_labels, axis=0))
        all_EEG.append(np.transpose(mat_data['data'], axes=[2, 0, 1]))

    channel_name = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
                    'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                    'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
                    'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
                    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
                    'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                    'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2']


    data_subject['train'] = {'X': all_EEG,
                             'Y': labels}

    data_subject['channel'] = channel_name
    data_subject['fs'] = 250
    data_subject['win_sel'] = win_sel

    return data_subject


def load_WBCIC_MI_3C(file_path, subject, win_sel):
    '''

    数据总通道是64，去除眼电等，还剩余59个通道，
    其中PZ作为参考电极，进行重参考，因此余下58个通道

    :param file_path:
    :param subject:
    :param win_sel:
    :return:
    '''


    data_subject = {}
    labels = []
    all_EEG = []
    for sessioni in range(1, 4):
        # 构建subject和session的文件夹名称

        mat_file = f"sub-{subject:03d}_ses-{sessioni:02d}_task-motorimagery_eeg.mat"
        mat_file_path = os.path.join(file_path, f"sub-{subject:03d}", f"ses-{sessioni:02d}", 'eeg', mat_file)

        mat_data = loadmat(mat_file_path)
        si_labels = mat_data['labels'] - np.min(np.unique(mat_data['labels']))
        labels.append(np.squeeze(si_labels))


        all_EEG.append(np.transpose(np.array(mat_data['data'], dtype='float32'), axes=[2, 0, 1]))

    channel_name = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8',
                    'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                    'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
                    'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8',
                    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
                    'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                    'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2']


    data_subject['train'] = {'X': all_EEG,
                             'Y': labels}

    data_subject['channel'] = channel_name
    data_subject['fs'] = 250
    data_subject['win_sel'] = win_sel

    return data_subject


def load_InHouse(file_path, subject, win_sel):
    data_subject = {}
    file_to_load = file_path + '/' + subject + '.mat'
    raw_eeg_subject = loadmat(file_to_load)
    channel_used_index = np.array([7, 8, 9, 10, 11, 12, 13,
                                   16, 17, 18, 19, 20, 21, 22,
                                   25, 26, 27, 28, 29, 30, 31,
                                   34, 35, 36, 37, 38, 39, 40,
                                   43, 44, 45, 46, 47, 48, 49]) - 1 # 30个通道
    fs = 100
    writeSamples = [int(win_sel[0] * fs), int(win_sel[1] * fs)]
    all_EEG = raw_eeg_subject['cnt'][channel_used_index, :]
    pos = raw_eeg_subject['pos'].squeeze()
    labels = raw_eeg_subject['mrk'].squeeze()


    EEG_trials = []
    for triali in range(labels.shape[0]):
        writeTime1 = int(pos[triali] + writeSamples[0])
        writeTime2 = int(pos[triali] + writeSamples[1])
        EEG_trials.append(all_EEG[:, writeTime1:writeTime2])  # append后，张量的格式为 trails X channels X time
    x_data = np.array(EEG_trials)
    label_data = labels - np.min(np.unique(labels))
    x_data = x_data - np.mean(x_data, axis=1, keepdims=True)
    data_subject['train'] = {'X': x_data,
                             'Y': label_data}
    data_subject['fs'] = fs
    data_subject['win_sel'] = win_sel



    return data_subject

#############################################################################
########################       Emotion        ###############################
#############################################################################

def load_SEED(file_path, subject, win_sel):
    data_subject = {}
    sub_file_name_all = {'1': ['1_20131027', '1_20131030', '1_20131107'],
                         '2': ['2_20140404', '2_20140413', '2_20140419'],
                         '3': ['3_20140603.mat', '3_20140611.mat', '3_20140629.mat'],
                         '4': ['4_20140705.mat', '4_20140621.mat', '4_20140702.mat'],
                         '5': ['5_20140418.mat', '5_20140506.mat', '5_20140411.mat'],
                         '6': ['6_20130712.mat', '6_20131113.mat', '6_20131016.mat'],
                         '7': ['7_20131030.mat', '7_20131027.mat', '7_20131106.mat'],
                         '8': ['8_20140511.mat', '8_20140514.mat', '8_20140521.mat'],
                         '9': ['9_20140627.mat', '9_20140704.mat', '9_20140620.mat'],
                         '10': ['10_20131211.mat', '10_20131204.mat', '10_20131130.mat'],
                         '11': ['11_20140618.mat', '11_20140625.mat', '11_20140630.mat'],
                         '12': ['12_20131127.mat', '12_20131207.mat', '12_20131201.mat'],
                         '13': ['13_20140527.mat', '13_20140610.mat', '13_20140603.mat'],
                         '14': ['14_20140627.mat', '14_20140601.mat', '14_20140615.mat'],
                         '15': ['15_20131105.mat', '15_20130709.mat', '15_20131016.mat'], }

    label_file = file_path + '\label.mat'
    labels_a_session = np.array(loadmat(label_file)['label']).squeeze()

    channel_name = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
                    'CB1', 'O1', 'OZ', 'O2', 'CB2']


    labels = []
    all_EEG = []
    for sub_filei in sub_file_name_all[str(subject)]:
        file_to_load = file_path + '\\' + sub_filei
        eeg_subject = loadmat(file_to_load)
        EEG_a_session = []
        for key, value in eeg_subject.items():
            if "eeg" in key:
                EEG_a_session.append(value)
            else:
                print("'eeg' is not in the string.")

        all_EEG.append(EEG_a_session)
        labels.append(labels_a_session - min(labels_a_session))
        assert len(EEG_a_session) == len(labels_a_session)


    data_subject['train'] = {'X': all_EEG,
                             'Y': labels}

    data_subject['channel'] = channel_name
    data_subject['fs'] = 200
    data_subject['win_sel'] = win_sel

    return data_subject



#############################################################################
########################       癫痫数据        ###############################
#############################################################################
# =========================================
# 工具函数
# =========================================
def time_to_seconds(tstr):
    h, m, s = map(int, tstr.split(":"))
    return h * 3600 + m * 60 + s


def parse_summary(path):
    records = []
    current = None

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith("File Name:"):
            if current is not None:
                records.append(current)
            fname = line.split("File Name:")[1].strip()
            current = {
                "fname": fname,
                "start_time": None,
                "end_time": None,
                "n_seizures": 0,
                "seizures": []
            }

        elif line.startswith("File Start Time:"):
            current["start_time"] = line.split("File Start Time:")[1].strip()

        elif line.startswith("File End Time:"):
            current["end_time"] = line.split("File End Time:")[1].strip()

        elif line.startswith("Number of Seizures in File:"):
            current["n_seizures"] = int(line.split(":")[1].strip())

        elif "Start Time" in line and "Seizure" in line:
            s = int(re.findall(r"Start Time:\s*(\d+)", line)[0])
            current["pending_start"] = s

        elif "End Time" in line and "Seizure" in line:
            e = int(re.findall(r"End Time:\s*(\d+)", line)[0])
            if "pending_start" in current:
                current["seizures"].append((current["pending_start"], e))
                del current["pending_start"]

    if current is not None:
        records.append(current)

    return records


def load_edf_channels(path, EDF_CHANNELS_18):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    edf_chs = raw.ch_names

    use_chs = [c for c in EDF_CHANNELS_18 if c in edf_chs]
    raw.pick_channels(use_chs)
    raw.load_data()
    return raw, use_chs


def extract_segment(raw, t0, t1, FS):
    start = int(t0 * FS)
    end   = int(t1 * FS)
    return raw.get_data(start=start, stop=end)



def cut_windows(eeg, FS, win_sec=10):
    win_len = win_sec * FS
    C, T = eeg.shape
    nw = T // win_len
    return np.stack([eeg[:, i*win_len:(i+1)*win_len] for i in range(nw)])


def load_chbmit(file_path, subject,  win_sel):
    # =========================================
    # 超参数（你可以随时调）
    # =========================================
    PRE_ICTAL_WINDOW = 3600  # 1 小时
    PRE_ICTAL_MIN = 10*60  # pre-ictal 至少 10 分钟
    INTER_ICTAL_GAP = 7200  # inter-ictal 必须离所有癫痫发作超过 2 小时
    WIN_SEC = 10  # 窗长（秒）
    FS = 256  # CHB-MIT 采样率

    # =========================================
    # 需要的 18 通道
    # =========================================
    EDF_CHANNELS_18 = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ]

    subject = int(subject)
    sub_name = f"chb{subject:02d}"
    sub_dir = os.path.join(file_path, sub_name)

    summary_path = os.path.join(sub_dir, f"{sub_name}-summary.txt")
    edf_infos = parse_summary(summary_path)

    # -------------------------
    # Step1：计算 EDF 绝对时间
    # -------------------------
    all_sz = []
    day_cnt = 0
    last_clock = None

    for info in edf_infos:
        st_clock = time_to_seconds(info["start_time"])

        if last_clock is not None and st_clock < last_clock:
            day_cnt += 1

        abs_start = st_clock + day_cnt * 86400
        info["abs_start"] = abs_start

        info["abs_seizures"] = []
        for s, e in info["seizures"]:
            abs_s = abs_start + s
            abs_e = abs_start + e
            info["abs_seizures"].append((abs_s, abs_e))
            all_sz.append((abs_s, abs_e))

        last_clock = st_clock

    all_sz.sort()

    # -------------------------
    # Step2：筛选 inter-ictal EDF
    # -------------------------
    valid_inter = []
    for info in edf_infos:
        if info["n_seizures"] != 0:
            continue

        st = info["abs_start"]
        et_clock = time_to_seconds(info["end_time"])
        st_clock = time_to_seconds(info["start_time"])

        dur = et_clock - st_clock
        if dur < 0:
            dur += 86400
        abs_end = st + dur

        whole_ok = True
        for (sz_s, sz_e) in all_sz:
            if abs(st - sz_s) < INTER_ICTAL_GAP or abs(abs_end - sz_s) < INTER_ICTAL_GAP:
                whole_ok = False
                break
            if st < sz_s < abs_end:
                whole_ok = False
                break

        if whole_ok:
            valid_inter.append((info, 0, dur))  # 整段有效

    # -------------------------
    # Step3：筛选 pre-ictal EDF
    # -------------------------
    valid_pre = []

    for info in edf_infos:
        if info["n_seizures"] == 0:
            continue

        for (abs_s, abs_e), (s, e) in zip(info["abs_seizures"], info["seizures"]):
            pre_start = max(info["abs_start"], abs_s - PRE_ICTAL_WINDOW)
            pre_end = abs_s
            pre_len = pre_end - pre_start

            if pre_len >= PRE_ICTAL_MIN:
                start_rel = pre_start - info["abs_start"]
                valid_pre.append((info, start_rel, pre_len))

# -------------------------
    # 第四步：读取 EDF + 提取原始数据
    # -------------------------
    X_list = []
    Y_list = []

    # -- inter-ictal (Y=0)
    for info, rel_start, dur in valid_inter:
        edf_path = os.path.join(sub_dir, info["fname"])
        raw, used_chs = load_edf_channels(edf_path, EDF_CHANNELS_18)
        # 如果通道数量不等于 18，则跳过该文件
        if len(used_chs) != len(EDF_CHANNELS_18):
            print(f"[跳过] {edf_path} 通道数={len(used_chs)}，需要={len(EDF_CHANNELS_18)}")
            continue

        eeg = extract_segment(raw, rel_start, rel_start + dur, FS)
        X_list.append(eeg)
        Y_list.append(0)

    # -- pre-ictal (Y=1)
    for info, rel_start, dur in valid_pre:
        edf_path = os.path.join(sub_dir, info["fname"])
        raw, used_chs = load_edf_channels(edf_path, EDF_CHANNELS_18)
        # 如果通道数量不等于 18，则跳过该文件
        if len(used_chs) != len(EDF_CHANNELS_18):
            print(f"[跳过] {edf_path} 通道数={len(used_chs)}，需要={len(EDF_CHANNELS_18)}")
            continue

        eeg = extract_segment(raw, rel_start, rel_start + dur, FS)
        X_list.append(eeg)
        Y_list.append(1)

    print(f"{sub_name}: 提取成功，总样本={len(X_list)}, pre={Y_list.count(1)}, inter={Y_list.count(0)}")

    # -------------------------
    # 第五步：组装输出
    # -------------------------
    data_subject = {
        'train': {
            'X': X_list,        # 每段 EEG（不同长度）
            'Y': Y_list         # 标签 list
        },
        'channel': EDF_CHANNELS_18,
        'fs': 256,
        'win_sel': WIN_SEC,
        'label_meaning': {
            0: 'inter-ictal（发作间期）',
            1: 'pre-ictal（发作前 1 小时内）'
        }
    }

    return data_subject




if __name__ == "__main__":
    # data_all = load_BNCI2014_001(file_path='H:\All_Dataset\Motor Imagery\BNCI2014_001', subject=1, win_sel=[-1, 4])
    # data_all = load_Lee2019_MI(file_path='H:\All_Dataset\Motor Imagery\Lee2019_MI', subject=1, win_sel=[-1, 4])


    # data_all = load_BNCI2014009(file_path='H:\All_Dataset\P300\BNCI2014009', subject=1, win_sel=[0, 1])




    # data_all = load_TsingHua(file_path='H:\All_Dataset\SSVEP\TsingHua', subject=1, win_sel=[0, 1])
    # data_all = load_Nakanishi2015(file_path='H:\All_Dataset\SSVEP\\Nakanishi2015', subject=1, win_sel=[0, 1])
    # data_all = load_HandWriting(file_path='H:/HandWritingData/正常人/', subject='data20230712/liuch20230712_250Hz', win_sel=[-0.1, 2.1])

    data_all = load_InHouse(file_path='H:\All_Dataset\Motor Imagery\InHouse', subject='cgq20200922_100Hz',win_sel=[-0.5, 4.5])
    print(data_all)
