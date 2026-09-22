
import numpy as np
import pandas as pd
from .data_process.signal_process import deal_CSI
from .data_process.extract_csi import extract_CSI_dat
import re, os, pickle
import scipy.io as scio
import zarr
from pathlib import Path
import glob, torch
import scipy.io as sio
import csiread
from CSIKit.reader import IWLBeamformReader
from CSIKit.util import csitools

#一次性处理该域下的所有文件
def get_Peoplecounting_csi(root_dir, domain_name):

    people_class = ['People1_In', 'People1_Out', 'People2_In', 'People2_Out',
                    'People3_In', 'People3_Out', 'People4_In', 'People4_Out']
    people_numbers = ['1','2','3','4']
    directions = ['In','Out']
    index = domain_name.split('_')
    roomid = int(index[index.index('Room') + 1]) if 'Room' in index else 0
    groupid = index[index.index('Group') + 1] if 'Group' in index else 0

    all_amp, all_pha, all_label = [], [], []

    for  people_number in people_numbers:
        for direction in directions:
            #Room_Group/Speed2_Inter1_People_Out_01-50
            for i in range(1,51):
                #基础设置下为Speed2_Inter1
                dat_file_name = f'Speed2_Inter1_People{people_number}_{direction}_{i:02d}.dat'
                dat_file_dir = os.path.join(root_dir, f'Room{roomid}_Group{groupid}', dat_file_name)
                #print(f"检查文件路径: {dat_file_dir}") 
                if os.path.isfile(dat_file_dir):
                    dat_data = csiread.Atheros(dat_file_dir, 3,2,tones=114,if_report=False)
                    dat_data.read()
                    csi_data = dat_data.csi # (2003, 114, rx-3(接收：实际2）,tx-2(实际为3))
                    #print('csi原始格式：',csi_data.shape) #(1998, 114, 3, 2)
                    data_amp = []
                    data_pha = []
                    ##deal_CSI输入数据为三维，（天线*3，子载波，时间）
                    for j in range(csi_data.shape[-1]): 
                        csi = csi_data[ :, :, :, j].transpose(2, 1, 0)
                        #print('csi输入格式：',csi.shape)
                        amp, pha = deal_CSI(csi, IFfilter=True, IFphasani=True, padding_length=2000)
                        data_amp.append(amp)
                        data_pha.append(pha)
                    all_amp.append(data_amp)
                    all_pha.append(data_pha)
                    #print(people_class.index(f'People{people_number}_{direction}'))
                    all_label.append(people_class.index(f'People{people_number}_{direction}'))
                else:
                    print(f'缺少dat: {dat_file_dir}')
                    #此处用于调试阶段
                    # raise ValueError(f'缺少mat: {dat_file_dir}')

    all_amp = np.array(all_amp)#(n,2,3,114,2000)
    all_pha = np.array(all_pha)
    #(n,6,114,2000)
    #print(all_amp.shape,all_pha.shape)
    all_amp = all_amp.reshape(all_amp.shape[0], all_amp.shape[1] *all_amp.shape[2], all_amp.shape[3],all_amp.shape[-1])
    all_pha = all_pha.reshape(all_pha.shape[0], all_pha.shape[1] * all_pha.shape[2], all_pha.shape[3], all_pha.shape[-1])

    all_label = np.array(all_label)#(n,)
    #print(all_amp.shape,all_pha.shape,all_label.shape)                

    return all_amp, all_pha, all_label


# Widar手势类别

widar_gestures = [
    "Push&Pull", "Sweep", "Clap", "Slide", "Draw-O(Horizontal)", "Draw-Zigzag(Horizontal)"
]

def get_widar_csi(root_dir, domain_name):
    """
    加载Widar3 CSI数据，支持多种domain筛选。
    返回: amp, pha, label, room, user, loc, ori
    """
    root_dir = os.path.join(root_dir, "CSI/")
    index = domain_name.split('_')
    # 解析domain参数
    def parse_index(key, default, all_list):
        if key in index:
            val = index[index.index(key) + 1]
            return val, [val]
        else:
            return default, all_list
    roomid, room_ids = parse_index('room', '1-3', ['1', '2', '3'])
    userid, user_ids = parse_index('user', '3', ['3'])
    gesid, ges_ids = parse_index('ges', '1-6', widar_gestures)
    locid, loc_ids = parse_index('loc', '1-5', ['1', '2', '3', '4', '5'])
    oriid, ori_ids = parse_index('ori', '1-5', ['1', '2', '3', '4', '5'])
    rxid, rx_ids = parse_index('rx', '1-6', ['1', '2', '3', '4', '5', '6'])
    # 构建缓存文件名
    data_file_name = f'room_{roomid}_user_{userid}_ges_{gesid}_loc_{locid}_ori_{oriid}_rx_{rxid}_dealcsidata.pkl'
    data_file = os.path.join(root_dir, f'room_{roomid}/user_{userid}/{data_file_name}')
    if not os.path.isfile(data_file):
        all_amp, all_pha, all_label = [], [], []
        for room in room_ids:
            for user in user_ids:
                for ges in ges_ids:
                    for loc in loc_ids:
                        for ori in ori_ids:
                            for rx in rx_ids:
                                mat_file_name = f'room_{room}_user_{user}_ges_{ges}_loc_{loc}_ori_{ori}_rx_{rx}_csi.mat'
                                mat_file = os.path.join(root_dir, f'matfile/{mat_file_name}')
                                if os.path.isfile(mat_file):
                                    mat = scio.loadmat(mat_file)
                                    print('处理：', mat_file)
                                    mat_datas = list(mat.values())[-1][0]
                                    for csi_data in mat_datas:
                                        amp, pha = deal_CSI(csi_data, IFfilter=True, IFphasani=True, padding_length=2500)
                                        all_amp.append(amp)
                                        all_pha.append(pha)
                                        all_label.append(widar_gestures.index(ges) if ges in widar_gestures else int(ges)-1)
                                else:
                                    raise ValueError(f'缺少mat: {mat_file}')
        all_amp = np.array(all_amp)
        all_pha = np.array(all_pha)
        all_label = np.array(all_label)
        with open(data_file, 'wb') as f:
            pickle.dump(all_amp, f)
            pickle.dump(all_pha, f)
            pickle.dump(all_label, f)
    else:
        with open(data_file, 'rb') as f:
            print('extracting:', data_file_name)
            all_amp = pickle.load(f)
            all_pha = pickle.load(f)
            all_label = pickle.load(f)
    # 构建domain标签
    n = all_amp.shape[0]
    all_room = np.full(n, int(roomid), dtype=int)
    all_user = np.full(n, int(userid), dtype=int)
    all_loc = np.full(n, int(locid), dtype=int)
    all_ori = np.full(n, int(oriid), dtype=int)
    return all_amp, all_pha, all_label, all_room, all_user, all_loc, all_ori

def get_CSIDA_csi(root_dir, domain_name):
    """
    加载CSIDA CSI数据，支持room/user/loc筛选。
    返回: amp, pha, label, room, user, loc, ori
    """
    index = domain_name.split('_')
    def get_val(key):
        return int(index[index.index(key) + 1]) if key in index else 0
    roomid = get_val('room')
    userid = get_val('user')
    locid = get_val('loc')
    group = zarr.open_group(Path(root_dir+'CSI_301/').as_posix(), mode="r")
    all_csi = group.csi_data_raw[:]#(2844, 1800, 3, 114)
    all_amp = group.csi_data_amp[:]#(2844, 1800, 3, 114)
    all_pha = group.csi_data_pha[:]
    all_gesture = group.csi_label_act[:]  # 0~5
    room_label = group.csi_label_env[:]  # 0,1
    loc_label = group.csi_label_loc[:]  # 0,1,2
    user_label = group.csi_label_user[:]  # 0,1,2,3,4
    
    index = np.arange(all_gesture.shape[0])
    mask = (room_label == roomid) & (user_label == userid) & (loc_label == locid)
    select = index[mask]
    all_sel_amp = all_amp[select]
    all_sel_pha = all_pha[select]
    all_sel_csi = all_csi[select]
    all_sel_gesture = all_gesture[select]
    # 转换为(n, 3, 114, 1800)
    all_sel_amp = all_sel_amp.transpose(0, 2, 3, 1)
    all_sel_pha = all_sel_pha.transpose(0, 2, 3, 1)
    all_sel_csi = all_sel_csi.transpose(0, 2, 3, 1)
    n = all_sel_amp.shape[0]
    sel_room = np.full(n, roomid, dtype=int)
    sel_user = np.full(n, userid, dtype=int)
    sel_loc = np.full(n, locid, dtype=int)
    ori_id = np.zeros(n, dtype=int)
    return all_sel_amp, all_sel_pha, all_sel_gesture, sel_room, sel_user, sel_loc, ori_id

def get_ARIL_csi(root_dir, domain_name):
    """
    加载ARIL CSI数据，按location筛选。
    返回: amp, pha, label, room, user, loc, ori
    """
    train_amp_data = scio.loadmat(os.path.join(root_dir, "train_data_split_amp.mat"))
    train_amp = train_amp_data['train_data']
    train_label = train_amp_data['train_activity_label']
    train_locids = train_amp_data['train_location_label']
    train_pha_data = scio.loadmat(os.path.join(root_dir, "train_data_split_pha.mat"))
    train_pha = train_pha_data['train_data']
    assert (train_pha_data['train_activity_label'] == train_label).all()
    assert (train_pha_data['train_location_label'] == train_locids).all()
    test_amp_data = scio.loadmat(os.path.join(root_dir, "test_data_split_amp.mat"))
    test_amp = test_amp_data['test_data']
    test_label = test_amp_data['test_activity_label']
    test_locids = test_amp_data['test_location_label']
    test_pha_data = scio.loadmat(os.path.join(root_dir, "test_data_split_pha.mat"))
    test_pha = test_pha_data['test_data']
    assert (test_pha_data['test_activity_label'] == test_label).all()
    assert (test_pha_data['test_location_label'] == test_locids).all()
    domain_label = int(domain_name)
    amp = np.concatenate((train_amp, test_amp), axis=0)
    pha = np.concatenate((train_pha, test_pha), axis=0)
    label = np.concatenate((train_label, test_label), axis=0)
    locids = np.concatenate((train_locids, test_locids), axis=0)
    label = np.squeeze(label).astype(np.int64)
    locids = np.squeeze(locids)
    index_arr = np.arange(label.shape[0])
    select = index_arr[locids == domain_label]
    all_amp = amp[select]
    all_pha = pha[select]
    all_label = label[select]
    n = all_amp.shape[0]
    room_id = np.zeros(n, dtype=int)
    user_id = np.zeros(n, dtype=int)
    loc_id = np.full(n, domain_label, dtype=int)
    ori_id = np.zeros(n, dtype=int)
    return all_amp, all_pha, all_label, room_id, user_id, loc_id, ori_id

def get_ntufi_humanid_csi(root_dir, domain_name):
    """
    加载NTU-Fi-HumanID CSI数据。
    根据domain_name，在root_dir下所有以0开头的文件夹中搜索包含domain_name的mat文件，提取CSIamp。
    返回: all_amp, all_y
    """
    classes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '015']
    all_amp = []
    all_y = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            raise ValueError(f"文件夹 {class_dir} 不存在")
    

    # # 获取所有以0开头的文件夹
    # all_folders = [f for f in glob.glob(os.path.join(root_dir, '0*')) if os.path.isdir(f)]
    # # 文件夹名转为数字并排序
    # folder_num_map = {folder: int(os.path.basename(folder)) for folder in all_folders}
    # sorted_folders = sorted(folder_num_map.items(), key=lambda x: x[1])
    # folder_to_label = {folder: idx for idx, (folder, _) in enumerate(sorted_folders)}

    # all_amp = []
    # all_y = []
    # for folder, _ in sorted_folders:
        # 搜索该文件夹下所有包含domain_name的mat文件
        mat_files = glob.glob(os.path.join(class_dir, f'*{domain_name}*.mat'))
        for mat_file in mat_files:
            mat = sio.loadmat(mat_file)
            # 假设CSIamp字段名为'CSIamp'，如有不同请修改
            if 'CSIamp' in mat:
                amp = mat['CSIamp']
                # amp = (amp - 42.3199)/4.9802 # 可选归一化
                # amp = amp[:,::4]  # 可选下采样
                # amp = amp.reshape(3, 114, 500)  # 可选
                amp = amp.reshape(3, 114, 2000)
            else:
                raise ValueError(f"mat文件 {mat_file} 中未找到 'CSIamp' 字段")
            all_amp.append(amp)
            all_y.append(class_idx)
    if all_amp:
        all_amp = np.array(all_amp)
        all_y = np.array(all_y)
    else:
        all_amp = np.array([])
        all_y = np.array([])
    return all_amp, all_y

def get_ntufi_har_csi(root_dir, domain_name):
    """
    加载NTU-Fi-HAR CSI数据。
    在root_dir+classes[i]的所有文件夹中，搜索包含classes[i]+str((domain_name*20)+0) 到 classes[i]+str((domain_name*20)+19) 的mat文件，提取CSIamp。
    root_dir+classes[i]下的文件夹名按规则转为数字，作为标签。
    返回: all_amp, all_y
    """
    classes = ['clean', 'circle', 'run', 'fall', 'walk', 'box']
    base = int(domain_name) * 20
    all_amp = []
    all_y = []
    for class_idx, class_name in enumerate(classes):
        label = class_idx
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            raise ValueError(f"文件夹 {class_dir} 不存在")
        # 生成要查找的文件名片段列表
        file_keys = [f"{class_name}{base + i}" for i in range(20)]
        for key in file_keys:
            mat_files = glob.glob(os.path.join(class_dir, f'*{key}*.mat'))
            for mat_file in mat_files:
                mat = sio.loadmat(mat_file)
                if 'CSIamp' in mat:
                    amp = mat['CSIamp']
                    # amp = (amp - 42.3199)/4.9802 # 可选归一化
                    # amp = amp[:,::4]  # 可选下采样
                    # amp = amp.reshape(3, 114, 500)  # 可选
                    amp = amp.reshape(3, 114, 2000)
                else:
                    raise ValueError(f"mat文件 {mat_file} 中未找到 'CSIamp' 字段")
                all_amp.append(amp)
                all_y.append(label)
    if all_amp:
        all_amp = np.array(all_amp)
        all_y = np.array(all_y)
    else:
        all_amp = np.array([])
        all_y = np.array([])
    return all_amp, all_y

def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+'/data/*.csv')
    label_list = glob.glob(root_dir+'/label/*.csv')
    all_amp = []
    all_label = []
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.transpose(0, 2, 1)
            data = data.reshape(-1,3,30,250)
            # data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        all_amp.append(data)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        all_label.append(label)
    return all_amp, all_label


def get_wimans_csi(root_dir, domain_name): #'band_5_room_0_user_0_loc_0'
    """
    加载WiMANS CSI数据。
    有padding, 滤波和相位校正
    """
    user_ids = ["0", "1", "2", "3", "4", "5"]    # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
    loc_ids = ["a", "b", "c", "d", "e"]  # select number(s) of locations, (e.g., ['a', 'b'], ['c', 'd', 'e'])
    wifi_bands = ["2.4", "5"]                      # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])  
    rooms = ["classroom", "meeting_room", "empty_room"]                  # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
    tasks = ["identity", "activity", "location"]
    length=3000  # default length of CSI

    activity=["nothing","walk","rotation","jump","wave","lie_down","pick_up","sit_down","stand_up"]
    
    index = domain_name.split('_')
    bandid = index[index.index('band') + 1] if 'band' in index else 0
    roomid = int(index[index.index('room') + 1]) if 'room' in index else 0
    userid = int(index[index.index('user') + 1]) if 'user' in index else 0
    locid = int(index[index.index('loc') + 1]) if 'loc' in index else 0
    
    label_dir = os.path.join(root_dir, 'annotation.csv')
    data_dir = os.path.join(root_dir, 'wifi_csi/mat')
    ## load annotation file as labels
    data_pd_y = pd.read_csv(label_dir, dtype = str)
    data_pd_y = data_pd_y[data_pd_y["number_of_users"].isin(['1'])]
    data_pd_y = data_pd_y[data_pd_y["wifi_band"].isin([bandid])]
    data_pd_y = data_pd_y[data_pd_y["environment"].isin([rooms[roomid]])]
    data_pd_y = data_pd_y[data_pd_y[f'user_{userid+1}_location'].isin([loc_ids[locid]])]
    
    data_activity_pd_y = data_pd_y[f'user_{userid+1}_activity']
    data_activity_y = data_activity_pd_y.to_numpy(copy = True).astype(str)
    data_y = np.array([activity.index(var_y) for var_y in data_activity_y])
    ## load CSI 
    var_label_list = data_pd_y["label"].to_list()
    var_path_list = [os.path.join(data_dir, var_label + ".mat") for var_label in var_label_list]
    data_amp = []
    data_pha = []
    for var_path in var_path_list:
        data = scio.loadmat(var_path)
        var_length = data["trace"].shape[0]
        csi = np.array([data["trace"][var_t][0][0][0][-1] for var_t in range(var_length)]) # (length, 3, 3, 30)
        csi = csi.transpose(1,2,3,0)  # (3, 3, 30, length)
        deal_amp = []
        deal_pha = []
        for i in range(csi.shape[0]): 
            #csi = csi[i].reshape(csi.shape[0], csi.shape[1]*csi.shape[2], csi.shape[-1])  # (3, 90, length)
            amp, pha = deal_CSI(csi[i], IFfilter=True, IFphasani=True, padding_length=length)
            deal_amp.append(amp)
            deal_pha.append(pha)
        data_amp.append(deal_amp)
        data_pha.append(deal_pha)
        # deal_amp = deal_amp.reshape(3, 3,
        #amp, pha = deal_CSI(csi, IFfilter=True, IFphasani=True, padding_length=length)
        # amp=amp.reshape(3,3,30, amp.shape[-1])
        # pha=pha.reshape(3,3,30, pha.shape[-1])
        # amp=amp.reshape(3,3,30, amp.shape[-1])
        # pha=pha.reshape(3,3,30, pha.shape[-1])
        
    # reshape
    data_amp = np.array(data_amp)
    data_pha = np.array(data_pha)
    data_amp = data_amp.reshape(data_amp.shape[0], data_amp.shape[1] * data_amp.shape[2], data_amp.shape[3], data_amp.shape[-1])
    data_pha = data_pha.reshape(data_pha.shape[0], data_pha.shape[1] * data_pha.shape[2], data_pha.shape[3], data_pha.shape[-1])

    return data_amp, data_pha, data_y

def get_xrf55_csi(root_dir, domain_name): #'rx_0_user_0'
    """
    加载XRF55 CSI数据。
    有padding, 滤波和相位校正
    """
    user_ids = ["01", "02", "03", 
                # "04", "07", "11", #这几个志愿者的lb 原始数据有点问题
                # "12", #这几个志愿者的lf 原始数据有点问题
                "05", "06",  "08", "09", "10",
                 "13", "14", "15", "16", "17", "18", "19", "20",
                "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                ]    
    rxs = ["lb", "lf", "rb"]                     
    # rooms = ["Scene_1", "Scene_2", "Scene_3", "Scene_4"]                 
    length=1000  # default length of CSI
    Human_Object_Interaction_Actions = ["01", "02", "03", "04", "05", "06","07", "08", "09", "10", "11", "12", "13", "14", "15"] 
    #15 Human-Object Interaction Actions. 
    # Whole-home daily: carrying weight, mopping the floor, using a phone, throwing something, picking something, putting something on the table
    # Kitchen: cutting something; Dress: wearing a hat, putting on clothing; Bathroom: blowing dry hair, combing hair, brushing teeth; Healthcare: drinking, eating, and smoking
    Human_Human_Interaction_Actions = ["16", "17", "18", "19", "20","21", "22"] 
    #7 Human-Human Interaction Actions.
    # Social actions: shaking hands, hugging, handing something to someone; 
    # Violence actions for applications of domestic violence and invasion detection: kicking someone, hitting someone with something, choking someone’s neck, and pushing someone.
    FitnessActions=["23", "24", "25", 
                    #"26", 
                    "27", "28", "29", "30"]
    #8 Fitness Actions. With equipment: hula hooping, weightlifting, jumping rope; Without equipment: body weight squats, Tai Chi, boxing, jumping jack, and high leg lifting.
    Body_Motion_Actions = ["31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44"]
    #14 Body Motion Actions. Whole-home daily: waving, clapping hands, jumping, walking, turning, running, sitting down, standing up; Healthcare: falling on the floor, stretching, patting on the shoulder; Musical instruments: playing Er-Hu, playing Ukulele, playing drum.
    Human_Computer_Interaction_Actions = ["45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55"]
    #11 Human-Computer Interaction Actions. Hand gestures: pushing, pulling, swiping left, swiping right, swiping up, swiping down, drawing a circle, drawing a cross; When hands are not free: foot stamping, shaking head, and nodding.
    activity = Human_Human_Interaction_Actions
    
    index = domain_name.split('_')
    rxid = int(index[index.index('rx') + 1]) if 'rx' in index else 0
    userid = int(index[index.index('user') + 1]) if 'user' in index else 0

    data_dir = os.path.join(root_dir+ f'XRF55_rawdata/WiFi/Scene_1/{rxs[rxid]}/{user_ids[userid]}/')
    # data_dir = os.path.join(root_dir+ f'XRF55/Scene1/WiFi/')
    # 搜索相关dat文件
    matched_files = [
    file for key in activity
    for file in glob.glob(os.path.join(data_dir, f"{user_ids[userid]}_{key}*"))
]
    data_amp = []
    data_pha = []
    data_y = []
    for file in sorted(matched_files):
        filename = file.split(".")[0]
        label = activity.index(filename.split("_")[-2])
        # csi = extract_CSI_dat(file)
        # csi = np.load(file) 
        # if file in ["/mnt/datasets/XRF55/XRF55_rawdata/WiFi/Scene_1/lb/04/04_26_11.mat"]:
        #     print("Skipping file:", file)
        #     continue
        # else:
        #     # 读取CSI数据
        # my_reader = IWLBeamformReader()
        # csi = my_reader.read_file(file)
        csidata = csiread.Intel(file, nrxnum=3, ntxnum=1, pl_size=10)
        csidata.read()
        csi = csidata.get_scaled_csi()
        # if len(csi_data.frames) == 0:
        #     continue
        # else:
        # csi, no_frames, no_subcarriers = csitools.get_CSI(csi_data) #可填metric == "amplitude" or  "phase"
        # CSI matrix is now returned as (no_frames, no_subcarriers, no_rx_ant, no_tx_ant).
        # if csi is None:
        #     print('No CSI data found in file:', file)
        #     continue
        csi = np.squeeze(csi)
        csi = csi.transpose(2,1,0)  # (9, 30, length)
        # csi = csi.reshape(3,3,30,csi.shape[-1])
        # amp, pha = csi[rxid], csi[rxid]
        amp, pha = deal_CSI(csi, IFfilter=True, IFphasani=True, padding_length=length)
        data_amp.append(amp)
        data_pha.append(pha)
        data_y.append(label)
    data_amp = np.array(data_amp)
    data_pha = np.array(data_pha)
    data_y = np.array(data_y)
    return data_amp, data_pha, data_y#(n, 9, 30, length)

# def get_mmfi_csi(root_dir, domain_name): #'rx_0_user_0'
#     """
#     加载MMFi CSI数据。
#     有padding, 滤波和相位校正
#     """                   
#     rooms = ["E01", "E02", "E03", "E04"]                 

#     user_ids = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
#                     'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
#                     'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
#     all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
#                    'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
#     Daily_actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
#     Rehabilitation_actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
#     activities = Daily_actions
        
#     index = domain_name.split('_')
#     roomid = int(index[index.index('room') + 1]) if 'room' in index else 0
#     userid = int(index[index.index('user') + 1]) if 'user' in index else 0

#     data_dir = os.path.join(root_dir, f'Data/{rooms[roomid]}/{user_ids[userid]}/')
#     # ground_truth = np.load(os.path.join(data_dir, f'A02/ground_truth.npy'))
#     # data_dir = os.path.join(root_dir+ f'XRF55/Scene1/WiFi/')

#     matched_dirs = [os.path.join(data_dir, f"{key}/wifi-csi/") for key in activities]
#     data_amp = []
#     data_pha = []
#     data_y = []
#     for dir in matched_dirs:
#         for csi_mat in sorted(glob.glob(os.path.join(dir, "frame*.mat"))):
#             amp = scio.loadmat(csi_mat)['CSIamp'] #(3, 114, 10)
#             pha = scio.loadmat(csi_mat)['CSIphase'] #(3, 114, 10)
#             # amp[np.isinf(amp)] = np.nan
#             # pha[np.isinf(pha)] = np.nan
#     # for subject, actions in self.data_source.items():
#     return data_amp, data_pha, data_y

# def get_wcbhar_csi(root_dir, domain_name): #'room_0_user_0'
#     """
#     加载WCBHAR CSI数据。
#     有padding, 滤波和相位校正
#     """  
#     index = domain_name.split('_')
#     roomid = int(index[index.index('room') + 1]) if 'room' in index else 0
#     paths = sorted(glob.glob(os.path.join(root_dir, f"room_{roomid+1}/*")) )
    
#     data_amp = []
#     data_pha = []
#     data_y = []
#     for index, path in enumerate(paths):
#         data_x = pd.read_csv(os.path.join(path, "data.csv"), header=None).values #(5230, 1026)
#         amplitudes = data_x[:, 114 * 1:114 * (1 + 4)] #(5230, 456)
#         phases = data_x[:, 114 * (1 + 4):114 * (1 + 2 * 4)] #(5230, 456)
#         amplitudes, phases = amplitudes[:-1], phases[:-1]  # fix the bug with the last element
#         amplitudes = amplitudes.reshape((-1, 114, 4))
#         phases = phases.reshape((-1, 114, 4))
#         amplitudes = amplitudes.transpose(1,2,0)
#         phases = phases.transpose(1,2,0)

#         data_y = pd.read_csv(os.path.join(path, "label.csv"), header=None).values
#         labels = data_y[:, 1]
        
        
#         data_len = phases.shape[0]
#         label_keys = list(set(labels))
#         class_to_idx = {
#             "standing": 0,
#             "walking": 1,
#             "get_down": 2,
#             "sitting": 3,
#             "get_up": 4,
#             "lying": 5,
#             "no_person": 6
#         }
#         idx_to_class = {v: k for k, v in class_to_idx.items()}

#         window = 32
#         step = 1
#         idx = index * step
#         all_xs, all_ys = [], []
#         # idx = idx * self.window

#         for index in range(idx, idx + window):
#             all_xs.append(np.append(amplitudes[index], amplitudes_pca[index]))
#             # all_ys.append(self.class_to_idx[self.labels[index]])

#         return np.array(all_xs), class_to_idx[labels[idx + window - 1]]

#         data_amp.append(amplitudes)
#         data_pha.append(phases)
#         data_y.append(labels)
#     data_amp = np.array(data_amp)
#     data_pha = np.array(data_pha)
#     data_y = np.array(data_y)
#     return data_amp, data_pha, data_y
