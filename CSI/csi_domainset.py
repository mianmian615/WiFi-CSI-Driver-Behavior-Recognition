import itertools
import numpy as np
import pandas as pd



# 数据集CSI数据shape定义
# 便于后续根据数据集类型自动适配输入shape

dataset_csi_size = {
    'Peoplecounting': {
        'amp': (6, 114, 2000),
        'pha': (6, 114, 2000),
        'amp+pha': (6, 228, 2000),
        'classes': 8,
    },    
    'Widar3': {
        'amp': (3, 30, 2500),
        'pha': (3, 30, 2500),
        'amp+pha': (3, 60, 2500),
        'amp': (22, 20, 20),
        'classes': 6,
    },
    'WiMANS': {
        'amp': (9, 30, 3000),
        'pha': (9, 30, 3000),
        'amp+pha': (9, 60, 3000),
        'classes': 9,
    },
    'XRF55': {
        'amp': (3, 30, 1000),
        'pha': (3, 30, 1000),
        'amp+pha': (3, 60, 1000),
        'classes': 7,
    },
    'CSIDA': {
        'amp': (3, 114, 1800),
        'pha': (3, 114, 1800),
        'amp+pha': (3, 228, 1800),
        'classes': 6,
    },
    'WCBHAR': {
        'amp': (3, 114, 2000),
        'pha': (3, 114, 2000),
        'amp+pha': (3, 228, 2000),
        'classes': 14,
    },
    'NTUFIHumanID': {
        'amp': (3, 114, 2000),
        'pha': (3, 114, 2000),
        'amp+pha': (3, 228, 2000),
        'classes': 14,
    },
    'NTUFIHAR': {
        'amp': (3, 114, 2000),
        'pha': (3, 114, 2000),
        'amp+pha': (3, 228, 2000),
        'classes': 6,
    },
    'CSI-Finger': {
        'amp': (3, 30, 4000),
        'pha': (3, 30, 4000),
        'amp+pha': (3, 60, 4000),
        'classes': 6,
    },
    'ARIL': {
        'amp': (1, 52, 192),
        'pha': (1, 52, 192),
        'amp+pha': (1, 104, 192),
        'classes': 6,
    },
}


def get_domains(dataset_type, domain_type, ibegin, imax, rxs=None):
    """
    获取指定数据集和domain类型下的所有domain划分方案。
    dataset_type: 数据集类型（如'Widar3', 'CSIDA', 'ARIL'等）
    domain_type: domain划分类型（如'room', 'user', 'loc', 'ori', 'rx'等）
    ibegin, imax: 控制domain组合的起止编号（用于分批处理）
    rxs: 指定接收天线编号（可选）
    返回：domain划分字典
    """
    dataset_domain_list = {}
    # Intel 5300 CSI Tools
    if dataset_type == 'Widar3':
        if domain_type in ['loc', 'ori']:
            dataset_domain_list = get_widar3_all_domains(ibegin, imax, domain_type, dataset_domain_list, rxs=['4'])
        elif domain_type in ['room']:
            dataset_domain_list = get_widar3_all_domains(ibegin, imax, domain_type, dataset_domain_list, rxs=['6'])
        elif domain_type in ['user']:
            dataset_domain_list = get_widar3_all_domains(ibegin, imax, domain_type, dataset_domain_list, rxs=['1'])
        else:
            raise ValueError('wrong')
    if dataset_type == 'WiMANS':
        dataset_domain_list = get_wimans_all_domains(ibegin, imax, domain_type, dataset_domain_list)
    if dataset_type == 'XRF55':
        dataset_domain_list = get_xrf55_all_domains(ibegin, imax, domain_type, dataset_domain_list)
    # Atheros CSI Tools
    if dataset_type == 'Peoplecounting':
        dataset_domain_list = get_Peoplecounting_all_domains(ibegin, imax, domain_type, dataset_domain_list)
    if dataset_type == 'CSIDA':
        dataset_domain_list = get_csida_all_domains(ibegin, imax, domain_type, dataset_domain_list)
    if dataset_type == 'WCBHAR':
        dataset_domain_list = get_wcbhar_all_domains(ibegin, imax, domain_type, dataset_domain_list)
    if dataset_type == 'NTUFIHumanID':
        dataset_domain_list = get_ntufihumanid_all_domains(ibegin, imax, domain_type, dataset_domain_list)
    if dataset_type == 'NTUFIHAR':
        dataset_domain_list = get_ntufihar_all_domains(ibegin, imax, domain_type, dataset_domain_list)
    # Others
    if dataset_type == 'ARIL':
        dataset_domain_list = get_aril_all_domains(ibegin, imax, domain_type, dataset_domain_list)
    return dataset_domain_list


# 以下为各数据集domain划分的具体实现
def get_Peoplecounting_all_domains(ibegin, imax, domain_type, domain_list):
   
    i = 0
    domain_list['Peoplecounting'] = []
    if domain_type == 'group':
        room = '1'
        groups = ['A','B','C']
        source_groups = list(itertools.combinations(groups, 2))
        for source_group in source_groups:
            i += 1
            target_group = set(groups).difference(set(source_group))
            source_domains = []
            target_domains = []
            for group in source_group:
                    source_domain_name = f'Room_{room}_Group_{group}'
                    source_domains.append(source_domain_name)
            for group in target_group:
                    target_domain_name = f'Room_{room}_Group_{group}'
                    target_domains.append(target_domain_name)
            dic = {
                'exp_type': f'cross_{domain_type}{i}', 
                'source_domains': source_domains, 
                'target_domains': target_domains,
                'n_subdomains': {'group': len(source_group)},
                }
            if i >= ibegin:
                domain_list['Peoplecounting'].append(dic)
            if i >= imax:
                return domain_list
    if domain_type == 'room': 
        group = 'A'
        rooms = ['1','2','3']
        source_rooms = list(itertools.combinations(rooms, 2))
        for source_room in source_rooms:
            i += 1
            target_room = set(rooms).difference(set(source_room))
            source_domains = []
            target_domains = []
            for room in source_room:
                    source_domain_name = f'Room_{room}_Group_{group}'
                    source_domains.append(source_domain_name)
            for room in target_room:
                    target_domain_name = f'Room_{room}_Group_{group}'
                    target_domains.append(target_domain_name)
            dic = {
                'exp_type': f'cross_{domain_type}{i}', 
                'source_domains': source_domains, 
                'target_domains': target_domains,
                'n_subdomains': {'room': len(source_room)},
                }
            if i >= ibegin:
                domain_list['Peoplecounting'].append(dic)
            if i >= imax:
                return domain_list

    return domain_list

def get_widar3_all_domains(ibegin, imax, domain_type, domain_list, rxs=None):
    """
    Widar3数据集domain划分生成器。
    支持room/ori/rx/loc/user多种domain划分。
    """
    i = 0
    domain_list['Widar3'] = []
    loc_ids = ['1', '2', '3', '4', '5']
    ori_ids = ['1', '2', '3', '4', '5']
    rx_ids = rxs if rxs is not None else ['1', '2', '3', '4', '5', '6']
    if domain_type == 'room':
        user = '3'
        rooms = ['1', '2', '3']
        for ori in ori_ids:
            for rx in rx_ids:
                for loc in loc_ids:
                    source_rooms = list(itertools.combinations(rooms, 2))
                    for source_room in source_rooms:
                        i += 1
                        target_room = set(rooms).difference(set(source_room))
                        source_domains = []
                        target_domains = []
                        for room in source_room:
                            source_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                            source_domains.append(source_domain_name)
                        for room in target_room:
                            target_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                            target_domains.append(target_domain_name)
                        dic = {
                            'exp_type': f'cross_{domain_type}{i}',
                            'source_domains': source_domains,
                            'target_domains': target_domains,
                            'n_subdomains': {'room': len(source_room)},
                        }
                        if i >= ibegin:
                            domain_list['Widar3'].append(dic)
                        if i >= imax:
                            return domain_list

    # widar_room_user_ids={
    #1:['1','2','3','5','10','11','12','13','14','15','16','17'], #5-17* 1/3****  2 *****
    #2:['1','2','3','6'],#1* 3** 2/6*** 
    # 3:['3','7','8','9'] #3-9 *
    # }  
    if domain_type == 'ori':
        widar_room_user_ids = {
            1: ['1', '2', '3']
            # 3:['3','7','8','9']
            }  
        for room in widar_room_user_ids:
            for loc in loc_ids:
                for rx in rx_ids:
                    for user in widar_room_user_ids[room]:
                        source_oris = list(itertools.combinations(ori_ids, 4))
                        for source_ori in source_oris:
                            i += 1
                            target_ori = set(ori_ids).difference(set(source_ori))
                            source_domains = []
                            target_domains = []
                            for ori in source_ori:
                                source_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                                source_domains.append(source_domain_name)
                            for ori in target_ori:
                                target_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                                target_domains.append(target_domain_name)
                            dic = {
                                'exp_type': f'cross_{domain_type}{i}', 
                                'source_domains': source_domains, 
                                'target_domains': target_domains,
                                'n_subdomains': {'ori': len(source_ori)},
                                }
                            if i >= ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i >= imax:
                                return domain_list

    if domain_type == 'rx':    
        for room in widar_room_user_ids:
            for loc in loc_ids:
                for ori in ori_ids:
                    for user in widar_room_user_ids[room]:
                        source_rxs = list(itertools.combinations(rx_ids, 5))
                        for source_rx in source_rxs:
                            i += 1
                            target_rx = set(rx_ids).difference(set(source_rx))
                            source_domains = []
                            target_domains = []
                            for rx in source_rx:
                                source_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                                source_domains.append(source_domain_name)
                            for rx in target_rx:
                                target_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                                target_domains.append(target_domain_name)
                            dic = {
                                'exp_type': f'cross_{domain_type}{i}', 
                                'source_domains': source_domains, 
                                'target_domains': target_domains,
                                'n_subdomains': {'rx': len(source_rx)},
                                }
                            if i >= ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i >= imax:
                                return domain_list

    if domain_type == 'loc': 
        widar_room_user_ids = {
            3: ['3', '7', '8', '9'] #3-9 *
            }     
        for ori in ori_ids:
            for rx in rx_ids:
                for room in widar_room_user_ids:
                    for user in widar_room_user_ids[room]:
                        source_locs = list(itertools.combinations(loc_ids, 4))
                        for source_loc in source_locs:
                            i += 1
                            target_loc = set(loc_ids).difference(set(source_loc))
                            source_domains = []
                            target_domains = []
                            for loc in source_loc:
                                source_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                                source_domains.append(source_domain_name)
                            for loc in target_loc:
                                target_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                                target_domains.append(target_domain_name)
                            dic = {
                                'exp_type': f'cross_{domain_type}{i}', 
                                'source_domains': source_domains, 
                                'target_domains': target_domains,
                                'n_subdomains': {'loc': len(source_loc)},
                                }
                            if i >= ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i >= imax:
                                return domain_list
    if domain_type == 'user': 
        widar_room_user_ids = {
            3: ['3', '7', '8', '9'] #3-9 *
            }  
        for loc in loc_ids:
            for ori in ori_ids:
                for rx in rx_ids:
                    for room in widar_room_user_ids:
                        source_users = list(itertools.combinations(widar_room_user_ids[room], len(widar_room_user_ids[room])-1))
                        for source_user in source_users:
                            i += 1
                            target_user = set(widar_room_user_ids[room]).difference(set(source_user))
                            source_domains = []
                            target_domains = []
                            for user in source_user:
                                source_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                                source_domains.append(source_domain_name)
                            for user in target_user:
                                target_domain_name = f'room_{room}_user_{user}_loc_{loc}_ori_{ori}_rx_{rx}'
                                target_domains.append(target_domain_name)
                            dic = {
                                'exp_type': f'cross_{domain_type}{i}', 
                                'source_domains': source_domains, 
                                'target_domains': target_domains,
                                'n_subdomains': {'user': len(source_user)},
                                }
                            if i >= ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i >= imax:
                                return domain_list
    return domain_list

def get_wimans_all_domains(ibegin, imax, domain_type, domain_list): #'room_0_user_0_loc_0'
    """
    WiMANS数据集domain划分生成器。
    支持set单一domain划分。
    """
    i = 0
    domain_list['WiMANS'] = []
    user_ids = [str(i) for i in range(6)]    # select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
    loc_ids = [str(i) for i in range(5)]  # select number(s) of locations, (e.g., ['a', 'b'], ['c', 'd', 'e'])
    wifi_bands = ["5","2.4"]                      # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])  
    rooms = [str(i) for i in range(3)]                 # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
    tasks = ["identity", "activity", "location"]
    for band in wifi_bands:
        if domain_type == 'room':
            for user in user_ids:
                for loc in loc_ids:
                    source_rooms = list(itertools.combinations(rooms, 2))
                    for source_room in source_rooms:
                        i += 1
                        target_room = set(rooms).difference(set(source_room))
                        source_domains = []
                        target_domains = []
                        for room in source_room:
                            source_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                            source_domains.append(source_domain_name)
                        for room in target_room:
                            target_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                            target_domains.append(target_domain_name)
                        dic = {
                            'exp_type': f'cross_band_{band}_{domain_type}{i}', 
                            'source_domains': source_domains, 
                            'target_domains': target_domains,
                            'n_subdomains': {'room': len(source_room)},
                            }
                        if i >= ibegin:
                            domain_list['WiMANS'].append(dic)
                        if i >= imax:
                            return domain_list
        if domain_type == 'loc':    
            for room in rooms:
                for user in user_ids:
                    source_locs = list(itertools.combinations(loc_ids, 4))
                    for source_loc in source_locs:
                        i += 1
                        target_loc = set(loc_ids).difference(set(source_loc))
                        source_domains = []
                        target_domains = []
                        for loc in source_loc:
                            source_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                            source_domains.append(source_domain_name)
                        for loc in target_loc:
                            target_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                            target_domains.append(target_domain_name)
                        dic = {
                            'exp_type': f'cross_band_{band}_{domain_type}{i}', 
                            'source_domains': source_domains, 
                            'target_domains': target_domains,
                            'n_subdomains': {'loc': len(source_loc)},
                            }
                        if i >= ibegin:
                            domain_list['WiMANS'].append(dic)
                        
                        if i >= imax:
                            return domain_list
        if domain_type == 'user': 
            for room in rooms:
                for loc in loc_ids:
                    source_users = list(itertools.combinations(user_ids, 5))
                    for source_user in source_users:
                        i += 1
                        target_user = set(user_ids).difference(set(source_user))
                        source_domains = []
                        target_domains = []
                        for user in source_user:
                            source_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                            source_domains.append(source_domain_name)
                        for user in target_user:
                            target_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                            target_domains.append(target_domain_name)
                        dic = {
                            'exp_type': f'cross_band_{band}_{domain_type}{i}', 
                            'source_domains': source_domains, 
                            'target_domains': target_domains,
                            'n_subdomains': {'user': len(source_user)},
                            }
                        if i >= ibegin:
                            domain_list['WiMANS'].append(dic)
                        
                        if i >= imax:
                            return domain_list

        if domain_type == 'room_user': 
            for room in rooms:
                for loc in loc_ids:
                    for user in user_ids: 
                        i += 1
                        target_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                        source_rooms = set(rooms).difference(set(room))
                        source_users = set(user_ids).difference(set(user))
                        source_domains = []
                        for source_user in source_users:
                            for source_room in source_rooms:
                                source_domain_name = f'band_{band}_room_{source_room}_user_{source_user}_loc_{loc}'
                                source_domains.append(source_domain_name)
                        dic = {
                            'exp_type': f'cross_band_{band}_{domain_type}{i}', 
                            'source_domains': source_domains, 
                            'target_domains': [target_domain_name],
                            'n_subdomains': {'room': len(source_rooms),
                                            'user': len(source_users)
                                            },
                            }
                        if i >= ibegin:
                            domain_list['WiMANS'].append(dic) 
                        if i >= imax:
                            return domain_list
        if domain_type == 'room_loc': 
            for room in rooms:
                for user in user_ids: 
                    for loc in loc_ids:
                        i += 1
                        target_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                        source_rooms = set(rooms).difference(set(room))
                        source_locs = set(loc_ids).difference(set(loc))
                        source_domains = []
                        for source_loc in source_locs:
                            for source_room in source_rooms:
                                source_domain_name = f'band_{band}_room_{source_room}_user_{user}_loc_{source_loc}'
                                source_domains.append(source_domain_name)
                        dic = {
                            'exp_type': f'cross_band_{band}_{domain_type}{i}', 
                            'source_domains': source_domains, 
                            'target_domains': [target_domain_name],
                            'n_subdomains': {'room': len(source_rooms),
                                            'loc': len(source_locs),
                                            }
                            }
                        if i >= ibegin:
                            domain_list['WiMANS'].append(dic)
                        
                        if i >= imax:
                            return domain_list

        if domain_type == 'user_loc':    
            for room in rooms:
                for user in user_ids: 
                    for loc in loc_ids:
                        i += 1
                        target_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                        source_users = set(user_ids).difference(set(user))
                        source_locs = set(loc_ids).difference(set(loc))
                        source_domains = []
                        for source_loc in source_locs:
                            for source_user in source_users:
                                source_domain_name = f'band_{band}_room_{room}_user_{source_user}_loc_{source_loc}'
                                source_domains.append(source_domain_name)
                        dic = {
                            'exp_type': f'cross_band_{band}_{domain_type}{i}', 
                            'source_domains': source_domains, 
                            'target_domains': [target_domain_name],
                            'n_subdomains': {'loc': len(source_locs),
                                            'user': len(source_users)},
                            }
                        if i >= ibegin:
                            domain_list['WiMANS'].append(dic)
                        
                        if i >= imax:
                            return domain_list

        if domain_type == 'room_user_loc':    
            for room in rooms:
                for user in user_ids: 
                    for loc in loc_ids:
                        i += 1
                        target_domain_name = f'band_{band}_room_{room}_user_{user}_loc_{loc}'
                        source_users = set(user_ids).difference(set(user))
                        source_rooms = set(rooms).difference(set(room))
                        source_locs = set(loc_ids).difference(set(loc))
                        source_domains = []
                        for source_room in source_rooms:
                            for source_loc in source_locs:
                                for source_user in source_users:
                                    source_domain_name = f'band_{band}_room_{source_room}_user_{source_user}_loc_{source_loc}'
                                    source_domains.append(source_domain_name)
                        dic = {
                            'exp_type': f'cross_band_{band}_{domain_type}{i}', 
                            'source_domains': source_domains, 
                            'target_domains': [target_domain_name],
                            'n_subdomains': {'room': len(source_rooms),
                                            'loc': len(source_locs),
                                            'user': len(source_users)
                                            },
                            }
                        if i >= ibegin:
                            domain_list['WiMANS'].append(dic)
                        
                        if i >= imax:
                            return domain_list
    
    return domain_list

def get_xrf55_all_domains(ibegin, imax, domain_type, domain_list):#'rx_0_user_0'
    """
    XRF55数据集domain划分生成器。
    """
    i = 0
    domain_list['XRF55'] = []
    user_ids = [i for i in range(26)]
    rx_ids = [i for i in range(3)]
    if domain_type == 'user':
        for rx in rx_ids:
            source_users = list(itertools.combinations(user_ids, 25))
            for source_user in source_users:
                i += 1
                target_user = set(user_ids).difference(set(source_user))
                source_domains = []
                target_domains = []
                for user in source_user:
                    source_domain_name = f'rx_{rx}_user_{user}'
                    source_domains.append(source_domain_name)
                for user in target_user:
                    target_domain_name = f'rx_{rx}_user_{user}'
                    target_domains.append(target_domain_name)
                dic = {
                    'exp_type': f'cross_{domain_type}{i}', 
                    'source_domains': source_domains, 
                    'target_domains': target_domains,
                    'n_subdomains': {'user': len(source_user)},
                    }
                if i >= ibegin:
                    domain_list['XRF55'].append(dic)
                if i >= imax:
                    return domain_list
    if domain_type == 'rx': 
        for user in user_ids:
            source_rxs = list(itertools.combinations(rx_ids, 2))
            for source_rx in source_rxs:
                i += 1
                target_rx = set(rx_ids).difference(set(source_rx))
                source_domains = []
                target_domains = []
                for rx in source_rx:
                    source_domain_name = f'rx_{rx}_user_{user}'
                    source_domains.append(source_domain_name)
                for rx in target_rx:
                    target_domain_name = f'rx_{rx}_user_{user}'
                    target_domains.append(target_domain_name)
                dic = {
                    'exp_type': f'cross_{domain_type}{i}', 
                    'source_domains': source_domains, 
                    'target_domains': target_domains,
                    'n_subdomains': {'rx': len(source_rx)},
                    }
                if i >= ibegin:
                    domain_list['XRF55'].append(dic)
                if i >= imax:
                    return domain_list
    return domain_list

def get_csida_all_domains(ibegin, imax, domain_type, domain_list):##room 0-1 user 0-4 loc 0-2
    """
    CSIDA数据集domain划分生成器。
    支持room/loc/user/room_user/room_loc/user_loc/room_user_loc多种domain划分。
    """
    i = 0
    domain_list['CSIDA'] = []
    loc_ids = ['0', '1', '2']
    user_ids = ['0', '1', '2', '3', '4']
    rooms = ['0', '1']
    common_locs = ['0', '1']
    
    csida_room_loc_ids = {
    '0': ['0', '1', '2'], 
    '1': ['0', '1'],
    }  
    if domain_type == 'room':
        for user in user_ids:
            for loc in common_locs:
                source_rooms = list(itertools.combinations(rooms, 1))
                for source_room in source_rooms:
                    i += 1
                    target_room = set(rooms).difference(set(source_room))
                    source_domains = []
                    target_domains = []
                    for room in source_room:
                        source_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                        source_domains.append(source_domain_name)
                    for room in target_room:
                        target_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                        target_domains.append(target_domain_name)
                    dic = {
                        'exp_type': f'cross_{domain_type}{i}', 
                        'source_domains': source_domains, 
                        'target_domains': target_domains,
                        'n_subdomains': {'room': len(source_room)},
                        }
                    if i >= ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i >= imax:
                        return domain_list
    if domain_type == 'loc':    
        for room in rooms:
            for user in user_ids:
                if room == '0':
                    loc_ids = csida_room_loc_ids[room]
                    source_locs = list(itertools.combinations(loc_ids, 2))
                else:
                    loc_ids = csida_room_loc_ids[room]
                    source_locs = list(itertools.combinations(loc_ids, 1))
                for source_loc in source_locs:
                    i += 1
                    target_loc = set(loc_ids).difference(set(source_loc))
                    source_domains = []
                    target_domains = []
                    for loc in source_loc:
                        source_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                        source_domains.append(source_domain_name)
                    for loc in target_loc:
                        target_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                        target_domains.append(target_domain_name)
                    dic = {
                        'exp_type': f'cross_{domain_type}{i}', 
                        'source_domains': source_domains, 
                        'target_domains': target_domains,
                        'n_subdomains': {'loc': len(source_loc)},
                        }
                    if i >= ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i >= imax:
                        return domain_list
    if domain_type == 'user': 
        for room in rooms:
            if room == '0':
                loc_ids = csida_room_loc_ids[room]
            else:
                loc_ids = csida_room_loc_ids[room]
            for loc in loc_ids:
                source_users = list(itertools.combinations(user_ids, 4))
                for source_user in source_users:
                    i += 1
                    target_user = set(user_ids).difference(set(source_user))
                    source_domains = []
                    target_domains = []
                    for user in source_user:
                        source_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                        source_domains.append(source_domain_name)
                    for user in target_user:
                        target_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                        target_domains.append(target_domain_name)
                    dic = {
                        'exp_type': f'cross_{domain_type}{i}', 
                        'source_domains': source_domains, 
                        'target_domains': target_domains,
                        'n_subdomains': {'user': len(source_user)},
                        }
                    if i >= ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i >= imax:
                        return domain_list

    if domain_type == 'room_user': 
        for room in rooms:
            if room == '0':
                loc_ids = csida_room_loc_ids[room]
            else:
                loc_ids = csida_room_loc_ids[room]
            for user in user_ids: 
                for loc in common_locs:
                    i += 1
                    target_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                    source_rooms = set(rooms).difference(set(room))
                    source_users = set(user_ids).difference(set(user))
                    source_domains = []
                    for source_user in source_users:
                        for source_room in source_rooms:
                            source_domain_name = f'room_{source_room}_user_{source_user}_loc_{loc}'
                            source_domains.append(source_domain_name)
                    dic = {
                        'exp_type': f'cross_{domain_type}{i}', 
                        'source_domains': source_domains, 
                        'target_domains': [target_domain_name],
                        'n_subdomains': {'room': len(source_rooms),
                                         'user': len(source_users)
                                        },
                        }
                    if i >= ibegin:
                        domain_list['CSIDA'].append(dic) 
                    if i >= imax:
                        return domain_list
    if domain_type == 'room_loc': 
        for room in rooms:
            if room == '0':
                loc_ids = csida_room_loc_ids[room]          
            else:
                loc_ids = csida_room_loc_ids[room]
            for user in user_ids: 
                for loc in loc_ids:
                    i += 1
                    target_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                    source_rooms = set(rooms).difference(set(room))
                    for source_room in source_rooms:
                        source_locs = set(csida_room_loc_ids[source_room]).difference(set(loc))
                        n_loc = len(source_locs)
                        source_domains = []
                        for source_loc in source_locs:
                            source_domain_name = f'room_{source_room}_user_{user}_loc_{source_loc}'
                            source_domains.append(source_domain_name)
                    dic = {
                        'exp_type': f'cross_{domain_type}{i}', 
                        'source_domains': source_domains, 
                        'target_domains': [target_domain_name],
                        'n_subdomains': {'room': len(source_rooms),
                                         'loc': len(source_locs),
                                        }
                        }
                    if i >= ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i >= imax:
                        return domain_list

    if domain_type == 'user_loc':    
        for room in rooms:
            if room == '0':
                loc_ids = csida_room_loc_ids[room]
                n_loc = 2
            else:
                loc_ids = csida_room_loc_ids[room]
                n_loc = 1
            for user in user_ids: 
                for loc in loc_ids:
                    i += 1
                    target_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                    source_users = set(user_ids).difference(set(user))
                    source_locs = set(loc_ids).difference(set(loc))
                    source_domains = []
                    for source_loc in source_locs:
                        for source_user in source_users:
                            source_domain_name = f'room_{room}_user_{source_user}_loc_{source_loc}'
                            source_domains.append(source_domain_name)
                    dic = {
                        'exp_type': f'cross_{domain_type}{i}', 
                        'source_domains': source_domains, 
                        'target_domains': [target_domain_name],
                        'n_subdomains': {'loc': len(source_locs),
                                        'user': len(source_users)},
                        }
                    if i >= ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i >= imax:
                        return domain_list

    if domain_type == 'room_user_loc':    
        for room in rooms:
            if room == '0':
                loc_ids = csida_room_loc_ids[room]
                n_loc = 2
            else:
                loc_ids = csida_room_loc_ids[room]
                n_loc = 1
            for user in user_ids: 
                for loc in loc_ids:
                    i += 1
                    target_domain_name = f'room_{room}_user_{user}_loc_{loc}'
                    source_users = set(user_ids).difference(set(user))
                    source_rooms = set(rooms).difference(set(room))
                    source_domains = []
                    for source_room in source_rooms:
                        source_locs = set(csida_room_loc_ids[source_room]).difference(set(loc))
                        n_loc = len(source_locs)
                        for source_loc in source_locs:
                            for source_user in source_users:
                                source_domain_name = f'room_{source_room}_user_{source_user}_loc_{source_loc}'
                                source_domains.append(source_domain_name)
                    dic = {
                        'exp_type': f'cross_{domain_type}{i}', 
                        'source_domains': source_domains, 
                        'target_domains': [target_domain_name],
                        'n_subdomains': {'room': len(source_rooms),
                                         'loc': len(source_locs),
                                        'user': len(source_users)
                                        },
                        }
                    if i >= ibegin:
                        domain_list['CSIDA'].append(dic)
                    
                    if i >= imax:
                        return domain_list
    
    return domain_list

def get_wcbhar_all_domains(ibegin, imax, domain_type, domain_list):
    """
    WCBHAR数据集domain划分生成器。
    """
    i = 0
    domain_list['WCBHAR'] = []
    room_ids = [i for i in range(3)]
    if domain_type == 'room':
        source_rooms = list(itertools.combinations(room_ids, 2))
        for source_room in source_rooms:
            i += 1
            target_room = set(room_ids).difference(set(source_room))
            source_domains = []
            target_domains = []
            for room in source_room:
                source_domain_name = f'room_{room}'
                source_domains.append(source_domain_name)
            for room in target_room:
                target_domain_name = f'room_{room}'
                target_domains.append(target_domain_name)
            dic = {
                'exp_type': f'cross_{domain_type}{i}', 
                'source_domains': source_domains, 
                'target_domains': target_domains,
                'n_subdomains': {'room': len(source_room)},
                }
            if i >= ibegin:
                domain_list['WCBHAR'].append(dic)
            if i >= imax:
                return domain_list
    return domain_list

def get_ntufihumanid_all_domains(ibegin, imax, domain_type, domain_list):
    """
    NTUFI数据集domain划分生成器。
    支持set单一domain划分。
    """
    i = 0
    domain_list['NTUFIHumanID'] = []
    set_ids = ['a', 'b', 'c']
    for a in range(len(set_ids)):
        i += 1
        target_domains = [set_ids[a]]  
        source_domains = []
        for b in range(len(set_ids)):
            if b != a:
                source_domains.append(set_ids[b])       
        dic = {
            'exp_type': f'cross_{domain_type}{i}', 
            'source_domains': source_domains, 
            'target_domains': target_domains,
            'n_subdomains': {'set': 2},
            }
        if i >= ibegin:
            domain_list['NTUFIHumanID'].append(dic)

        if i >= imax:
            return domain_list
    return domain_list


def get_ntufihar_all_domains(ibegin, imax, domain_type, domain_list):
    """
    NTUFI数据集domain划分生成器。
    支持set单一domain划分。
    """
    i = 0
    domain_list['NTUFIHAR'] = []
    user_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for a in range(len(user_ids)):
        i += 1
        target_domains = [user_ids[a]]  
        source_domains = []
        for b in range(len(user_ids)):
            if b != a:
                source_domains.append(user_ids[b])       
        dic = {
            'exp_type': f'cross_{domain_type}{i}', 
            'source_domains': source_domains, 
            'target_domains': target_domains,
            'n_subdomains': {'user': 9},
            }
        if i >= ibegin:
            domain_list['NTUFIHAR'].append(dic)

        if i >= imax:
            return domain_list
    return domain_list

def get_aril_all_domains(ibegin, imax, domain_type, domain_list):##room 0-1 user 0-4 loc 0-2
    """
    ARIL数据集domain划分生成器。
    支持loc单一domain划分。
    """
    i = 0
    domain_list['ARIL'] = []
    loc_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for a in range(len(loc_ids)):
        i += 1
        target_domains = [str(loc_ids[a])]  
        source_domains = []
        for b in range(len(loc_ids)):
            if b != a:
                source_domains.append(str(loc_ids[b]))       
        dic = {
            'exp_type': f'cross_{domain_type}{i}', 
            'source_domains': source_domains, 
            'target_domains': target_domains,
            'n_subdomains': {'loc': 15},
            }
        if i >= ibegin:
            domain_list['ARIL'].append(dic)
        
        if i >= imax:
            return domain_list
    return domain_list