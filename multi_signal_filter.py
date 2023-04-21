import numpy as np


class MultiSignalFilter(object):
    def __init__(self):
        self.signal_cache = {}      # 存储各id信号机轨迹的字典
        self.cache_size = 50        # 信号机连续轨迹长度

    def update(self, bbox, identities=None):
        current_id = []     # 当前帧包含信号机的id

        # 更新各id信号机轨迹信息
        for i, box in enumerate(bbox):
            # 获取当前帧包含信号机的id，并添加到存储列表中
            id = identities[i]
            if id not in current_id:
                current_id.append(id)

            # 新出现的id需要创建轨迹存储列表
            if id not in self.signal_cache:
                self.signal_cache[id] = []

            # 写入信号机轨迹数据，限制长度为cache_size
            if len(self.signal_cache[id]) >= self.cache_size:
                del(self.signal_cache[id][0])
            position = [(box[0]+box[2])/2, (box[1]+box[3])/2]
            self.signal_cache[id].append(position)

        # 对当前帧包含信号机进行筛选过滤
        target_signal_id = self.filter(current_id)

        return target_signal_id

    def filter(self, current_id):
        # if len(current_id) == 1:
        #     return current_id[0]
        for id in current_id:
            # 轨迹长度不足，无法进行筛选
            if len(self.signal_cache[id]) < self.cache_size:
                return 0

            # 获取当前id信号机轨迹信息，对轨迹进行拟合，进而做出判断
            cache = np.array(self.signal_cache[id])
            x = cache[:, 0]
            y = cache[:, 1]
            p = np.polyfit(x, y, 1)

            if p[0] > 0:
                # print(p[0])
                return id

        return 0
