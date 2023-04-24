import numpy as np


class MultiSignalFilter(object):
    def __init__(self):
        self.signal_cache = {}  # 存储各id信号机轨迹的字典
        self.cache_size = 80  # 信号机连续轨迹长度

    def update(self, bbox, identities=None):
        current_id = []  # 当前帧包含信号机的id

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
                del (self.signal_cache[id][0])
            position = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            self.signal_cache[id].append(position)

        # 对当前帧包含信号机进行筛选过滤
        target_signal_id = self.filter(current_id)

        return target_signal_id

    def filter(self, current_id):
        # 获取当前帧id信号机轨迹信息，对轨迹进行拟合
        fiting_result = []
        for id in current_id:
            # 轨迹长度不足，无法进行筛选
            if len(self.signal_cache[id]) < self.cache_size:
                return 0
            cache = np.array(self.signal_cache[id])
            x = cache[:, 0]
            y = cache[:, 1]
            p = np.polyfit(x, y, 1)
            fiting_result.append(p)

        # 对当前帧id信号机轨迹信息拟合结果进行筛选
        lowest_i = 0
        for i, id in enumerate(current_id):
            if fiting_result[i][0] > 0:
                print(fiting_result[i][0])
                continue
            # if self.compare(current_id[i], current_id[lowest_i], fiting_result[i], fiting_result[lowest_i]) and i != 0:
            if self.compare(current_id[i], current_id[lowest_i]) and i != 0:
                lowest_i = i

        return current_id[lowest_i]

    # def compare(self, id1, id2, p1, p2):
    #     # 对两个id的信号机轨迹拟合直线l1、l2进行比较，得到分界线lm
    #     # l1: y = p10 * x + p11
    #     # l2: y = p20 * x + p21
    #     # lm: y = (p10+p20)/2 * (x-xc) + yc
    #     xc = (p2[1] - p1[1]) / (p1[0] - p2[0])  # xc,yc为两条拟合直线的交点坐标
    #     yc = p1[0] * xc + p1[1]
    #
    #     # 获取原轨迹的中点坐标
    #     m1 = self.signal_cache[id1][int(self.cache_size / 2)]
    #     m2 = self.signal_cache[id2][int(self.cache_size / 2)]
    #
    #     # 计算原轨迹中点与分割线lm的相对位置关系
    #     d1 = (p1[0] + p2[0]) * (m1[0] - xc) / 2 + yc - m1[1]
    #     d2 = (p1[0] + p2[0]) * (m2[0] - xc) / 2 + yc - m2[1]
    #
    #     # 若l1的轨迹坐标在l2的轨迹坐标下方，返回0；否则，返回1
    #     # 注意：由于轨迹坐标与实际位置关于y=height/2对称，轨迹坐标在下方意味着实际位置在上方
    #     if d1 > 0 and d2 < 0:
    #         return 0
    #     else:
    #         return 1

    def compare(self, id1, id2):
        # 分别获取两个id的信号机轨迹l1、l2的各两个端点ep1,ep2,ep3,ep4
        ep1 = self.signal_cache[id1][0]
        ep2 = self.signal_cache[id1][-1]
        ep3 = self.signal_cache[id2][0]
        ep4 = self.signal_cache[id2][-1]

        # 获取原轨迹的中点坐标
        m1 = self.signal_cache[id1][int(self.cache_size / 2)]
        m2 = self.signal_cache[id2][int(self.cache_size / 2)]

        # 计算ep1和ep3的中点c1,，ep2和ep4的中点c2
        # 得到两个id的信号机轨迹l1、l2的分割直线lm: y = (c2[1] - c1[1]) / (c2[0] - c1[0]) * (x - c1[0]) + c1[1]
        # 计算原轨迹中点与分割线lm的相对位置关系
        c1 = [(ep1[0] + ep3[0]) / 2, (ep1[1] + ep3[1]) / 2]
        c2 = [(ep2[0] + ep4[0]) / 2, (ep2[1] + ep4[1]) / 2]
        d1 = (c2[1] - c1[1]) / (c2[0] - c1[0]) * (m1[0] - c1[0]) + c1[1] - m1[1]
        d2 = (c2[1] - c1[1]) / (c2[0] - c1[0]) * (m2[0] - c1[0]) + c1[1] - m2[1]

        # 若l1的轨迹坐标在l2的轨迹坐标下方，返回0；否则，返回1
        # 注意：由于轨迹坐标与实际位置关于y=height/2对称，轨迹坐标在下方意味着实际位置在上方
        if d1 > 0 and d2 < 0:
            return 0
        else:
            return 1
