import os
import numpy as np
import argparse
import configparser


def search_data(
    sequence_length,
    num_of_depend,
    label_start_idx,
    num_for_predict,
    units,
    points_per_hour,
):  
    """
    确定预测所需的历史数据索引范围

    input:
        sequence_length: int, length of all history data                             # 所有历史数据的长度
        num_of_depend: int,                                                          # 需要依赖的历史数据段数据（需要几个周/天/小时的片段）
        label_start_idx: int, the first index of predicting target                   # 预测目标的起始索引
        num_for_predict: int, the number of points will be predicted for each sample # 每个样本要预测的时间点数量 
        units: int, week: 7 * 24, day: 24, recent(hour): 1                           # 时间单位
        points_per_hour: int, number of points per hour, depends on data             # 每个小时数据点数

    return:
        list[(start_idx, end_idx)]                                                   # 历史时间段

    """

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    # 存储索引
    x_idx = []
    # 需要几周/天/小时的片段
    for i in range(1, num_of_depend + 1):
        # 得到起始索引 = 当前时间点 - 时间单位*每个小时数据点数
        start_idx = label_start_idx - points_per_hour * units * i
        # 结束时间点 = 起始索引 + 要预测的时间点数 
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    # 当你从历史数据中提取多个时间段时，最早的时间段通常在列表的开头，最近的时间段在列表的末尾，故需要反转
    return x_idx[::-1]


def get_sample_indices(
    data_sequence,
    num_of_weeks,
    num_of_days,
    num_of_hours,
    label_start_idx,
    num_for_predict,
    points_per_hour=12,
):

    """
    该函数用于从数据序列中提取特定时间范围（周、天、小时）的样本和预测目标，得到当前时间点的（采样索引点），三种时间模式的样本数据集和预测值
    input:
        data_sequence: np.ndarray
                       shape is (sequence_length, num_of_vertices, num_of_features)     # 采样数，节点，特征
    
        num_of_weeks, num_of_days, num_of_hours: int                                    # 需要几个周，天，小时的片段
        label_start_idx: int, the first index of predicting target                      # 预测目标的起始索引
        num_for_predict: int, the number of points will be predicted for each sample    # 每个样本要预测的点数
        points_per_hour: int, default 12, number of points per hour                     # 每个小时的数据点数
    
    returns: 
        week_sample: np.ndarray                                                         # 周的采样点数（2*12）,节点数，特征数
                    shape is (num_of_weeks * points_per_hour,                           # 某周的某个小时内的数据，某+1周的某个小时的数据 
                            num_of_vertices, num_of_features)                           

        day_sample: np.ndarray                                                          # 天的采样点数（2*12）,节点数，特征数
                    shape is (num_of_days * points_per_hour,                            # 某天的某个小时内的数据，某+1天的某个小时的数据
                            num_of_vertices, num_of_features)                           

        hour_sample: np.ndarray                                                         # 小时的采样点数（2*12）,节点数，特征数                           
                    shape is (num_of_hours * points_per_hour,
                            num_of_vertices, num_of_features)

        target: np.ndarray                                                              # 预测值
                shape is (num_for_predict, num_of_vertices, num_of_features)            # 预测的采样点数，节点，特征
    """


    # 初始化
    week_sample, day_sample, hour_sample = None, None, None

    # 检查数据范围
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    # 提取周样本
    if num_of_weeks > 0:
        # 提取周样本的索引，返回索引列表[起始，结束]
        week_indices = search_data(       
            data_sequence.shape[0],
            num_of_weeks,                       # 需要几个周的片段
            label_start_idx,
            num_for_predict,
            7 * 24,
            points_per_hour,
        )
        if not week_indices:
            return None, None, None, None

        # 根据索引，取出样本值
        week_sample = np.concatenate(
            [data_sequence[i:j] for i, j in week_indices], axis=0
        )

    # 提取天样本
    if num_of_days > 0:
        day_indices = search_data(
            data_sequence.shape[0],
            num_of_days,                      # 需要几天的片段
            label_start_idx,
            num_for_predict,
            24,
            points_per_hour,
        )
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate(
            [data_sequence[i:j] for i, j in day_indices], axis=0
        )

    # 提取小时样本
    if num_of_hours > 0:
        hour_indices = search_data(
            data_sequence.shape[0],    # data_sequence.shape[0]:16992 data_sequence.shape: (16992, 307, 3) 
            num_of_hours,
            label_start_idx,
            num_for_predict,
            1,
            points_per_hour,
        )
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate(
            [data_sequence[i:j] for i, j in hour_indices], axis=0
        )

    # 取出目标值，这里就是 按周，日，小时的跨度取值，用三个时间模式来预测下一个小时的值
    # 这里 周片段，日片段，小时片段都是为预测当前时间下一个时间段的值
    target = data_sequence[label_start_idx : label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(
    graph_signal_matrix_filename,
    num_of_weeks,
    num_of_days,
    num_of_hours,
    num_for_predict,
    points_per_hour=12,
    save=False,
):
    """
    该函数用于从包含图信号矩阵的数据文件中读取数据，生成用于训练、验证和测试的样本，并可选择将处理后的数据保存到文件中
    input:
        graph_signal_matrix_filename: str, path of graph signal matrix file  # 图矩阵的文件名
        num_of_weeks, num_of_days, num_of_hours: int                         # 周数，天数，小时数
        num_for_predict: int                                                 # 每个样本需要预测的数据点数
        points_per_hour: int, default 12, depends on data                    # 每个小时的数据点数（默认12值，5分钟采样一次）

    return:
        all: dcit    # 包含训练，验证和测试数据集及其统计信息的字典
    """

    # 数据加载
    # data_seq.shape: (16992, 307, 3) 采样数，节点数，特征维度   # 原始数据集
    data_seq = np.load(graph_signal_matrix_filename)[
        "data"
    ]  # (sequence_length, num_of_vertices, num_of_features)

    # 样本生成
    all_samples = []

    # 遍历每次采样，调用 get_sample_indices 生成样本
    # 就是根据当前的时间点，在数据集中找出不同时间模式的样本（可以根据开始时间，结束时间，和采样数，来得到当前的时间点）
    # 就是根据当前的 采样索引值就可以得到当前的时间点，再根据当前的时间点，生成 样本和目标值
    for idx in range(data_seq.shape[0]):            # data_seq.shape[0]: 16992
        """
        得到当前时间点的（采样索引点），三种时间模式的样本数据集和预测值
        """
        sample = get_sample_indices(
            data_seq,               # 采样数
            num_of_weeks,            # 要几周的片段  
            num_of_days,
            num_of_hours,
            idx,                  # idx 就是当前时间点，预测值的开始索引
            num_for_predict,
            points_per_hour,
        )
        if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        """
        这里的增加维度是用于批次，方便模型后期处理多个样本
        """
        if num_of_weeks > 0:
            # week_sample.shape:(24, 307, 3)  采样数，节点，特征维数
            week_sample = np.expand_dims(week_sample, axis=0).transpose(
                (0, 2, 3, 1)
            ) 
            # 操作后 week_sample.shape:(1, 307, 3, 24) 批次，节点数，特征数，采样数
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose(
                (0, 2, 3, 1)
            )
            sample.append(day_sample)

        if num_of_hours > 0:
            # hour_sample.shape: (24, 307, 3)  采样数，节点，特征维数
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose(
                (0, 2, 3, 1)
            )  
            # 操作后 hour_sample.shape:(1, 307, 3, 24) 批次，节点数，特征数，采样数
            # 这里的采样数，都是针对当前时刻，片段内的采样点的数量!
            sample.append(hour_sample)

        # 预测值（相同的操作）
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[
            :, :, 0, :
        ] 
        # target.shape: (1, 307, 12) 批次，节点数据，特征数
        sample.append(target)

        # 用于标记或识别当前样本在时间序列中的位置
        # 将当前样本的时间索引 idx 转成数组 time_sample.shape: (1, 1)
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        # len(all_samples): 12949
        # len(sample):5
        all_samples.append(
            sample
        )  # sampe：[(week_sample),(day_sample),(hour_sample), target, time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    # 数据集划分 6:2:2
    """
    len(all_samples): 16957
    all_samples 有 16957 个当前时刻（采样点），每个当前时刻都有所有节点三个模式的样本数据和目标值和时间戳！即数据集内容构建完成
    表示第一个时间模式（周）: all_samples[-1][0].shape:(1, 307, 3, 24)   批次，节点数，特征数，时间内采样数
    表示第二个时间模式（天）: all_samples[-1][0].shape:(1, 307, 3, 24)
    表示第三个时间模式（时）: all_samples[-1][0].shape:(1, 307, 3, 24)
    目标值：all_samples[-1][3].shape:(1, 307, 12)
    时间戳：all_samples[-1][4].shape:(1, 1)
    """
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    # 训练集
    training_set = [
        np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])
    ]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    
    # 验证集
    validation_set = [
        np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1:split_line2])
    ]

    # 测试集
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    """
    融合不同时间模式
    training_set.shape: [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    将不同时间点的数据按模式重组并连接， 取三项，沿着最后一个轴（采样数轴）拼接时，数据是按顺序直接拼接
    拼接后：(B, N, F, Tw + Td + Th)  
    """
    """
    这里的 72 是由三个时间模式的采样数（时间步长）相加得来的，每个时间模式表示一段时间内的数据片段
    这里的 24 表示 2 * 12 即 "num_of_weeks/day/hours=2" 每小时采集12次得来的，共周期2次，每次1小时内的片段（各模式下）
    """
    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T') (7769, 307, 3, 72)  24+24+24=72
    val_x = np.concatenate(validation_set[:-2], axis=-1)  # (2590, 307, 3, 72)
    test_x = np.concatenate(testing_set[:-2], axis=-1)    # (2590, 307, 3, 72)  # 这里的批次还没有设置，这里表示总的数据量

    # 最后两项：目标值和时间戳
    train_target = training_set[-2]  # (B,N,T) (7769, 307, 12)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    # 标准化数据集
    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(
        train_x, val_x, test_x
    )

    all_data = {
        "train": {
            "x": train_x_norm,
            "target": train_target,
            "timestamp": train_timestamp,
        },
        "val": {
            "x": val_x_norm,
            "target": val_target,
            "timestamp": val_timestamp,
        },
        "test": {
            "x": test_x_norm,
            "target": test_target,
            "timestamp": test_timestamp,
        },
        "stats": {
            "_mean": stats["_mean"],
            "_std": stats["_std"],
        },
    }
    print("train x:", all_data["train"]["x"].shape)                 # (7769, 307, 3, 72) 批次大小，节点数，特征维度，采样数（时间步长）
    print("train target:", all_data["train"]["target"].shape)       # (7769, 307, 12)    批次大小，节点数，
    print("train timestamp:", all_data["train"]["timestamp"].shape) # (7769, 1)
    print()
    print("val x:", all_data["val"]["x"].shape)
    print("val target:", all_data["val"]["target"].shape)
    print("val timestamp:", all_data["val"]["timestamp"].shape)
    print()
    print("test x:", all_data["test"]["x"].shape)
    print("test target:", all_data["test"]["target"].shape)
    print("test timestamp:", all_data["test"]["timestamp"].shape)
    print()
    print("train data _mean :", stats["_mean"].shape, stats["_mean"])
    print("train data _std :", stats["_std"].shape, stats["_std"])

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split(".")[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = (
            os.path.join(
                dirpath,
                file
                + "_r"
                + str(num_of_hours)
                + "_d"
                + str(num_of_days)
                + "_w"
                + str(num_of_weeks),
            )
            + "_astcgn"
        )
        print("save file:", filename)
        np.savez_compressed(
            filename,
            train_x=all_data["train"]["x"],
            train_target=all_data["train"]["target"],
            train_timestamp=all_data["train"]["timestamp"],
            val_x=all_data["val"]["x"],
            val_target=all_data["val"]["target"],
            val_timestamp=all_data["val"]["timestamp"],
            test_x=all_data["test"]["x"],
            test_target=all_data["test"]["target"],
            test_timestamp=all_data["test"]["timestamp"],
            mean=all_data["stats"]["_mean"],
            std=all_data["stats"]["_std"],
        )
    return all_data


def normalization(train, val, test):
    """
    input:
        train, val, test: np.ndarray (B,N,F,T)
    
    returns:
        stats: dict, two keys: mean and std
        train_norm, val_norm, test_norm: np.ndarray,
                                         shape is the same as original
    """

    assert (
        train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    )  # ensure the num of nodes is the same
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)
    print("mean.shape:", mean.shape)
    print("std.shape:", std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {"_mean": mean, "_std": std}, train_norm, val_norm, test_norm


if __name__ == "__main__":
    # prepare dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configurations/METR_LA_astgcn.conf",
        type=str,
        help="configuration file path",
    )
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print("Read configuration file: %s" % (args.config))
    config.read(args.config)
    data_config = config["Data"]
    training_config = config["Training"]

    adj_filename = data_config["adj_filename"]
    graph_signal_matrix_filename = data_config["graph_signal_matrix_filename"]
    if config.has_option("Data", "id_filename"):
        id_filename = data_config["id_filename"]
    else:
        id_filename = None

    num_of_vertices = int(data_config["num_of_vertices"])
    points_per_hour = int(data_config["points_per_hour"])
    num_for_predict = int(data_config["num_for_predict"])
    len_input = int(data_config["len_input"])
    dataset_name = data_config["dataset_name"]
    num_of_weeks = int(training_config["num_of_weeks"])
    num_of_days = int(training_config["num_of_days"])
    num_of_hours = int(training_config["num_of_hours"])
    num_of_vertices = int(data_config["num_of_vertices"])
    points_per_hour = int(data_config["points_per_hour"])
    num_for_predict = int(data_config["num_for_predict"])    # 12 表示预测下一个小时的值
    graph_signal_matrix_filename = data_config["graph_signal_matrix_filename"]
    data = np.load(graph_signal_matrix_filename)
    data["data"].shape

    all_data = read_and_generate_dataset(
        graph_signal_matrix_filename,   # 就是 npz 文件
        num_of_weeks,
        num_of_days,
        num_of_hours,
        num_for_predict,
        points_per_hour=points_per_hour,
        save=True,
    )
