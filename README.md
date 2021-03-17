# code_tempimport numpy as np
import pandas as pd
from scipy.stats import pearsonr


'''
功能：提取时间类型字段（离散时间）特征
输入：dataframe，包含一列，为待提取特征的时间列
输出：dataframe，待提取特征的时间列，年，月，日，时，分，秒，一天中的第几分钟，星期几，一年中的第几天，一年中的第几个星期，一年中的第几周，一年中的哪个季度
作者：冯帆
日期：20210317
'''


def dispersed_time_feature(input_data):
    try:
        df = input_data.applymap(lambda x: pd.Timestamp(x))
        # 年份
        df['year'] = df.apply(lambda x: x.year)
        # 月份
        df['month'] = df.apply(lambda x: x.month)
        # 日
        df['day'] = df.apply(lambda x: x.day)
        # 小时
        df['hour'] = df.apply(lambda x: x.hour)
        # 分钟
        df['minute'] = df.apply(lambda x: x.minute)
        # 秒数
        df['second'] = df.apply(lambda x: x.second)
        # 一天中的第几分钟
        df['minuteofday'] = df.apply(lambda x: x.minute + x.hour * 60)
        # 星期几；
        df['dayofweek'] = df.apply(lambda x: x.dayofweek)
        # 一年中的第几天
        df['dayofyear'] = df.apply(lambda x: x.dayofyear)
        # 一年中的第几周
        df['week'] = df.apply(lambda x: x.week)
        # 一年中的哪个季度
        season_dict = {
            1: 1, 2: 1, 3: 1,
            4: 2, 5: 2, 6: 2,
            7: 3, 8: 3, 9: 3,
            10: 4, 11: 4, 12: 4,
        }
        df['season'] = df['month'].map(season_dict)
        return df
    except ValueError:
        print('请确保输入input_data，并且其类型为dataframe')
        return 0


if __name__ == '__main__':
    data = ['2019-01-01 01:22:26', '2019-02-02 04:34:52', '2019-03-03 06:16:40',
    '2019-04-04 08:11:38', '2019-05-05 10:52:39', '2019-06-06 12:06:25',
    '2019-07-07 14:05:25', '2019-08-08 16:51:33', '2019-09-09 18:28:28',
    '2019-10-10 20:55:12', '2019-11-11 22:55:12', '2019-12-12 00:55:12']
    df = pd.DataFrame({'时间': data})
    print(df)
    dispersed_time_feature(input_data=df)
    
    
    
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

'''
功能：用皮尔逊相关系数对特征进行筛选
输入：特征列input_feature，标签列input_label（两者类型都是DataFrame）
      和筛选特征数目num（小于输入的特征列列数，默认值为特征列列数）、
      显著性r的临界值r_crisis（默认值为0.1）以及皮尔逊系数临界值p_crisis（默认值为0.5）
输出：满足显著性r的临界值和皮尔逊相关系数临界值且小于等于num的特征列列名，类型List
作者：冯帆
日期：20210316
'''


def feature_pearson_select(input_feature, input_label, num=None, r_crisis=0.1, p_crisis=0.5):
    try:
        all_num = input_feature.shape[1]
        if num is None:
            num = all_num
        else:
            pass
        if input_feature.shape[0] != input_label.shape[0]:
            print('错误：请确保输入的特征列和标签列的行数一致（样本数和标签数相同，可以一一对应）')
            return 0
        else:
            data = pd.concat([input_feature, input_label], axis=1)
        # 获取各个特征与标签列的皮尔逊相关系数
        corr = data.corr(method='pearson', min_periods=0.03)
        # 对获取的皮尔逊相关系数矩阵取绝对值
        corr = corr.abs()
        # 获取每个特征和标签值的皮尔逊相关系数值
        corr = pd.DataFrame(corr.iloc[:-1, -1])
        corr.columns = ['a']
        # 过滤掉不满足皮尔逊相关系数条件的特征列，并对特征列根据皮尔逊系数值进行倒序排序
        corr = corr[corr['a'] >= p_crisis].sort_values(by=["a"], axis=0, ascending=False)
        # num_temp用来记录当前满足条件的特征列数量
        num_temp = corr.shape[0]
        # result用来记录最后满足条件的特征列
        result = []
        # 对满足前面条件的特征列进行显著性检验
        for i in range(0, num):
            if i >= num_temp:
                pass
            else:
                (p, r) = pearsonr(np.squeeze(np.array(input_feature[corr.index[i]], dtype='float64')),
                                  np.squeeze(np.array(input_label, dtype='float64')))
                if r < r_crisis:
                    result.append(corr.index[i])
                else:
                    pass
    except ValueError:
        print('请确保已输入特征列和标签列')
        return 0

    return result

