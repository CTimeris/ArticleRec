import collections
import os
import pickle
import numpy as np
from tqdm import tqdm
from dataset import get_all_click_sample, data_path, get_item_info_df, get_item_emb_dict, save_path, get_all_click_df
from utils import get_item_info_dict, get_hist_and_last_click, get_user_item_time, get_item_topk_click, metrics_recall

# 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
# metric_recall = True
metric_recall = False

# 采样数据
# all_click_df = get_all_click_sample(data_path)
# print(all_click_df)

# 全量数据
all_click_df = get_all_click_df(data_path, offline=False)

# 对时间戳归一化,用于在关联规则的时候计算权重
max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)
# print(all_click_df)

# 文章的属性，数据集中的文章内容embedding
item_info_df = get_item_info_df(data_path)
item_emb_dict = get_item_emb_dict(data_path)

# 获取文章的属性信息，保存成字典的形式方便查询
item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)
# print(item_type_dict)

# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict = {'itemcf_recall': {},
                          'usercf_recall': {},
                          'embedding_sim_item_recall': {},
                          'youtubednn_recall': {}}


def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}

    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]

        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))

        return norm_sorted_item_list

    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]

        for user_id, sorted_item_list in user_recall_items.items():  # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score

    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict_rank, open(os.path.join(save_path, 'final_recall_items_dict.pkl'), 'wb'))

    return final_recall_items_dict_rank


# 根据前面召回的情况来调整参数的值
weight_dict = {'itemcf_recall': 1.0,
               'usercf_recall': 1.0,
               'item2vec_recall': 1.0,
               'youtubednn_recall': 1.0}

# 需要运行每个召回通道，或者读取每个pkl文件（超级大）获得结果
# embedding_sim()
# ItemCF()
# UserCF()
# YoutubeDNN()
user_multi_recall_dict['itemcf_recall'] = pickle.load(open(save_path + 'itemcf_recall_dict.pkl', 'rb'))
user_multi_recall_dict['usercf_recall'] = pickle.load(open(save_path + 'usercf_recall_dict.pkl', 'rb'))
user_multi_recall_dict['item2vec_recall'] = pickle.load(open(save_path + 'embedding_sim_item_recall.pkl', 'rb'))
user_multi_recall_dict['youtubednn_recall'] = pickle.load(open(save_path + 'youtubednn_recall_dict.pkl', 'rb'))


# 最终合并之后每个用户召回150个商品进行排序
final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, topk=150)
