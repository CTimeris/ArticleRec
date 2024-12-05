import collections
import math
import pickle
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from dataset import save_path
from multi_recall import all_click_df, item_created_time_dict, metric_recall, user_multi_recall_dict
from utils import get_user_item_time, get_hist_and_last_click, get_item_topk_click, metrics_recall


def itemcf_sim(df, item_created_time_dict):
    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    # 遍历每个用户的交互历史
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素，对于：位置、物品id、物品点击时间
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1        # 统计物品i的点击次数
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if (i == j):
                    continue

                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(
                    len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    # 将得到的相似性矩阵保存到本地
    # 字典：物品：{物品：相似度}
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

    return i2i_sim_


# 基于商品的召回i2i
def i2icf(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click,
                         item_created_time_dict, emb_i2i_sim):
    """
            基于文章协同过滤的召回
            :param user_id: 用户id
            :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
            :param i2i_sim: 字典，文章相似性矩阵
            :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
            :param recall_item_num: 整数， 最后的召回文章数量
            :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
            :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

            return: 召回的文章列表 [(item1, score1), (item2, score2)...]
        """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}

    item_rank = {}
    # 遍历交互历史
    for loc, (i, click_time) in enumerate(user_hist_items):
        # 按相似度排序，遍历top k
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue

            # 文章创建时间差权重
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            # 相似文章和历史点击文章序列中历史文章所在的位置权重
            loc_weight = (0.9 ** (len(user_hist_items) - loc))

            content_weight = 1.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]

            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank


# ItemCF部分
if __name__ == '__main__':
    # 相似度计算
    i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)

    # 先进行itemcf召回, 为了召回评估，提取最后一次点击
    if metric_recall:
        trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    else:
        trn_hist_click_df = all_click_df

    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(trn_hist_click_df)

    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
    emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))
    # emb_i2v_sim = pickle.load(open(save_path + 'emb_i2v_sim.pkl', 'rb'))

    sim_item_topk = 20
    recall_item_num = 10
    item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

    for user in tqdm(trn_hist_click_df['user_id'].unique()):
        user_recall_items_dict[user] = i2icf(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                             recall_item_num,item_topk_click, item_created_time_dict,
                                             emb_i2i_sim)  # 修改i2i(itemcf)或者i2v(item2vec)

    user_multi_recall_dict['itemcf_recall'] = user_recall_items_dict
    pickle.dump(user_multi_recall_dict['itemcf_recall'], open(save_path + 'itemcf_recall_dict.pkl', 'wb'))

    if metric_recall:
        # 召回效果评估
        metrics_recall(user_multi_recall_dict['itemcf_recall'], trn_last_click_df, topk=recall_item_num)

