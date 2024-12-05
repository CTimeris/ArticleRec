import collections
import pickle
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from ItemCF import i2icf
from dataset import data_path, save_path
from multi_recall import all_click_df, metric_recall, user_multi_recall_dict, item_created_time_dict
from utils import get_hist_and_last_click, get_user_item_time, get_item_topk_click, metrics_recall


# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def emb_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵

        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """

    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                 0) + sim_value

    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))

    return item_sim_dict


# 基于item2vec的embedding的相似度计算
def emb_sim_i2v(click_df, item_emb_dic, save_path, topk):
    """
        基于item2vec的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵

        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """
    emb_list = []
    df = pd.DataFrame(columns=['article_id', 'embedding'])
    for k, v in item_emb_dic.items():
        row_list = []
        for i in v:
            row_list.append(i)
        emb_list.append(row_list)
        new_row = pd.DataFrame({'article_id': k, 'embedding': row_list})
        df = pd.concat([df, new_row])

    item_emb_np = np.array(emb_list)
    # print(len(item_emb_np))
    # print(item_emb_np[0])
    # 建立faiss索引
    d = len(item_emb_np[0])
    item_index = faiss.IndexFlatIP(d)
    # print(item_emb_np.ndim)
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = df.iloc[target_idx]['article_id']
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = df.iloc[rele_idx]['article_id']
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                 0) + sim_value

    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2v_sim.pkl', 'wb'))

    return item_sim_dict


# 用数据集原始embedding计算相似度
# item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')

# 利用item2vec的embedding计算相似度
item_emb_dic = pickle.load(open(save_path + 'item_w2v_emb.pkl', 'rb'))

# emb_i2i_sim = emb_sim(all_click_df, item_emb_df, save_path, topk=10)
emb_i2v_sim = emb_sim_i2v(all_click_df, item_emb_dic, save_path, topk=10)


if __name__ == '__main__':

    # 这里是为了召回评估，所以提取最后一次点击
    if metric_recall:
        trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    else:
        trn_hist_click_df = all_click_df

    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(trn_hist_click_df)
    i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

    sim_item_topk = 20
    recall_item_num = 10

    item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

    for user in tqdm(trn_hist_click_df['user_id'].unique()):
        user_recall_items_dict[user] = i2icf(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                                            recall_item_num, item_topk_click, item_created_time_dict,
                                                            emb_i2v_sim)

    pickle.dump(user_multi_recall_dict['embedding_sim_item_recall'],
                open(save_path + 'embedding_sim_item_recall.pkl', 'wb'))

    if metric_recall:
        # 召回效果评估
        metrics_recall(user_multi_recall_dict['embedding_sim_item_recall'], trn_last_click_df, topk=recall_item_num)

