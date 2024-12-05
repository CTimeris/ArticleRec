import os

import pandas as pd

from dataset import data_path, get_trn_val_tst_data, save_path, reduce_mem
from feature_process import create_feature, active_level, hot_level, device_fea, user_time_hob_fea, user_cat_hob_fea
from recall_process import get_recall_list, recall_dict_2_df, get_user_recall_item_label_df, make_tuple_func, \
    get_article_info_df, get_embedding
from utils import get_hist_and_last_click


if __name__ == '__main__':
    click_trn, click_val, click_tst, val_ans = get_trn_val_tst_data(data_path, offline=False)
    click_trn_hist, click_trn_last = get_hist_and_last_click(click_trn)

    if click_val is not None:
        click_val_hist, click_val_last = click_val, val_ans
    else:
        click_val_hist, click_val_last = None, None

    click_tst_hist = click_tst
    """
    通过召回将数据转换成三元组的形式（user1, item1, label）的形式，观察发现正负样本差距极度不平衡，
    可以先对负样本进行下采样，下采样的目的一方面缓解了正负样本比例的问题，另一方面也减小了做排序特征的压力
    只对负样本进行下采样，负采样之后，保证所有的用户和文章仍然出现在采样之后的数据中
    下采样的比例可以根据实际情况人为的控制，做完负采样之后，更新此时新的用户召回文章列表，
    因为后续做特征的时候可能用到相对位置的信息。负采样也可以留在做完特征进行。
    """
    # 读取召回列表
    recall_list_dict = get_recall_list(save_path, single_recall_model='i2i_itemcf')  # 这里只选择了单路召回的结果，也可以选择多路召回结果
    # 将召回数据转换成df
    recall_list_df = recall_dict_2_df(recall_list_dict)
    # 给训练验证数据打标签，并负采样
    trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df = (
        get_user_recall_item_label_df(click_trn_hist,
                                      click_val_hist,
                                      click_tst_hist,
                                      click_trn_last,
                                      click_val_last,
                                      recall_list_df,
                                      click_val))

    trn_user_item_label_tuples = trn_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
    trn_user_item_label_tuples_dict = dict(zip(trn_user_item_label_tuples['user_id'], trn_user_item_label_tuples[0]))

    if val_user_item_label_df is not None:
        val_user_item_label_tuples = val_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
        val_user_item_label_tuples_dict = dict(zip(val_user_item_label_tuples['user_id'], val_user_item_label_tuples[0]))
    else:
        val_user_item_label_tuples_dict = None

    tst_user_item_label_tuples = tst_user_item_label_df.groupby('user_id').apply(make_tuple_func).reset_index()
    tst_user_item_label_tuples_dict = dict(zip(tst_user_item_label_tuples['user_id'], tst_user_item_label_tuples[0]))

    # *************************************************************************************
    # 特征工程部分
    article_info_df = get_article_info_df()
    all_click = click_trn.append(click_tst)
    item_content_emb_dict, item_w2v_emb_dict = get_embedding(save_path, all_click)
    # 获取训练验证及测试数据中召回列文章相关特征
    trn_user_item_feats_df = create_feature(trn_user_item_label_tuples_dict.keys(), trn_user_item_label_tuples_dict, \
                                        click_trn_hist, article_info_df, item_content_emb_dict)

    if val_user_item_label_tuples_dict is not None:
        val_user_item_feats_df = create_feature(val_user_item_label_tuples_dict.keys(), val_user_item_label_tuples_dict, \
                                            click_val_hist, article_info_df, item_content_emb_dict)
    else:
        val_user_item_feats_df = None

    # 根据召回物品对其和历史交互中的物品关系做特征
    tst_user_item_feats_df = create_feature(tst_user_item_label_tuples_dict.keys(), tst_user_item_label_tuples_dict, \
                                        click_tst_hist, article_info_df, item_content_emb_dict)
    """
    已有的特征和可构造特征：
    文章自身的特征：文章字数，文章创建时间，文章的embedding
    用户点击环境特征：设备的特征
    对于用户和商品还可以构造的特征：
    基于用户的点击文章次数和点击时间构造可以表现用户活跃度的特征
    基于文章被点击次数和时间构造可以反映文章热度的特征
    用户的时间统计特征：根据其点击的历史文章列表的点击时间和文章的创建时间做统计特征，比如求均值，可以反映用户对于文章时效的偏好
    用户的主题爱好特征：对于用户点击的历史文章主题进行一个统计， 然后对于当前文章看看是否属于用户已经点击过的主题
    用户的字数爱好特征：对于用户点击的历史文章的字数统计， 求一个均值
    """
    # 读取文章特征
    articles = pd.read_csv(data_path + 'articles.csv')
    articles = reduce_mem(articles)

    # 日志数据，就是前面的所有数据
    if click_val is not None:
        all_data = click_trn.append(click_val)
    all_data = click_trn.append(click_tst)
    all_data = reduce_mem(all_data)
    # 拼上文章信息
    all_data = all_data.merge(articles, left_on='click_article_id', right_on='article_id')

    # 活跃用户
    user_act_fea = active_level(all_data, ['user_id', 'click_article_id', 'click_timestamp'])
    # 热门文章
    article_hot_fea = hot_level(all_data, ['user_id', 'click_article_id', 'click_timestamp'])

    # 基于原来的日志表做一个DataFrame，存放用户特有的信息，主要包括点击习惯，爱好特征
    # 设备
    device_cols = ['user_id', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']
    user_device_info = device_fea(all_data, device_cols)
    # 时间
    user_time_hob_cols = ['user_id', 'click_timestamp', 'created_at_ts']
    user_time_hob_info = user_time_hob_fea(all_data, user_time_hob_cols)
    # 主题
    user_category_hob_cols = ['user_id', 'category_id']
    user_cat_hob_info = user_cat_hob_fea(all_data, user_category_hob_cols)
    # 字数
    user_wcou_info = all_data.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_wcou_info.rename(columns={'words_count': 'words_hbo'}, inplace=True)
    # 所有表进行合并
    user_info = pd.merge(user_act_fea, user_device_info, on='user_id')
    user_info = user_info.merge(user_time_hob_info, on='user_id')
    user_info = user_info.merge(user_cat_hob_info, on='user_id')
    user_info = user_info.merge(user_wcou_info, on='user_id')
    user_info.to_csv(save_path + 'user_info.csv', index=False)

    # 把用户信息直接读入进来
    user_info = pd.read_csv(save_path + 'user_info.csv')

    if os.path.exists(save_path + 'trn_user_item_feats_df.csv'):
        trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv')

    if os.path.exists(save_path + 'tst_user_item_feats_df.csv'):
        tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df.csv')

    if os.path.exists(save_path + 'val_user_item_feats_df.csv'):
        val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
    else:
        val_user_item_feats_df = None

    # 拼上用户特征
    # 下面是线下验证的
    trn_user_item_feats_df = trn_user_item_feats_df.merge(user_info, on='user_id', how='left')

    if val_user_item_feats_df is not None:
        val_user_item_feats_df = val_user_item_feats_df.merge(user_info, on='user_id', how='left')
    else:
        val_user_item_feats_df = None

    tst_user_item_feats_df = tst_user_item_feats_df.merge(user_info, on='user_id', how='left')

    articles = pd.read_csv(data_path + 'articles.csv')
    articles = reduce_mem(articles)

    # 拼上文章特征
    trn_user_item_feats_df = trn_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')

    if val_user_item_feats_df is not None:
        val_user_item_feats_df = val_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')
    else:
        val_user_item_feats_df = None

    tst_user_item_feats_df = tst_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')

    # 召回文章的主题是否在用户的爱好里面
    trn_user_item_feats_df['is_cat_hab'] = trn_user_item_feats_df.apply(lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)
    if val_user_item_feats_df is not None:
        val_user_item_feats_df['is_cat_hab'] = val_user_item_feats_df.apply(lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)
    else:
        val_user_item_feats_df = None
    tst_user_item_feats_df['is_cat_hab'] = tst_user_item_feats_df.apply(lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1)

    # 验证
    del trn_user_item_feats_df['cate_list']

    if val_user_item_feats_df is not None:
        del val_user_item_feats_df['cate_list']
    else:
        val_user_item_feats_df = None

    del tst_user_item_feats_df['cate_list']

    del trn_user_item_feats_df['article_id']

    if val_user_item_feats_df is not None:
        del val_user_item_feats_df['article_id']
    else:
        val_user_item_feats_df = None

    del tst_user_item_feats_df['article_id']

    # 保存特征
    trn_user_item_feats_df.to_csv(save_path + 'trn_user_item_feats_df.csv', index=False)
    if val_user_item_feats_df is not None:
        val_user_item_feats_df.to_csv(save_path + 'val_user_item_feats_df.csv', index=False)
    tst_user_item_feats_df.to_csv(save_path + 'tst_user_item_feats_df.csv', index=False)

