import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
from dataset import save_path

from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# 排序结果
def rank_res(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    res = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    res.columns = [int(col) if isinstance(col, int) else col for col in res.columns.droplevel(0)]
    # 定义列名
    res = res.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    res.to_csv(save_name, index=False, header=True)


# 排序结果归一化
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


# 五折交叉验证，以用户为目标进行五折划分，这一部分与前面的单独训练和验证是分开的
def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


def LGBMRanker_rank(trn_user_item_feats_df_rank_model, val_user_item_feats_df_rank_model,
                    trn_user_item_feats_df, tst_user_item_feats_df, tst_user_item_feats_df_rank_model, offline=False):
    # LGBMRanker
    # 定义特征列
    lgb_cols = ['sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min', 'sim_sum',
                'sim_mean', 'score', 'click_size', 'time_diff_mean', 'active_level',
                'click_environment', 'click_deviceGroup', 'click_os', 'click_country',
                'click_region', 'click_referrer_type', 'user_time_hob1', 'user_time_hob2',
                'words_hbo', 'category_id', 'created_at_ts', 'words_count']

    # 排序模型分组
    trn_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
    g_train = trn_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

    if offline:
        val_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
        g_val = val_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

    # 排序模型定义
    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16)

    # 排序模型训练
    if offline:
        lgb_ranker.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'],
                       group=g_train,
                       eval_set=[
                           (val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])],
                       eval_group=[g_val], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, )
    else:
        lgb_ranker.fit(trn_user_item_feats_df[lgb_cols], trn_user_item_feats_df['label'], group=g_train)

    # 模型预测
    tst_user_item_feats_df['pred_score'] = lgb_ranker.predict(tst_user_item_feats_df[lgb_cols],
                                                              num_iteration=lgb_ranker.best_iteration_)

    # 将这里的排序结果保存一份，用户后面的模型融合
    tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_ranker_score.csv',
                                                                                 index=False)

    # 预测结果重新排序, 及生成结果
    rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
    rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
    rank_res(rank_results, topk=5, model_name='lgb_ranker')

    # 交叉验证
    k_fold = 5
    trn_df = trn_user_item_feats_df_rank_model
    user_set = get_kfold_users(trn_df, n=k_fold)

    score_list = []
    score_df = trn_df[['user_id', 'click_article_id', 'label']]
    sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

    # 五折交叉验证，并将中间结果保存用于staking
    for n_fold, valid_user in enumerate(user_set):
        train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user
        valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]

        # 训练集与验证集的用户分组
        train_idx.sort_values(by=['user_id'], inplace=True)
        g_train = train_idx.groupby(['user_id'], as_index=False).count()["label"].values

        valid_idx.sort_values(by=['user_id'], inplace=True)
        g_val = valid_idx.groupby(['user_id'], as_index=False).count()["label"].values

        # 定义模型
        lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                    max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7,
                                    subsample_freq=1,
                                    learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16)
        # 训练模型
        lgb_ranker.fit(train_idx[lgb_cols], train_idx['label'], group=g_train,
                       eval_set=[(valid_idx[lgb_cols], valid_idx['label'])], eval_group=[g_val],
                       eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, )

        # 预测验证集结果
        valid_idx['pred_score'] = lgb_ranker.predict(valid_idx[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

        # 对输出结果进行归一化
        valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

        valid_idx.sort_values(by=['user_id', 'pred_score'])
        valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

        # 将验证集的预测结果放到一个列表中，后面进行拼接
        score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

        # 如果线上测试，需要计算每次交叉验证的结果相加，最后求平均
        if not offline:
            sub_preds += lgb_ranker.predict(tst_user_item_feats_df_rank_model[lgb_cols], lgb_ranker.best_iteration_)

        score_df_ = pd.concat(score_list, axis=0)
        score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
        # 保存训练集交叉验证产生的新特征
        score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
            save_path + 'trn_lgb_ranker_feats.csv', index=False)

        # 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
        tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
        tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(
            lambda x: norm_sim(x))
        tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
        tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])[
            'pred_score'].rank(ascending=False, method='first')

        # 保存测试集交叉验证的新特征
        tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
            save_path + 'tst_lgb_ranker_feats.csv', index=False)

        # 预测结果重新排序, 及生成结果
        rank_results = tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score']]
        rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
        rank_res(rank_results, topk=5, model_name='lgb_ranker')


def LGBMClass(trn_user_item_feats_df_rank_model, val_user_item_feats_df_rank_model,
                    trn_user_item_feats_df, tst_user_item_feats_df, tst_user_item_feats_df_rank_model, offline=False):
    # 定义特征列
    lgb_cols = ['sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min', 'sim_sum',
                'sim_mean', 'score', 'click_size', 'time_diff_mean', 'active_level',
                'click_environment', 'click_deviceGroup', 'click_os', 'click_country',
                'click_region', 'click_referrer_type', 'user_time_hob1', 'user_time_hob2',
                'words_hbo', 'category_id', 'created_at_ts', 'words_count']

    # 模型及参数的定义
    lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=500, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs= 16, verbose=10)

    if offline:
        lgb_Classfication.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'],
                              eval_set=[(val_user_item_feats_df_rank_model[lgb_cols],
                                         val_user_item_feats_df_rank_model['label'])],
                              eval_metric=['auc', ], early_stopping_rounds=50, )
    else:
        lgb_Classfication.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'])

    # 模型预测
    tst_user_item_feats_df['pred_score'] = lgb_Classfication.predict_proba(tst_user_item_feats_df[lgb_cols])[:,1]

    # 将这里的排序结果保存一份，用于后面的模型融合
    tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_cls_score.csv', index=False)

    # 预测结果重新排序, 及生成结果
    rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
    rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
    rank_res(rank_results, topk=5, model_name='lgb_cls')

    #  这一部分与前面的单独训练和验证是分开的
    k_fold = 5
    trn_df = trn_user_item_feats_df_rank_model
    user_set = get_kfold_users(trn_df, n=k_fold)

    score_list = []
    score_df = trn_df[['user_id', 'click_article_id', 'label']]
    sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

    # 五折交叉验证，并将中间结果保存用于staking
    for n_fold, valid_user in enumerate(user_set):
        train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user
        valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]

        # 模型及参数的定义
        lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                               max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7,
                                               subsample_freq=1,
                                               learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16,
                                               verbose=10)
        # 训练模型
        lgb_Classfication.fit(train_idx[lgb_cols], train_idx['label'],
                              eval_set=[(valid_idx[lgb_cols], valid_idx['label'])],
                              eval_metric=['auc', ], early_stopping_rounds=50, )

        # 预测验证集结果
        valid_idx['pred_score'] = lgb_Classfication.predict_proba(valid_idx[lgb_cols],
                                                                  num_iteration=lgb_Classfication.best_iteration_)[:, 1]
        # 对输出结果进行归一化 分类模型输出的值本身就是一个概率值不需要进行归一化
        # valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

        valid_idx.sort_values(by=['user_id', 'pred_score'])
        valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

        # 将验证集的预测结果放到一个列表中，后面进行拼接
        score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

        # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
        if not offline:
            sub_preds += lgb_Classfication.predict_proba(tst_user_item_feats_df_rank_model[lgb_cols],
                                                         num_iteration=lgb_Classfication.best_iteration_)[:, 1]

    score_df_ = pd.concat(score_list, axis=0)
    score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
    # 保存训练集交叉验证产生的新特征
    score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
        save_path + 'trn_lgb_cls_feats.csv', index=False)

    # 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
    tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
    tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(
        lambda x: norm_sim(x))
    tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
    tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])[
        'pred_score'].rank(ascending=False, method='first')

    # 保存测试集交叉验证的新特征
    tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
        save_path + 'tst_lgb_cls_feats.csv', index=False)

    # 预测结果重新排序, 及生成提交结果
    rank_results = tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score']]
    rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
    rank_res(rank_results, topk=5, model_name='lgb_cls')


# DIN数据准备函数
def get_din_feats_columns(df, dense_fea, sparse_fea, behavior_fea, his_behavior_fea, emb_dim=32, max_len=100):
    """
    数据准备函数:
    df: 数据集
    dense_fea: 数值型特征列
    sparse_fea: 离散型特征列
    behavior_fea: 用户的候选行为特征列
    his_behavior_fea: 用户的历史行为特征列
    embedding_dim: embedding的维度， 这里为了简单， 统一把离散型特征列采用一样的隐向量维度
    max_len: 用户序列的最大长度
    """

    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique() + 1, embedding_dim=emb_dim) for feat
                              in sparse_fea]

    dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_fea]

    var_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=df['click_article_id'].nunique() + 1,
                                                       embedding_dim=emb_dim, embedding_name='click_article_id'),
                                            maxlen=max_len) for feat in his_behavior_fea]

    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns

    # 建立x, x是一个字典的形式
    x = {}
    for name in get_feature_names(dnn_feature_columns):
        if name in his_behavior_fea:
            # 这是历史行为序列
            his_list = [l for l in df[name]]
            x[name] = pad_sequences(his_list, maxlen=max_len, padding='post')  # 二维数组
        else:
            x[name] = df[name].values

    return x, dnn_feature_columns




