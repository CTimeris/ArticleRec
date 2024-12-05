import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from deepctr.models import DIN
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
from dataset import save_path, data_path
from rank_models import LGBMRanker_rank, LGBMClass, get_din_feats_columns, rank_res, get_kfold_users, norm_sim

if __name__ == '__main__':

    offline = False

    # 重新读取数据的时候，发现click_article_id是一个浮点数，所以将其转换成int类型
    trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv')
    trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype(int)

    if offline:
        val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
        val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype(int)
    else:
        val_user_item_feats_df = None

    tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df.csv')
    tst_user_item_feats_df['click_article_id'] = tst_user_item_feats_df['click_article_id'].astype(int)

    # 做特征的时候为了方便，给测试集也打上了一个无效的标签，这里直接删掉就行
    del tst_user_item_feats_df['label']

    trn_user_item_feats_df_rank_model = trn_user_item_feats_df.copy()

    if offline:
        val_user_item_feats_df_rank_model = val_user_item_feats_df.copy()

    tst_user_item_feats_df_rank_model = tst_user_item_feats_df.copy()

    # 数据有点大，需要注释其他模型分开跑
    # LGBMRanker
    LGBMRanker_rank(trn_user_item_feats_df_rank_model, val_user_item_feats_df_rank_model,
                    trn_user_item_feats_df, tst_user_item_feats_df, tst_user_item_feats_df_rank_model, offline)

    # LGBMClassifier
    LGBMClass(trn_user_item_feats_df_rank_model, val_user_item_feats_df_rank_model,
              trn_user_item_feats_df, tst_user_item_feats_df, tst_user_item_feats_df_rank_model, offline)

    # DIN模型
    """
    def DIN(dnn_feature_columns, history_feature_list, dnn_use_bn=False, dnn_hidden_units=(200, 80), 
    dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice", att_weight_normalization=False, 
    l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, task='binary'):

    DIN的参数：
    dnn_feature_columns: 特征列， 包含数据所有特征的列表
    history_feature_list: 用户历史行为列， 反应用户历史行为的特征的列表
    dnn_use_bn: 是否使用BatchNormalization
    dnn_hidden_units: 全连接层网络的层数和每一层神经元的个数， 一个列表或者元组
    dnn_activation_relu: 全连接网络的激活单元类型
    att_hidden_size: 注意力层的全连接网络的层数和每一层神经元的个数
    att_activation: 注意力层的激活单元类型
    att_weight_normalization: 是否归一化注意力得分
    l2_reg_dnn: 全连接网络的正则化系数
    l2_reg_embedding: embedding向量的正则化稀疏
    dnn_dropout: 全连接网络的神经元的失活概率
    task: 任务， 可以是分类， 也可是是回归

    在具体使用的时候，必须要传入特征列和历史行为列，但是再传入之前，要进行一下特征列的预处理。具体如下：
    1、处理数据集，得到数据，由于是基于用户过去的行为去预测用户是否点击当前文章，
    所以需要把数据的特征列划分成：数值型特征，离散型特征 和 历史行为特征列三部分，对于每一部分，DIN模型的处理会有不同。

    2、对于离散型特征，在数据集中就是那些类别型的特征，比如user_id，这种类别型特征，首先要经过embedding处理得到每个特征的低维稠密型表示， 
    既然要经过embedding，那么就需要为每一列的类别特征的取值建立一个字典，并指明embedding维度，所以在使用deepctr的DIN模型准备数据的时候，
    需要通过SparseFeat函数指明这些类别型特征，这个函数的传入参数就是列名，列的唯一取值(建立字典用)和embedding维度。

    3、对于用户历史行为特征列，比如文章id，文章的类别等这种，同样的需要先经过embedding处理，只不过和上面不一样的地方是，对于这种特征，
    在得到每个特征的embedding表示之后，还需要通过一个Attention_layer计算用户的历史行为和当前候选文章的相关性以此得到当前用户的embedding向量，
    这个向量就可以基于当前的候选文章与用户过去点击过的历史文章的相似性的程度来反应用户的兴趣，并且随着用户的不同的历史点击来变化，去动态的模拟用户兴趣的变化过程。

    这类特征对于每个用户都是一个历史行为序列，对于每个用户，历史行为序列长度会不一样，可能有的用户点击的历史文章多，有的点击的历史文章少，所以还需要把这个长度统一起来，
    在为DIN模型准备数据的时候，首先要通过SparseFeat函数指明这些类别型特征，然后还需要通过VarLenSparseFeat函数再进行序列填充，
    使得每个用户的历史序列一样长，所以这个函数参数中会有个maxlen，来指明序列的最大长度是多少。

    3、对于连续型特征列，只需要用DenseFeat函数来指明列名和维度即可。

    处理完特征列之后，把相应的数据与列进行对应，就得到了最后的数据。
    """
    # 准备历史数据
    if offline:
        all_data = pd.read_csv(data_path + '/train_click_log.csv')
    else:
        trn_data = pd.read_csv(data_path + '/train_click_log.csv')
        tst_data = pd.read_csv(data_path + '/testA_click_log.csv')
        all_data = trn_data.append(tst_data)
    hist_click = all_data[['user_id', 'click_article_id']].groupby('user_id').agg({list}).reset_index()
    his_behavior_df = pd.DataFrame()
    his_behavior_df['user_id'] = hist_click['user_id']
    his_behavior_df['hist_click_article_id'] = hist_click['click_article_id']
    # 准备特征数据
    trn_user_item_feats_df_din_model = trn_user_item_feats_df.copy()
    if offline:
        val_user_item_feats_df_din_model = val_user_item_feats_df.copy()
    else:
        val_user_item_feats_df_din_model = None
    tst_user_item_feats_df_din_model = tst_user_item_feats_df.copy()
    trn_user_item_feats_df_din_model = trn_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')
    if offline:
        val_user_item_feats_df_din_model = val_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')
    else:
        val_user_item_feats_df_din_model = None
    tst_user_item_feats_df_din_model = tst_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')

    # 把特征分开
    sparse_fea = ['user_id', 'click_article_id', 'category_id', 'click_environment', 'click_deviceGroup',
                  'click_os', 'click_country', 'click_region', 'click_referrer_type', 'is_cat_hab']

    behavior_fea = ['click_article_id']

    hist_behavior_fea = ['hist_click_article_id']

    dense_fea = ['sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min', 'sim_sum', 'sim_mean', 'score',
                 'rank', 'click_size', 'time_diff_mean', 'active_level', 'user_time_hob1', 'user_time_hob2',
                 'words_hbo', 'words_count']

    # dense特征进行归一化, 神经网络训练都需要将数值进行归一化处理
    mm = MinMaxScaler()

    # 下面是做一些特殊处理，当在其他的地方出现无效值的时候，不处理无法进行归一化，刚开始可以先把他注释掉，在运行了下面的代码
    # 之后如果发现报错，应该先去想办法处理如何不出现inf之类的值
    # trn_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)
    # tst_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)

    for feat in dense_fea:
        trn_user_item_feats_df_din_model[feat] = mm.fit_transform(trn_user_item_feats_df_din_model[[feat]])

        if val_user_item_feats_df_din_model is not None:
            val_user_item_feats_df_din_model[feat] = mm.fit_transform(val_user_item_feats_df_din_model[[feat]])

        tst_user_item_feats_df_din_model[feat] = mm.fit_transform(tst_user_item_feats_df_din_model[[feat]])

    # 准备训练数据
    x_trn, dnn_feature_columns = get_din_feats_columns(trn_user_item_feats_df_din_model, dense_fea,
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_trn = trn_user_item_feats_df_din_model['label'].values

    if offline:
        # 准备验证数据
        x_val, dnn_feature_columns = get_din_feats_columns(val_user_item_feats_df_din_model, dense_fea,
                                                           sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
        y_val = val_user_item_feats_df_din_model['label'].values

    dense_fea = [x for x in dense_fea if x != 'label']
    x_tst, dnn_feature_columns = get_din_feats_columns(tst_user_item_feats_df_din_model, dense_fea,
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    # 建立模型
    model = DIN(dnn_feature_columns, behavior_fea)
    # 查看模型结构
    model.summary()
    # 模型编译
    model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy', tf.keras.metrics.AUC()])
    # 模型训练
    if offline:
        history = model.fit(x_trn, y_trn, verbose=1, epochs=10, validation_data=(x_val, y_val) , batch_size=256)
    else:
        # 也可以使用上面的语句用自己采样出来的验证集
        # history = model.fit(x_trn, y_trn, verbose=1, epochs=3, validation_split=0.3, batch_size=256)
        history = model.fit(x_trn, y_trn, verbose=1, epochs=2, batch_size=256)
    # 模型预测
    tst_user_item_feats_df_din_model['pred_score'] = model.predict(x_tst, verbose=1, batch_size=256)
    tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'din_rank_score.csv', index=False)

    # 预测结果重新排序, 及生成提交结果
    rank_results = tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score']]
    rank_res(rank_results, topk=5, model_name='din')

    k_fold = 5
    trn_df = trn_user_item_feats_df_din_model
    user_set = get_kfold_users(trn_df, n=k_fold)

    score_list = []
    score_df = trn_df[['user_id', 'click_article_id', 'label']]
    sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

    dense_fea = [x for x in dense_fea if x != 'label']
    x_tst, dnn_feature_columns = get_din_feats_columns(tst_user_item_feats_df_din_model, dense_fea,
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)

    # 五折交叉验证，并将中间结果保存用于staking
    for n_fold, valid_user in enumerate(user_set):
        train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user
        valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]

        # 准备训练数据
        x_trn, dnn_feature_columns = get_din_feats_columns(train_idx, dense_fea,
                                                           sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
        y_trn = train_idx['label'].values

        # 准备验证数据
        x_val, dnn_feature_columns = get_din_feats_columns(valid_idx, dense_fea,
                                                           sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
        y_val = valid_idx['label'].values

        history = model.fit(x_trn, y_trn, verbose=1, epochs=2, validation_data=(x_val, y_val), batch_size=256)

        # 预测验证集结果
        valid_idx['pred_score'] = model.predict(x_val, verbose=1, batch_size=256)

        valid_idx.sort_values(by=['user_id', 'pred_score'])
        valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

        # 将验证集的预测结果放到一个列表中，后面进行拼接
        score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

        # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
        if not offline:
            sub_preds += model.predict(x_tst, verbose=1, batch_size=256)[:, 0]

        score_df_ = pd.concat(score_list, axis=0)
        score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
        # 保存训练集交叉验证产生的新特征
        score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
            save_path + 'trn_din_cls_feats.csv', index=False)

        # 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
        tst_user_item_feats_df_din_model['pred_score'] = sub_preds / k_fold
        tst_user_item_feats_df_din_model['pred_score'] = tst_user_item_feats_df_din_model['pred_score'].transform(
            lambda x: norm_sim(x))
        tst_user_item_feats_df_din_model.sort_values(by=['user_id', 'pred_score'])
        tst_user_item_feats_df_din_model['pred_rank'] = tst_user_item_feats_df_din_model.groupby(['user_id'])[
            'pred_score'].rank(ascending=False, method='first')

        # 保存测试集交叉验证的新特征
        tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
            save_path + 'tst_din_cls_feats.csv', index=False)

