import pandas as pd
from dataset import save_path
from rank_models import rank_res


if __name__ == '__main__':
    # stacking
    # 读取多个模型的交叉验证生成的结果文件
    # 训练集
    trn_lgb_ranker_feats = pd.read_csv(save_path + 'trn_lgb_ranker_feats.csv')
    trn_lgb_cls_feats = pd.read_csv(save_path + 'trn_lgb_cls_feats.csv')
    trn_din_cls_feats = pd.read_csv(save_path + 'trn_din_cls_feats.csv')

    # 测试集
    tst_lgb_ranker_feats = pd.read_csv(save_path + 'tst_lgb_ranker_feats.csv')
    tst_lgb_cls_feats = pd.read_csv(save_path + 'tst_lgb_cls_feats.csv')
    tst_din_cls_feats = pd.read_csv(save_path + 'tst_din_cls_feats.csv')

    # 将多个模型输出的特征进行拼接

    finall_trn_ranker_feats = trn_lgb_ranker_feats[['user_id', 'click_article_id', 'label']]
    finall_tst_ranker_feats = tst_lgb_ranker_feats[['user_id', 'click_article_id']]

    for idx, trn_model in enumerate([trn_lgb_ranker_feats, trn_lgb_cls_feats, trn_din_cls_feats]):
        for feat in ['pred_score', 'pred_rank']:
            col_name = feat + '_' + str(idx)
            finall_trn_ranker_feats[col_name] = trn_model[feat]

    for idx, tst_model in enumerate([tst_lgb_ranker_feats, tst_lgb_cls_feats, tst_din_cls_feats]):
        for feat in ['pred_score', 'pred_rank']:
            col_name = feat + '_' + str(idx)
            finall_tst_ranker_feats[col_name] = tst_model[feat]

    # 定义一个逻辑回归模型再次拟合交叉验证产生的特征对测试集进行预测
    # 这里需要注意的是，在做交叉验证的时候可以构造多一些与输出预测值相关的特征，来丰富这里简单模型的特征
    from sklearn.linear_model import LogisticRegression

    feat_cols = ['pred_score_0', 'pred_rank_0', 'pred_score_1', 'pred_rank_1', 'pred_score_2', 'pred_rank_2']

    trn_x = finall_trn_ranker_feats[feat_cols]
    trn_y = finall_trn_ranker_feats['label']

    tst_x = finall_tst_ranker_feats[feat_cols]

    # 定义模型
    lr = LogisticRegression()

    # 模型训练
    lr.fit(trn_x, trn_y)

    # 模型预测
    finall_tst_ranker_feats['pred_score'] = lr.predict_proba(tst_x)[:, 1]

    # 预测结果重新排序, 及生成结果
    rank_results = finall_tst_ranker_feats[['user_id', 'click_article_id', 'pred_score']]
    rank_res(rank_results, topk=5, model_name='ensumble_staking')


