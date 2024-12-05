import pandas as pd
from gensim.models import Word2Vec
import logging, pickle
from sklearn.preprocessing import MinMaxScaler
from dataset import data_path, save_path


def trian_item_word2vec(click_df, embed_size=16, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转换成句子的形式
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()

    # 为了方便查看训练的进度，设定一个log信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # 参数分别为：词向量维度、指定Skip-gram、上下文窗口、随机种子、线程数、最小词频（低于此词频的会忽略）、迭代次数
    w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=24, min_count=1, epochs=10)

    # 保存成字典的形式
    item_w2v_emb_dict = {k: w2v.wv[k] for k in click_df['click_article_id'] if k in w2v.wv}
    with open(save_path + save_name, 'wb') as f:
        pickle.dump(item_w2v_emb_dict, f)
    print("embedding保存完成")
    return item_w2v_emb_dict


trn_click = pd.read_csv(data_path + 'train_click_log.csv')
item_df = pd.read_csv(data_path + 'articles.csv')
item_df = item_df.rename(columns={'article_id': 'click_article_id'})  # 重命名，方便match
item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

# 对每个用户的点击时间戳进行排序
trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
# 计算用户点击文章的次数，并添加新的一列count
trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')
# 用户点击日志，训练集和测试集
trn_click = trn_click.merge(item_df, how='left', on=['click_article_id'])
tst_click = tst_click.merge(item_df, how='left', on=['click_article_id'])
# 合并训练集和测试集
user_click_merge = pd.concat([trn_click, tst_click], ignore_index=True)

# 对时间归一化
mm = MinMaxScaler()
user_click_merge['click_timestamp'] = mm.fit_transform(user_click_merge[['click_timestamp']])
user_click_merge['created_at_ts'] = mm.fit_transform(user_click_merge[['created_at_ts']])


user_click_merge = user_click_merge.sort_values('click_timestamp')
# {id: embedding}
item_w2v_emb_dict = trian_item_word2vec(user_click_merge)
print(len(item_w2v_emb_dict))

