# ArticleRec
一个文章推荐系统，包括召回和排序。

github无法上传大于25M的文件，所以数据集需要下载：

新建data目录：将数据集下载到该目录下，地址：https://tianchi.aliyun.com/competition/entrance/531842/information

新建temp目录：所有的pkl文件会保存在此目录下。

在笔记本的RTX4060上，整个推荐系统在完整数据集上一共需要跑20小时以上，最短的itemcf需要跑1个小时20分钟左右，在跑前需要先跑item_word2vec，获得embedding表征，或者使用数据集中的embedding。
