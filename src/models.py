import numpy as np
from surprise import SVD, Dataset
from surprise.model_selection import GridSearchCV
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from typing import Tuple, Dict, Any, Optional
import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatrixFactorizationModel:
    def __init__(self, data_processor):
        """矩阵分解模型基类"""
        self.data_processor = data_processor
        self.model = None

    def train_surprise_svd(self, n_factors=100, n_epochs=25, lr_all=0.005, reg_all=0.02, 
                          biased=True, init_mean=3.5) -> None:
        """使用surprise库的SVD模型训练"""
        logger.info(f"开始训练surprise SVD模型，因子数: {n_factors}, 迭代: {n_epochs}")
        data = self.data_processor.get_surprise_data()
        trainset = data.build_full_trainset()
        
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            biased=biased,
            init_mean=init_mean,
            random_state=42
        )
        self.model.fit(trainset)
        logger.info("surprise SVD模型训练完成")

    def grid_search_surprise(self, param_grid=None) -> None:
        """使用网格搜索优化surprise模型超参数"""
        if param_grid is None:
            param_grid = {
                'n_factors': [50, 100, 150],
                'n_epochs': [20, 30],
                'lr_all': [0.002, 0.005],
                'reg_all': [0.02, 0.04]
            }
        
        logger.info("开始网格搜索超参数...")
        data = self.data_processor.get_surprise_data()
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(data)
        
        logger.info(f"最佳RMSE: {gs.best_score['rmse']}")
        logger.info(f"最佳参数: {gs.best_params['rmse']}")
        self.model = gs.best_estimator['rmse']

    def train_spark_als(self, rank=100, max_iter=15, reg_param=0.01, 
                       alpha=10.0, implicit_prefs=False) -> None:
        """使用Spark的ALS模型训练（适用于大规模数据）"""
        logger.info(f"开始训练Spark ALS模型，rank: {rank}, 迭代: {max_iter}")
        try:
            spark = SparkSession.builder \
                .appName("Movie Recommendation") \
                .config("spark.executor.memory", "8g") \
                .config("spark.driver.memory", "4g") \
                .getOrCreate()
            
            ratings_spark = spark.createDataFrame(self.data_processor.ratings)
            
            als = ALS(
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                rank=rank,
                maxIter=max_iter,
                regParam=reg_param,
                alpha=alpha,
                implicitPrefs=implicit_prefs,
                coldStartStrategy="drop"
            )
            
            self.model = als.fit(ratings_spark)
            logger.info("Spark ALS模型训练完成")
            spark.stop()
        except Exception as e:
            logger.error(f"Spark训练出错: {e}")
            raise

    def custom_svd(self, k=50, steps=500, learning_rate=0.001, reg_param=0.01) -> Tuple[np.ndarray, np.ndarray]:
        """自定义SGD实现的矩阵分解（教学目的）"""
        if self.data_processor.rating_matrix is None:
            self.data_processor.preprocess_data()
        
        R = self.data_processor.rating_matrix.toarray()
        m, n = R.shape
        logger.info(f"自定义SVD训练，矩阵大小: {m}x{n}, 潜在因子: {k}")
        
        # 初始化参数
        P = np.random.normal(scale=1./np.sqrt(k), size=(m, k))
        Q = np.random.normal(scale=1./np.sqrt(k), size=(n, k))
        b_u = np.zeros(m)  # 用户偏差
        b_i = np.zeros(n)  # 物品偏差
        b = np.mean(R[R>0])  # 全局偏差
        
        # 记录用户-物品交互索引
        user_idx, item_idx = np.where(R > 0)
        interactions = list(zip(user_idx, item_idx))
        loss_history = []
        
        # SGD迭代
        for step in tqdm.tqdm(range(steps), desc="训练进度"):
            np.random.shuffle(interactions)
            for u, i in interactions:
                pred = b + b_u[u] + b_i[i] + P[u].dot(Q[i].T)
                e = R[u, i] - pred
                b_u[u] += learning_rate * (e - reg_param * b_u[u])
                b_i[i] += learning_rate * (e - reg_param * b_i[i])
                P[u] += learning_rate * (e * Q[i] - reg_param * P[u])
                Q[i] += learning_rate * (e * P[u] - reg_param * Q[i])
            
            # 每100步计算损失
            if (step+1) % 100 == 0:
                pred_matrix = b + b_u[:, np.newaxis] + b_i[np.newaxis, :] + P.dot(Q.T)
                mask = R > 0
                loss = np.sum((R[mask] - pred_matrix[mask])** 2) + \
                       reg_param * (np.sum(b_u**2) + np.sum(b_i** 2) + 
                                   np.sum(P**2) + np.sum(Q**2))
                loss_history.append(loss)
                if step % 500 == 0:
                    logger.info(f"步骤 {step+1}, 损失: {loss:.4f}")
        
        logger.info(f"自定义SVD训练完成，最终损失: {loss_history[-1]:.4f}")
        return P, Q

    def predict(self, user_id: int, movie_id: int) -> float:
        """预测用户对电影的评分"""
        if self.model is None:
            logger.warning("模型未训练，使用默认SVD模型")
            self.train_surprise_svd()
        
        # 转换为模型所需的ID
        if user_id not in self.data_processor.user2idx:
            logger.warning(f"用户 {user_id} 不在训练数据中")
            return 3.0  # 默认评分
        
        if movie_id not in self.data_processor.movie2idx:
            logger.warning(f"电影 {movie_id} 不在训练数据中")
            return 3.0  # 默认评分
        
        user_idx = self.data_processor.user2idx[user_id]
        movie_idx = self.data_processor.movie2idx[movie_id]
        
        # 使用surprise模型预测
        pred = self.model.predict(user_idx, movie_idx)
        return pred.est
