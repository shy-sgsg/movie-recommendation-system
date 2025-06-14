import pandas as pd
from typing import List, Tuple, Dict, Set
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self, data_processor, model):
        """推荐系统类，负责生成个性化推荐"""
        self.data_processor = data_processor
        self.model = model
        self.movies, self.genre_counts = data_processor.get_movie_genres()
    
    def get_user_watched_movies(self, user_id: int) -> Set[int]:
        """获取用户已评分的电影"""
        if user_id not in self.data_processor.user2idx:
            logger.warning(f"用户 {user_id} 不在数据集中")
            return set()
        
        user_ratings = self.data_processor.ratings[self.data_processor.ratings['userId'] == user_id]
        return set(user_ratings['movieId'].unique())
    
    def get_unwatched_movies(self, user_id: int) -> List[int]:
        """获取用户未评分的电影"""
        watched = self.get_user_watched_movies(user_id)
        all_movies = set(self.data_processor.ratings['movieId'].unique())
        return list(all_movies - watched)
    
    def get_movie_info(self, movie_id: int) -> Tuple[str, str]:
        """获取电影名称和类型"""
        if movie_id not in self.movies['movieId'].values:
            return "未知电影", "未知类型"
        
        movie = self.movies[self.movies['movieId'] == movie_id].iloc[0]
        return movie['title'], movie['genres']
    
    def generate_recommendations(self, user_id: int, n=10, 
                                consider_genre=True) -> List[Dict[str, any]]:
        """
        为用户生成推荐列表
        consider_genre: 是否考虑用户历史偏好类型
        """
        if user_id not in self.data_processor.user2idx:
            logger.error(f"用户 {user_id} 不存在")
            return []
        
        logger.info(f"为用户 {user_id} 生成推荐...")
        unwatched = self.get_unwatched_movies(user_id)
        
        if not unwatched:
            logger.info(f"用户 {user_id} 已评分所有电影，无法生成推荐")
            return []
        
        # 预测未观看电影的评分
        predictions = []
        for movie_id in unwatched:
            pred_rating = self.model.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]
        
        # 构建推荐结果
        recommendations = []
        for movie_id, rating in top_n:
            title, genres = self.get_movie_info(movie_id)
            recommendations.append({
                'movie_id': movie_id,
                'title': title,
                'genres': genres,
                'predicted_rating': rating
            })
        
        logger.info(f"推荐生成完成，共 {len(recommendations)} 部电影")
        return recommendations
    
    def get_user_genre_preference(self, user_id: int) -> Dict[str, float]:
        """获取用户的类型偏好"""
        if user_id not in self.data_processor.user2idx:
            return {}
        
        user_ratings = self.data_processor.ratings[self.data_processor.ratings['userId'] == user_id]
        user_movies = pd.merge(user_ratings, self.movies, on='movieId')
        
        # 计算类型评分加权平均
        genres = set()
        for g_list in user_movies['genres'].str.split('|'):
            genres.update(g_list)
        
        genre_preference = {genre: 0 for genre in genres}
        total_ratings = 0
        
        for _, movie in user_movies.iterrows():
            movie_genres = movie['genres'].split('|')
            rating = movie['rating']
            total_ratings += rating
            
            for genre in movie_genres:
                genre_preference[genre] += rating
        
        # 归一化偏好
        if total_ratings > 0:
            for genre in genre_preference:
                genre_preference[genre] /= total_ratings
        
        return genre_preference
    
    def get_similar_users(self, user_id: int, n=5) -> List[Tuple[int, float]]:
        """获取相似用户（基于评分相似度）"""
        if user_id not in self.data_processor.user2idx:
            return []
        
        # 简化实现：使用Pearson相关系数计算用户相似度
        user_idx = self.data_processor.user2idx[user_id]
        R = self.data_processor.rating_matrix.toarray()
        user_ratings = R[user_idx]
        
        similarities = []
        for i in range(R.shape[0]):
            if i == user_idx:
                continue
            
            other_ratings = R[i]
            # 只考虑两者都评分的电影
            both_rated = (user_ratings > 0) & (other_ratings > 0)
            if np.sum(both_rated) < 5:
                continue  # 至少5部共同评分电影
            
            # 计算Pearson相关系数
            user_sub = user_ratings[both_rated] - np.mean(user_ratings[both_rated])
            other_sub = other_ratings[both_rated] - np.mean(other_ratings[both_rated])
            if np.std(user_sub) * np.std(other_sub) == 0:
                similarity = 0
            else:
                similarity = np.sum(user_sub * other_sub) / (np.std(user_sub) * np.std(other_sub))
            
            similarities.append((i, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = [(self.data_processor.idx2user[i], sim) for i, sim in similarities[:n]]
        return similar_users
