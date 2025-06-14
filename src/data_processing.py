import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # You can remove this line if not using Chinese
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from typing import Tuple, Dict, Set, List

class DataProcessor:
    def __init__(self, data_path: str = "../data/ml-latest-small/"):
        """Data processor for loading, cleaning, and transforming MovieLens dataset"""
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.user2idx = None
        self.idx2user = None
        self.movie2idx = None
        self.idx2movie = None
        self.rating_matrix = None
        self.sparsity = 0.0

    def load_data(self) -> None:
        """Load ratings and movies data"""
        try:
            self.ratings = pd.read_csv(f"{self.data_path}ratings.csv")
            self.movies = pd.read_csv(f"{self.data_path}movies.csv")
            print(f"Data loaded successfully - Ratings: {len(self.ratings)}, Movies: {len(self.movies)}")
        except FileNotFoundError:
            print(f"Data files not found. Please make sure the data is in {self.data_path}")
            raise

    def explore_data(self) -> None:
        """Data exploration and visualization"""
        if self.ratings is None or self.movies is None:
            self.load_data()
        
        n_users = self.ratings['userId'].nunique()
        n_movies = self.ratings['movieId'].nunique()
        n_ratings = len(self.ratings)
        self.sparsity = 1 - n_ratings / (n_users * n_movies)
        
        print(f"Number of users: {n_users}, Number of movies: {n_movies}, Sparsity: {self.sparsity*100:.2f}%")
        
        # Ratings distribution visualization
        plt.figure(figsize=(8, 5))
        sns.countplot(x='rating', hue='rating', data=self.ratings, palette='Blues_d', legend=False)
        plt.title('Ratings Distribution Histogram')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.7)
        plt.tight_layout()
        plt.savefig('../results/ratings_distribution.png')
        plt.close()

    def preprocess_data(self, min_ratings=5, min_movie_ratings=10) -> None:
        """数据预处理：清洗异常值、过滤低活跃度用户和电影"""
        if self.ratings is None or self.movies is None:
            self.load_data()
        
        # 过滤异常评分
        self.ratings = self.ratings[(self.ratings['rating'] >= 0.5) & 
                                    (self.ratings['rating'] <= 5.0)]
        
        # 过滤低活跃度用户和电影
        user_activity = self.ratings['userId'].value_counts()
        active_users = user_activity[user_activity >= min_ratings].index
        movie_activity = self.ratings['movieId'].value_counts()
        active_movies = movie_activity[movie_activity >= min_movie_ratings].index
        self.ratings = self.ratings[self.ratings['userId'].isin(active_users) & 
                                   self.ratings['movieId'].isin(active_movies)]
        
        # 重建索引映射
        self.user2idx = {u: i for i, u in enumerate(self.ratings['userId'].unique())}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.movie2idx = {m: i for i, m in enumerate(self.ratings['movieId'].unique())}
        self.idx2movie = {i: m for m, i in self.movie2idx.items()}
        
        # 构建稀疏评分矩阵
        row = self.ratings['userId'].map(self.user2idx).values
        col = self.ratings['movieId'].map(self.movie2idx).values
        data = self.ratings['rating'].values
        self.rating_matrix = csr_matrix((data, (row, col)), 
                                       shape=(len(self.user2idx), len(self.movie2idx)))
        
        # 时间特征提取
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['year'] = self.ratings['timestamp'].dt.year
        self.ratings['month'] = self.ratings['timestamp'].dt.month
        
        # 计算用户首次评分时间
        user_first_rating = self.ratings.groupby('userId')['timestamp'].min().reset_index()
        user_first_rating.columns = ['userId', 'first_rating']
        self.ratings = pd.merge(self.ratings, user_first_rating, on='userId')
        self.ratings['rating_duration'] = (self.ratings['timestamp'] - self.ratings['first_rating']).dt.days
        
        print(f"预处理完成 - 剩余用户: {len(self.user2idx)}, 电影: {len(self.movie2idx)}")

    def get_surprise_data(self) -> Dataset:
        """获取适合surprise库的数据集格式"""
        if self.ratings is None:
            self.preprocess_data()
        
        reader = Reader(rating_scale=(0.5, 5.0))
        return Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)

    def get_movie_genres(self) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """获取电影类型独热编码"""
        if self.movies is None:
            self.load_data()
        
        # 提取所有类型
        genres = set()
        for g_list in self.movies['genres'].str.split('|'):
            genres.update(g_list)
        
        # 独热编码
        for genre in genres:
            self.movies[genre] = self.movies['genres'].str.contains(genre, regex=False).astype(int)
        
        # 计算类型出现频次
        genre_counts = self.movies[list(genres)].sum().sort_values(ascending=False)
        return self.movies, dict(genre_counts)
