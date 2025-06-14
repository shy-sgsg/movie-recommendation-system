# 基于矩阵分解的电影推荐系统

本项目实现了一个基于矩阵分解的电影推荐系统，使用经典的MovieLens数据集，通过SVD等矩阵分解算法实现高效的个性化电影推荐。系统采用模块化设计，涵盖数据处理、模型训练、推荐生成和性能评估全流程。

## 项目结构

```
movie-recommendation-system/
├── data/                # 数据集存储目录
│   ├── ml-latest-small/
│   │   ├── ratings.csv  # 用户评分数据
│   │   └── movies.csv   # 电影元数据
├── src/                 # 源代码目录
│   ├── data_processing.py  # 数据加载与预处理模块
│   ├── models.py        # 矩阵分解模型实现
│   ├── recommendation.py # 推荐生成模块
│   ├── evaluation.py    # 模型评估模块
│   └── main.py          # 主程序入口
├── requirements.txt     # 依赖包列表
├── results/             # 结果存储目录
├── report.md            # Markdown版报告
├── report.pdf           # PDF版报告
└── README.md            # 项目说明文档
```

## 环境要求

- **Python版本**：3.8+
- **核心依赖**：
  ```text
  pandas>=1.5.3     # 数据处理
  numpy>=1.23.5     # 数值计算
  scipy>=1.10.1     # 科学计算
  matplotlib>=3.7.1  # 数据可视化
  seaborn>=0.12.2   # 统计可视化
  surprise>=1.1.1    # 推荐系统库
  pyspark>=3.4.1     # 分布式计算（可选）
  tqdm>=4.65.0      # 进度条显示
  ```

## 快速开始

### 1. 数据准备

1. 从[MovieLens官网](https://grouplens.org/datasets/movielens/)下载 `ml-latest-small`数据集
2. 解压后放入 `data/`目录，确保目录结构包含：
   - `data/ml-latest-small/ratings.csv`
   - `data/ml-latest-small/movies.csv`

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行系统

```bash
python src/main.py
```

## 模块功能说明

### 数据处理模块（data_processing.py）

```python
class DataProcessor:
    def load_data(self) -> None:
        """加载评分和电影数据，支持异常处理"""
  
    def explore_data(self) -> None:
        """数据探索与可视化，包括稀疏度计算和评分分布绘图"""
  
    def preprocess_data(self, min_ratings=5, min_movie_ratings=10) -> None:
        """数据预处理流程：
        - 异常评分过滤
        - 低活跃度用户/电影过滤
        - 稀疏矩阵构建
        - 时间特征工程"""
  
    def get_surprise_data(self) -> Dataset:
        """生成适用于surprise库的数据集格式"""
  
    def get_movie_genres(self) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """电影类型独热编码处理，返回类型频次统计"""
```

### 矩阵分解模型模块（models.py）

```python
class MatrixFactorizationModel:
    def train_surprise_svd(self, n_factors=100, n_epochs=25, ...) -> None:
        """基于surprise库的SVD模型训练，支持偏置项和正则化"""
  
    def grid_search_surprise(self, param_grid=None) -> None:
        """超参数网格搜索，自动优化潜在因子数、学习率等参数"""
  
    def train_spark_als(self, rank=100, max_iter=15, ...) -> None:
        """基于Spark的ALS模型训练，支持大规模数据分布式计算"""
  
    def custom_svd(self, k=50, steps=500, ...) -> Tuple[np.ndarray, np.ndarray]:
        """自定义SGD实现的矩阵分解，用于教学和原理理解"""
  
    def predict(self, user_id: int, movie_id: int) -> float:
        """用户对电影的评分预测，支持冷启动默认评分处理"""
```

### 推荐生成模块（recommendation.py）

```python
class Recommender:
    def get_user_watched_movies(self, user_id: int) -> Set[int]:
        """获取用户已评分电影集合"""
  
    def get_unwatched_movies(self, user_id: int) -> List[int]:
        """获取用户未评分电影列表"""
  
    def generate_recommendations(self, user_id: int, n=10, ...) -> List[Dict]:
        """生成个性化推荐列表，支持类型偏好过滤"""
  
    def get_user_genre_preference(self, user_id: int) -> Dict[str, float]:
        """计算用户类型偏好权重，基于历史评分加权平均"""
  
    def get_similar_users(self, user_id: int, n=5) -> List[Tuple[int, float]]:
        """基于Pearson相关系数的相似用户发现"""
```

### 模型评估模块（evaluation.py）

```python
class ModelEvaluator:
    def evaluate_rmse_mae(self, test_size=0.25) -> Tuple[float, float]:
        """计算RMSE和MAE评估指标"""
  
    def evaluate_factor_dimension(self, factors_range: List[int]) -> Dict[int, float]:
        """潜在因子数对模型性能的影响评估，包含可视化"""
  
    def evaluate_regularization(self, reg_range: List[float]) -> Dict[float, float]:
        """正则化系数优化评估，包含对数坐标可视化"""
  
    def plot_prediction_vs_actual(self, n=1000) -> None:
        """预测评分与实际评分的散点图可视化"""
```

## 实验结果分析

运行主程序后，结果将保存在 `results/`目录，包含：

- **评分分布直方图**：展示数据集评分频次分布
- **潜在因子数评估图**：不同因子维度下的RMSE变化曲线
- **正则化评估图**：正则化系数与模型性能的关系曲线
- **预测vs实际评分散点图**：模型预测准确性可视化
- **推荐结果日志**：包含用户推荐列表、类型偏好和相似用户

## 扩展方向

1. **混合推荐模型**：结合内容特征（如电影类型、导演）与协同过滤
2. **时间动态模型**：引入评分时间衰减因子，捕捉用户偏好变化
3. **深度学习模型**：实现神经协同过滤(NCF)，处理非线性特征交互
4. **大规模数据优化**：基于Spark实现分布式ALS，支持亿级数据
5. **冷启动解决方案**：结合知识图谱和迁移学习，解决新用户/电影推荐难题

本项目完整实现了矩阵分解在推荐系统中的应用，代码采用模块化设计，便于理解和扩展，适合作为矩阵分解算法的学习案例和推荐系统工程实践参考。
