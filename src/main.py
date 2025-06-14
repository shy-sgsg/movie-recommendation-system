import os
from data_processing import DataProcessor
from models import MatrixFactorizationModel
from recommendation import Recommender
from evaluation import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主程序：基于矩阵分解的电影推荐系统"""
    # 确保结果目录存在
    if not os.path.exists('../results'):
        os.makedirs('../results')
    
    logger.info("=== 电影推荐系统启动 ===")
    
    # 1. 数据处理
    logger.info("=== 数据处理阶段 ===")
    data_processor = DataProcessor()
    data_processor.load_data()
    data_processor.explore_data()
    data_processor.preprocess_data()
    
    # 2. 模型训练
    logger.info("=== 模型训练阶段 ===")
    model = MatrixFactorizationModel(data_processor)
    
    # 使用网格搜索找到最佳超参数
    model.grid_search_surprise()
    
    # 或者使用预设的最佳参数训练
    # model.train_surprise_svd(n_factors=100, n_epochs=25, lr_all=0.005, reg_all=0.02)
    
    # 3. 模型评估
    logger.info("=== 模型评估阶段 ===")
    evaluator = ModelEvaluator(data_processor, model)
    
    # 基本性能评估
    rmse, mae = evaluator.evaluate_rmse_mae()
    logger.info(f"模型性能 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # 因子维度评估
    factor_range = [20, 50, 100, 150, 200]
    factor_results = evaluator.evaluate_factor_dimension(factor_range)
    best_factors = min(factor_results, key=factor_results.get)
    logger.info(f"最佳潜在因子数: {best_factors}, RMSE: {factor_results[best_factors]:.4f}")
    
    # 正则化评估
    reg_range = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08]
    reg_results = evaluator.evaluate_regularization(reg_range)
    best_reg = min(reg_results, key=reg_results.get)
    logger.info(f"最佳正则化系数: {best_reg}, RMSE: {reg_results[best_reg]:.4f}")
    
    # 预测vs实际评分可视化
    evaluator.plot_prediction_vs_actual()
    
    # 4. 生成推荐
    logger.info("=== 推荐生成阶段 ===")
    recommender = Recommender(data_processor, model)
    
    # 示例：为用户1生成推荐
    sample_user_id = 1
    recommendations = recommender.generate_recommendations(sample_user_id, n=10)
    
    logger.info(f"为用户 {sample_user_id} 生成的前10条推荐:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec['title']} (预测评分: {rec['predicted_rating']:.2f}, 类型: {rec['genres']})")
    
    # 获取用户类型偏好
    genre_preference = recommender.get_user_genre_preference(sample_user_id)
    logger.info(f"用户 {sample_user_id} 的类型偏好:")
    for genre, score in sorted(genre_preference.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"- {genre}: {score:.2f}")
    
    # 获取相似用户
    similar_users = recommender.get_similar_users(sample_user_id, n=3)
    logger.info(f"与用户 {sample_user_id} 相似的用户:")
    for user_id, similarity in similar_users:
        logger.info(f"- 用户 {user_id}, 相似度: {similarity:.2f}")
    
    logger.info("=== 推荐系统运行完成 ===")

if __name__ == "__main__":
    main()
