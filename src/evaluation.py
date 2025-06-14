import numpy as np
from surprise import accuracy
from surprise import SVD
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, data_processor, model):
        """模型评估类，负责评估推荐系统性能"""
        self.data_processor = data_processor
        self.model = model
    
    def evaluate_rmse_mae(self, test_size=0.25) -> Tuple[float, float]:
        """评估模型的RMSE和MAE"""
        logger.info(f"开始评估模型性能，测试集大小: {test_size}")
        data = self.data_processor.get_surprise_data()
        trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
        
        self.model.model.fit(trainset)
        predictions = self.model.model.test(testset)
        
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        logger.info(f"评估完成 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae
    
    def evaluate_factor_dimension(self, factors_range: List[int], 
                                test_size=0.25) -> Dict[int, float]:
        """评估不同潜在因子数对模型性能的影响"""
        logger.info(f"开始因子维度评估，范围: {factors_range}")
        data = self.data_processor.get_surprise_data()
        results = {}
        
        for factors in factors_range:
            logger.info(f"评估因子数: {factors}")
            trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
            
            temp_model = SVD(n_factors=factors, n_epochs=20, lr_all=0.005, reg_all=0.02)
            temp_model.fit(trainset)
            predictions = temp_model.test(testset)
            
            rmse = accuracy.rmse(predictions)
            results[factors] = rmse
            logger.info(f"因子数 {factors} 的RMSE: {rmse:.4f}")
        
        # 可视化结果
        self._plot_factor_evaluation(results)
        return results
    
    def _plot_factor_evaluation(self, results: Dict[int, float]) -> None:
        """Visualize factor dimension evaluation results"""
        factors = sorted(results.keys())
        rmse = [results[f] for f in factors]
        
        plt.figure(figsize=(10, 6))
        plt.plot(factors, rmse, 'o-', linewidth=2)
        plt.xlabel('Number of Latent Factors', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.title('Effect of Latent Factors on Model Performance', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Mark the minimum value
        min_idx = rmse.index(min(rmse))
        plt.plot(factors[min_idx], rmse[min_idx], 'ro', label=f'Best Factor: {factors[min_idx]}')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('../results/factor_evaluation.png')
        plt.close()
    
    def evaluate_regularization(self, reg_range: List[float], 
                              test_size=0.25) -> Dict[float, float]:
        """评估不同正则化系数对模型性能的影响"""
        logger.info(f"开始正则化评估，范围: {reg_range}")
        data = self.data_processor.get_surprise_data()
        results = {}
        
        for reg in reg_range:
            logger.info(f"评估正则化系数: {reg}")
            trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
            
            temp_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=reg)
            temp_model.fit(trainset)
            predictions = temp_model.test(testset)
            
            rmse = accuracy.rmse(predictions)
            results[reg] = rmse
            logger.info(f"正则化系数 {reg} 的RMSE: {rmse:.4f}")
        
        # 可视化结果
        self._plot_regularization_evaluation(results)
        return results
    
    def _plot_regularization_evaluation(self, results: Dict[float, float]) -> None:
        """Visualize regularization evaluation results"""
        reg = sorted(results.keys())
        rmse = [results[r] for r in reg]
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(reg, rmse, 'o-', linewidth=2)
        plt.xlabel('Regularization Coefficient λ', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.title('Effect of Regularization on Model Generalization', fontsize=16)
        plt.grid(True, which='both', alpha=0.3)
        
        # Mark the minimum value
        min_idx = rmse.index(min(rmse))
        plt.plot(reg[min_idx], rmse[min_idx], 'ro', label=f'Best λ: {reg[min_idx]}')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('../results/reg_evaluation.png')
        plt.close()
    
    def plot_prediction_vs_actual(self, n=1000) -> None:
        """Plot scatter of predicted vs actual ratings"""
        logger.info("Plotting predicted vs actual ratings scatter plot")
        data = self.data_processor.get_surprise_data()
        trainset = data.build_full_trainset()
        testset = trainset.build_testset()
        
        self.model.model.fit(trainset)
        predictions = self.model.model.test(testset[:n])  # Take first n samples
        
        actual = [pred.r_ui for pred in predictions]
        predicted = [pred.est for pred in predictions]
        
        plt.figure(figsize=(10, 8))
        sns.regplot(x=actual, y=predicted, scatter_kws={'alpha': 0.5})
        plt.xlabel('Actual Rating', fontsize=14)
        plt.ylabel('Predicted Rating', fontsize=14)
        plt.title('Predicted vs Actual Ratings', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../results/pred_vs_actual.png')
        plt.close()
