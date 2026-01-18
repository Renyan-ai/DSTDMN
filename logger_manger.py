"""
自动化日志文件管理系统
功能：
- 日志分级（DEBUG, INFO, WARNING, ERROR）
- 自动日志轮转（按大小和时间）
- 训练指标独立记录
- 自动归档和清理过期日志
- 结构化输出（控制台+文件）
"""

import logging
import os
import sys
import json
import time
import shutil
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional


class LoggerManager:
    """统一的日志管理类"""
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: str = None,
        log_level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        auto_clean_days: int = 30,
        console_output: bool = True
    ):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志根目录
            experiment_name: 实验名称（如果为None，将使用时间戳）
            log_level: 日志级别
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
            auto_clean_days: 自动清理多少天前的日志
            console_output: 是否输出到控制台
        """
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.auto_clean_days = auto_clean_days
        self.console_output = console_output
        
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name if experiment_name else timestamp
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 创建子目录
        self.train_log_dir = os.path.join(self.experiment_dir, "training")
        self.metric_log_dir = os.path.join(self.experiment_dir, "metrics")
        self.model_log_dir = os.path.join(self.experiment_dir, "models")
        self.archive_dir = os.path.join(self.experiment_dir, "archive")
        
        for directory in [self.train_log_dir, self.metric_log_dir, 
                         self.model_log_dir, self.archive_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 设置日志器
        self.logger = self._setup_logger(log_level)
        self.metric_logger = self._setup_metric_logger()
        
        # 训练指标缓存
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.epoch_times = []
        
        # 记录实验配置
        self.config = {}
        
        # 执行自动清理
        self._auto_clean_old_logs()
        
        self.logger.info(f"日志管理器初始化完成 - 实验目录: {self.experiment_dir}")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """设置主日志器"""
        logger = logging.getLogger(f"STAMTN_{self.experiment_name}")
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers.clear()
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器（轮转）
        log_file = os.path.join(self.train_log_dir, "training.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 错误日志单独记录
        error_file = os.path.join(self.train_log_dir, "error.log")
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        return logger
    
    def _setup_metric_logger(self) -> logging.Logger:
        """设置指标日志器"""
        metric_logger = logging.getLogger(f"STAMTN_Metrics_{self.experiment_name}")
        metric_logger.setLevel(logging.INFO)
        metric_logger.handlers.clear()
        metric_logger.propagate = False
        
        # 指标日志文件
        metric_file = os.path.join(self.metric_log_dir, "metrics.log")
        metric_handler = logging.FileHandler(metric_file, encoding='utf-8')
        metric_formatter = logging.Formatter('%(asctime)s - %(message)s')
        metric_handler.setFormatter(metric_formatter)
        metric_logger.addHandler(metric_handler)
        
        return metric_logger
    
    def save_config(self, config: Dict[str, Any]):
        """保存实验配置"""
        self.config = config
        config_file = os.path.join(self.experiment_dir, "config.json")
        
        # 将argparse.Namespace转换为字典
        if hasattr(config, '__dict__'):
            config = vars(config)
        
        # 转换为JSON可序列化的格式
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_config[key] = value
            else:
                serializable_config[key] = str(value)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"实验配置已保存: {config_file}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Epoch {epoch}/{total_epochs} 开始")
        self.logger.info(f"{'='*60}")
    
    def log_epoch_end(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                     train_time: float, val_time: float):
        """记录epoch结束"""
        # 记录到主日志
        self.logger.info(f"Epoch {epoch} 训练时间: {train_time:.4f} 秒")
        self.logger.info(f"Epoch {epoch} 验证时间: {val_time:.4f} 秒")
        self.logger.info(f"训练指标 - " + " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
        self.logger.info(f"验证指标 - " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        
        # 记录到指标日志
        metric_str = json.dumps({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'train_time': train_time,
            'val_time': val_time
        }, ensure_ascii=False)
        self.metric_logger.info(metric_str)
        
        # 缓存指标
        train_metrics['epoch'] = epoch
        val_metrics['epoch'] = epoch
        self.train_metrics.append(train_metrics.copy())
        self.val_metrics.append(val_metrics.copy())
        self.epoch_times.append({'epoch': epoch, 'train_time': train_time, 'val_time': val_time})
    
    def log_iteration(self, epoch: int, iteration: int, metrics: Dict):
        """记录训练迭代"""
        log_str = f"Epoch: {epoch:03d}, Iter: {iteration:03d} - " + \
                  " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(log_str)
    
    def log_test_results(self, test_metrics: Dict, horizon_metrics: list = None):
        """记录测试结果"""
        self.logger.info(f"{'='*60}")
        self.logger.info("测试结果:")
        self.logger.info(f"{'='*60}")
        
        for key, value in test_metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        if horizon_metrics:
            self.logger.info("\n各预测时间步的详细指标:")
            for i, metrics in enumerate(horizon_metrics):
                self.logger.info(f"Horizon {i+1}: " + 
                               " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        
        # 保存测试结果
        self.test_metrics = test_metrics
        test_file = os.path.join(self.metric_log_dir, "test_results.json")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump({
                'overall': test_metrics,
                'horizons': horizon_metrics if horizon_metrics else []
            }, f, indent=4, ensure_ascii=False)
    
    def log_model_info(self, model_info: str):
        """记录模型信息"""
        self.logger.info("模型结构:")
        self.logger.info(model_info)
        
        model_info_file = os.path.join(self.model_log_dir, "model_info.txt")
        with open(model_info_file, 'w', encoding='utf-8') as f:
            f.write(model_info)
    
    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """记录最佳模型信息"""
        self.logger.info(f"### 发现更好的模型! ###")
        self.logger.info(f"Epoch: {epoch}, {metric_name}: {metric_value:.4f}")
        
        best_model_info = {
            'epoch': epoch,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        best_info_file = os.path.join(self.model_log_dir, "best_model_info.json")
        with open(best_info_file, 'w', encoding='utf-8') as f:
            json.dump(best_model_info, f, indent=4, ensure_ascii=False)
    
    def save_metrics_to_csv(self):
        """保存所有指标到CSV文件"""
        if self.train_metrics:
            train_df = pd.DataFrame(self.train_metrics)
            train_csv = os.path.join(self.metric_log_dir, "train_metrics.csv")
            train_df.to_csv(train_csv, index=False)
            self.logger.info(f"训练指标已保存: {train_csv}")
        
        if self.val_metrics:
            val_df = pd.DataFrame(self.val_metrics)
            val_csv = os.path.join(self.metric_log_dir, "val_metrics.csv")
            val_df.to_csv(val_csv, index=False)
            self.logger.info(f"验证指标已保存: {val_csv}")
        
        if self.epoch_times:
            time_df = pd.DataFrame(self.epoch_times)
            time_csv = os.path.join(self.metric_log_dir, "epoch_times.csv")
            time_df.to_csv(time_csv, index=False)
            self.logger.info(f"训练时间已保存: {time_csv}")
    
    def _auto_clean_old_logs(self):
        """自动清理过期日志"""
        if not os.path.exists(self.log_dir):
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.auto_clean_days)
        cleaned_count = 0
        
        for item in os.listdir(self.log_dir):
            item_path = os.path.join(self.log_dir, item)
            
            # 跳过当前实验目录
            if item == self.experiment_name:
                continue
            
            if os.path.isdir(item_path):
                # 检查目录修改时间
                mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                
                if mtime < cutoff_date:
                    # 归档到archive目录（可选）
                    archive_path = os.path.join(self.log_dir, "archive", item)
                    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                    
                    try:
                        shutil.move(item_path, archive_path)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"警告: 无法归档目录 {item_path}: {e}")
        
        if cleaned_count > 0:
            print(f"已归档 {cleaned_count} 个超过 {self.auto_clean_days} 天的日志目录")
    
    def get_model_save_path(self) -> str:
        """获取模型保存路径"""
        return self.model_log_dir
    
    def get_experiment_dir(self) -> str:
        """获取实验目录路径"""
        return self.experiment_dir
    
    def finalize(self):
        """完成日志记录，执行最终操作"""
        self.logger.info("训练完成，正在保存最终指标...")
        self.save_metrics_to_csv()
        
        # 创建实验摘要
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.experiment_name if not self.config else self.config.get('start_time', self.experiment_name),
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_epochs': len(self.train_metrics),
            'config': self.config,
            'best_model_path': os.path.join(self.model_log_dir, "best_model.pth")
        }
        
        if self.test_metrics:
            summary['test_results'] = self.test_metrics
        
        summary_file = os.path.join(self.experiment_dir, "experiment_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"实验摘要已保存: {summary_file}")
        self.logger.info(f"所有日志文件保存在: {self.experiment_dir}")


def setup_logger(
    log_dir: str = "./logs",
    experiment_name: str = None,
    log_level: str = "INFO",
    **kwargs
) -> LoggerManager:
    """
    快速设置日志管理器的便捷函数
    
    Args:
        log_dir: 日志根目录
        experiment_name: 实验名称
        log_level: 日志级别
        **kwargs: 其他LoggerManager参数
    
    Returns:
        LoggerManager实例
    """
    return LoggerManager(
        log_dir=log_dir,
        experiment_name=experiment_name,
        log_level=log_level,
        **kwargs
    )


if __name__ == "__main__":
    # 测试日志管理器
    logger_mgr = setup_logger(experiment_name="test_experiment")
    
    logger_mgr.save_config({'learning_rate': 0.001, 'batch_size': 64})
    logger_mgr.log_epoch_start(1, 10)
    
    train_metrics = {'loss': 0.5, 'mae': 0.3, 'rmse': 0.4}
    val_metrics = {'loss': 0.45, 'mae': 0.28, 'rmse': 0.38}
    
    logger_mgr.log_epoch_end(1, train_metrics, val_metrics, 120.5, 30.2)
    logger_mgr.finalize()
    
    print("日志管理器测试完成!")

