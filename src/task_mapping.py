"""
Task to Task ID Mapping for HiDe-Prompt Inference

This module provides a simple and efficient mapping mechanism from task names/types
to task IDs for direct prompt selection during inference phase.
"""

import json
import logging
from typing import Dict, Union, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskMapper:
    """
    A mapping class to convert task names/types to task IDs for HiDe-Prompt inference.
    
    支持多种任务映射方式：
    1. 从配置文件加载映射
    2. 动态创建映射
    3. 默认基础映射
    """
    
    def __init__(self, mapping_config: Optional[Union[str, Dict, Path]] = None):
        """
        Initialize TaskMapper with optional mapping configuration.
        
        Args:
            mapping_config: Can be:
                - str/Path: Path to JSON configuration file
                - dict: Direct mapping dictionary
                - None: Use default mapping
        """
        self.task_to_id = {}
        self.id_to_task = {}
        
        if mapping_config is None:
            self._load_default_mapping()
        elif isinstance(mapping_config, (str, Path)):
            self._load_from_file(mapping_config)
        elif isinstance(mapping_config, dict):
            self._load_from_dict(mapping_config)
        else:
            raise ValueError(f"Unsupported mapping_config type: {type(mapping_config)}")
    
    def _load_default_mapping(self):
        """Load default task mapping - empty for continual learning where task_id is based on execution order."""
        # 在continual learning中，task_id是基于执行顺序的，不是固定的任务映射
        # 因此默认映射为空，具体的task_id由训练脚本根据配置目录动态确定
        default_mapping = {}
        
        self._load_from_dict(default_mapping)
        logger.info(f"Initialized empty task mapping for continual learning")
    
    def _load_from_file(self, file_path: Union[str, Path]):
        """Load task mapping from JSON file."""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Mapping file {file_path} not found, using default mapping")
            self._load_default_mapping()
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 支持嵌套配置格式
            if 'task_mappings' in config:
                mapping = {}
                for category in config['task_mappings'].values():
                    mapping.update(category)
            elif 'mapping' in config:
                mapping = config['mapping']
            else:
                mapping = config
                
            self._load_from_dict(mapping)
            logger.info(f"Loaded task mapping from {file_path} with {len(self.task_to_id)} tasks")
            
        except Exception as e:
            logger.error(f"Failed to load mapping from {file_path}: {e}")
            self._load_default_mapping()
    
    def _load_from_dict(self, mapping: Dict):
        """Load task mapping from dictionary."""
        self.task_to_id = mapping.copy()
        self.id_to_task = {v: k for k, v in mapping.items()}
        
        # 验证映射的有效性
        if len(self.task_to_id) != len(self.id_to_task):
            logger.warning("Duplicate task IDs found in mapping, some tasks may be overridden")
    
    def get_task_id(self, task_name: str) -> int:
        """
        Get task ID for given task name.
        
        Args:
            task_name: Task name or identifier
            
        Returns:
            Task ID (integer)
            
        Raises:
            ValueError: If task name is not found in mapping
        """
        if task_name not in self.task_to_id:
            # 尝试模糊匹配
            task_name_lower = task_name.lower()
            for task, task_id in self.task_to_id.items():
                if task.lower() == task_name_lower:
                    return task_id
            
            # 如果还是找不到，记录错误并返回默认值
            logger.error(f"Task '{task_name}' not found in mapping. Available tasks: {list(self.task_to_id.keys())}")
            raise ValueError(f"Unknown task: {task_name}")
        
        return self.task_to_id[task_name]
    
    def get_task_name(self, task_id: int) -> str:
        """
        Get task name for given task ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task name
            
        Raises:
            ValueError: If task ID is not found in mapping
        """
        if task_id not in self.id_to_task:
            logger.error(f"Task ID {task_id} not found in mapping. Available IDs: {list(self.id_to_task.keys())}")
            raise ValueError(f"Unknown task ID: {task_id}")
        
        return self.id_to_task[task_id]
    
    def add_task(self, task_name: str, task_id: int):
        """Add a new task mapping."""
        if task_name in self.task_to_id:
            logger.warning(f"Task '{task_name}' already exists, updating mapping")
        if task_id in self.id_to_task:
            logger.warning(f"Task ID {task_id} already exists, updating mapping")
        
        self.task_to_id[task_name] = task_id
        self.id_to_task[task_id] = task_name
    
    def remove_task(self, task_name: str):
        """Remove a task mapping."""
        if task_name not in self.task_to_id:
            logger.warning(f"Task '{task_name}' not found in mapping")
            return
        
        task_id = self.task_to_id[task_name]
        del self.task_to_id[task_name]
        del self.id_to_task[task_id]
    
    def get_all_tasks(self) -> List[str]:
        """Get all available task names."""
        return list(self.task_to_id.keys())
    
    def get_all_task_ids(self) -> List[int]:
        """Get all available task IDs."""
        return list(self.id_to_task.keys())
    
    def get_max_task_id(self) -> int:
        """Get the maximum task ID."""
        return max(self.id_to_task.keys()) if self.id_to_task else -1
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save current mapping to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "task_mappings": {
                "general": self.task_to_id
            },
            "created_by": "TaskMapper",
            "num_tasks": len(self.task_to_id)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved task mapping to {file_path}")
    
    def __str__(self):
        return f"TaskMapper({len(self.task_to_id)} tasks: {list(self.task_to_id.keys())})"
    
    def __repr__(self):
        return self.__str__()


# 创建全局默认实例
default_task_mapper = TaskMapper()


def get_task_id(task_name: str, mapper: Optional[TaskMapper] = None) -> int:
    """
    便捷函数：获取任务ID
    
    Args:
        task_name: 任务名称
        mapper: 可选的TaskMapper实例，默认使用全局默认实例
    
    Returns:
        任务ID
    """
    if mapper is None:
        mapper = default_task_mapper
    return mapper.get_task_id(task_name)


def get_task_name(task_id: int, mapper: Optional[TaskMapper] = None) -> str:
    """
    便捷函数：获取任务名称
    
    Args:
        task_id: 任务ID
        mapper: 可选的TaskMapper实例，默认使用全局默认实例
    
    Returns:
        任务名称
    """
    if mapper is None:
        mapper = default_task_mapper
    return mapper.get_task_name(task_id)


def get_task_order_from_config(task_config_dir: str) -> Optional[int]:
    """
    从任务配置目录中提取任务顺序ID。
    
    用于continual learning中，从配置路径推断当前任务在序列中的位置。
    例如：configs/order1_configs/amazon -> 返回当前是第几个任务
    
    Args:
        task_config_dir: 任务配置目录路径
        
    Returns:
        任务顺序ID，如果无法确定则返回None
    """
    import os
    import json
    
    if not task_config_dir or not os.path.exists(task_config_dir):
        return None
    
    try:
        # 查找train_tasks.json文件
        train_config_path = os.path.join(task_config_dir, "train_tasks.json")
        if not os.path.exists(train_config_path):
            return None
            
        # 读取配置文件，找到非空的任务
        with open(train_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 统计当前配置中有多少个任务（非空的任务列表）
        active_tasks = []
        for task_type, datasets in config.items():
            if datasets:  # 非空列表
                active_tasks.append(task_type)
        
        # 如果只有一个活跃任务，这表明是continual learning的单任务训练
        if len(active_tasks) == 1:
            # 从目录结构推断顺序
            # 例如：从 configs/order1_configs/amazon 推断这是order1的某个任务
            path_parts = task_config_dir.replace('\\', '/').split('/')
            
            order_info = None
            dataset_name = None
            
            for i, part in enumerate(path_parts):
                if 'order' in part.lower() and 'config' in part.lower():
                    order_info = part
                    if i + 1 < len(path_parts):
                        dataset_name = path_parts[i + 1]
                    break
            
            if order_info and dataset_name:
                # 基于order和dataset确定任务顺序
                # 这里需要根据实际的任务执行顺序来确定
                # 暂时返回一个基于目录名的简单映射
                order_num = order_info.lower().replace('order', '').replace('_configs', '')
                
                # 根据数据集名称和order确定在continual learning序列中的位置
                # 基于图中实际的执行顺序进行映射
                task_order_map = {
                    # Order 1 sequence: dbpedia → amazon → yahoo → ag(agnews)
                    ('1', 'dbpedia'): 0,  # 第1个任务
                    ('1', 'amazon'): 1,   # 第2个任务
                    ('1', 'yahoo'): 2,    # 第3个任务
                    ('1', 'agnews'): 3,   # 第4个任务
                    
                    # Order 2 sequence: dbpedia → amazon → ag(agnews) → yahoo
                    ('2', 'dbpedia'): 0,  # 第1个任务
                    ('2', 'amazon'): 1,   # 第2个任务
                    ('2', 'agnews'): 2,   # 第3个任务
                    ('2', 'yahoo'): 3,    # 第4个任务
                    
                    # Order 3 sequence: yahoo → amazon → ag(agnews) → dbpedia
                    ('3', 'yahoo'): 0,    # 第1个任务
                    ('3', 'amazon'): 1,   # 第2个任务
                    ('3', 'agnews'): 2,   # 第3个任务
                    ('3', 'dbpedia'): 3,  # 第4个任务
                }
                
                return task_order_map.get((order_num, dataset_name), 0)
        
    except Exception as e:
        logger.warning(f"Failed to get task order from config {task_config_dir}: {e}")
    
    return None


if __name__ == "__main__":
    # 测试代码
    mapper = TaskMapper()
    
    print("默认任务映射:")
    for task in mapper.get_all_tasks():
        print(f"  {task} -> {mapper.get_task_id(task)}")
    
    # 测试获取任务ID
    test_tasks = ["SC", "amazon", "entity_extraction", "NLI"]
    print("\n测试任务ID获取:")
    for task in test_tasks:
        try:
            task_id = mapper.get_task_id(task)
            print(f"  {task} -> {task_id}")
        except ValueError as e:
            print(f"  {task} -> ERROR: {e}")
    
    # 测试任务顺序获取
    print("\n测试任务顺序获取:")
    test_configs = [
        # Order 1 tests: dbpedia → amazon → yahoo → agnews
        "configs/order1_configs/dbpedia",   # 应该是 0
        "configs/order1_configs/amazon",    # 应该是 1
        "configs/order1_configs/yahoo",     # 应该是 2
        "configs/order1_configs/agnews",    # 应该是 3
        
        # Order 2 tests: dbpedia → amazon → agnews → yahoo
        "configs/order2_configs/dbpedia",   # 应该是 0
        "configs/order2_configs/amazon",    # 应该是 1
        "configs/order2_configs/agnews",    # 应该是 2
        "configs/order2_configs/yahoo",     # 应该是 3
        
        # Order 3 tests: yahoo → amazon → agnews → dbpedia
        "configs/order3_configs/yahoo",     # 应该是 0
        "configs/order3_configs/amazon",    # 应该是 1
        "configs/order3_configs/agnews",    # 应该是 2
        "configs/order3_configs/dbpedia",   # 应该是 3
    ]
    for config_path in test_configs:
        order_id = get_task_order_from_config(config_path)
        expected_msg = ""
        if "order1" in config_path and "dbpedia" in config_path:
            expected_msg = " (期望: 0 - order1第1个任务)"
        elif "order1" in config_path and "amazon" in config_path:
            expected_msg = " (期望: 1 - order1第2个任务)"
        elif "order3" in config_path and "yahoo" in config_path:
            expected_msg = " (期望: 0 - order3第1个任务)"
        elif "order3" in config_path and "dbpedia" in config_path:
            expected_msg = " (期望: 3 - order3第4个任务)"
        print(f"  {config_path} -> task_id: {order_id}{expected_msg}")
