#!/usr/bin/env python3
"""
Advanced Football Prediction Model Trainer
Handles model training, validation, and optimization
Author: AI Assistant
Version: 1.0.0
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.integration import sklearn as optuna_sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Advanced model trainer for football predictions"""
    
    def __init__(self, config_path: str = None):
        """Initialize model trainer"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.best_params = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        
        # Model configurations
        self.model_configs = self._get_model_configs()
        
        logger.info("ModelTrainer initialized successfully")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'model_training': {
                'test_size': 0.2,
                'cross_validation_folds': 5,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 1
            },
            'hyperparameter_optimization': {
                'n_trials': 100,
                'timeout': 3600,
                'n_jobs': 4,
                'sampler': 'TPESampler'
            },
            'feature_selection': {
                'k_best': 50,
                'score_func': 'f_classif',
                'enable_selection': True
            },
            'model_validation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                'threshold_tuning': True,
                'calibration': True
            },
            'ensemble_methods': {
                'enable_voting': True,
                'enable_stacking': True,
                'enable_blending': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations with hyperparameter spaces"""
        return {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False]
                },
                'optuna_params': {
                    'n_estimators': ('int', 50, 500),
                    'max_depth': ('int', 5, 30),
                    'min_samples_split': ('int', 2, 20),
                    'min_samples_leaf': ('int', 1, 10),
                    'max_features': ('categorical', ['auto', 'sqrt', 'log2']),
                    'bootstrap': ('categorical', [True, False])
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'optuna_params': {
                    'n_estimators': ('int', 50, 500),
                    'learning_rate': ('float', 0.01, 0.3),
                    'max_depth': ('int', 3, 15),
                    'min_samples_split': ('int', 2, 20),
                    'min_samples_leaf': ('int', 1, 10),
                    'subsample': ('float', 0.6, 1.0)
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'optuna_params': {
                    'n_estimators': ('int', 50, 500),
                    'learning_rate': ('float', 0.01, 0.3),
                    'max_depth': ('int', 3, 15),
                    'min_child_weight': ('int', 1, 10),
                    'gamma': ('float', 0, 0.5),
                    'subsample': ('float', 0.6, 1.0),
                    'colsample_bytree': ('float', 0.6, 1.0)
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'max_depth': [-1, 5, 10, 15],
                    'min_child_samples': [20, 50, 100],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'optuna_params': {
                    'n_estimators': ('int', 50, 500),
                    'learning_rate': ('float', 0.01, 0.3),
                    'num_leaves': ('int', 20, 200),
                    'max_depth': ('int', -1, 20),
                    'min_child_samples': ('int', 10, 200),
                    'subsample': ('float', 0.6, 1.0),
                    'colsample_bytree': ('float', 0.6, 1.0)
                }
            },
            'catboost': {
                'model': CatBoostClassifier,
                'params': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128]
                },
                'optuna_params': {
                    'iterations': ('int', 50, 500),
                    'learning_rate': ('float', 0.01, 0.3),
                    'depth': ('int', 4, 12),
                    'l2_leaf_reg': ('float', 0.1, 10),
                    'border_count': ('int', 32, 255)
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [100, 200, 500]
                },
                'optuna_params': {
                    'C': ('float', 0.01, 100),
                    'penalty': ('categorical', ['l1', 'l2']),
                    'solver': ('categorical', ['liblinear', 'saga']),
                    'max_iter': ('int', 100, 1000)
                }
            },
            'svm': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'degree': [2, 3, 4]
                },
                'optuna_params': {
                    'C': ('float', 0.01, 100),
                    'kernel': ('categorical', ['linear', 'rbf', 'poly']),
                    'gamma': ('categorical', ['scale', 'auto']),
                    'degree': ('int', 2, 5)
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'sgd'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [200, 500, 1000]
                },
                'optuna_params': {
                    'hidden_layer_sizes': ('categorical', [(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                    'activation': ('categorical', ['relu', 'tanh']),
                    'solver': ('categorical', ['adam', 'sgd']),
                    'alpha': ('float', 0.0001, 0.1),
                    'learning_rate': ('categorical', ['constant', 'adaptive']),
                    'max_iter': ('int', 200, 2000)
                }
            }
        }
    
    def train_individual_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             model_name: str, optimize_hyperparameters: bool = True,
                             use_optuna: bool = True) -> Dict[str, Any]:
        """
        Train an individual model with optional hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            model_name: Name of the model to train
            optimize_hyperparameters: Whether to optimize hyperparameters
            use_optuna: Whether to use Optuna for optimization
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"Training {model_name} model...")
            
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            model_config = self.model_configs[model_name]
            
            # Create pipeline with preprocessing
            pipeline_steps = []
            
            # Add feature scaling
            if model_name in ['svm', 'neural_network', 'logistic_regression']:
                pipeline_steps.append(('scaler', StandardScaler()))
            
            # Add feature selection
            if self.config['feature_selection']['enable_selection']:
                score_func = self._get_score_function()
                pipeline_steps.append(('feature_selection', SelectKBest(
                    score_func=score_func,
                    k=self.config['feature_selection']['k_best']
                )))
            
            # Create base model
            if optimize_hyperparameters and use_optuna:
                best_params = self._optimize_with_optuna(
                    X_train, y_train, model_name, model_config['optuna_params']
                )
                model = model_config['model'](**best_params, random_state=42)
            else:
                model = model_config['model'](random_state=42)
            
            pipeline_steps.append(('model', model))
            
            # Create pipeline
            pipeline = Pipeline(pipeline_steps)
            
            # Train model
            if optimize_hyperparameters and not use_optuna:
                pipeline = self._grid_search_optimization(
                    pipeline, X_train, y_train, model_config['params']
                )
            else:
                pipeline.fit(X_train, y_train)
            
            # Evaluate model
            results = self._evaluate_model(pipeline, X_test, y_test, model_name)
            
            # Store model
            self.models[model_name] = pipeline
            
            logger.info(f"{model_name} training completed. Accuracy: {results['accuracy']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {'error': str(e), 'model_name': model_name}
    
    def _get_score_function(self) -> Callable:
        """Get feature selection score function"""
        score_func_name = self.config['feature_selection']['score_func']
        
        if score_func_name == 'f_classif':
            return f_classif
        elif score_func_name == 'mutual_info_classif':
            return mutual_info_classif
        else:
            return f_classif
    
    def _grid_search_optimization(self, pipeline: Pipeline, X_train: np.ndarray, 
                                y_train: np.ndarray, param_grid: Dict[str, List]) -> Pipeline:
        """Perform grid search hyperparameter optimization"""
        logger.info("Performing grid search optimization...")
        
        # Create parameter grid for pipeline
        pipeline_param_grid = {}
        for param, values in param_grid.items():
            pipeline_param_grid[f'model__{param}'] = values
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            pipeline_param_grid,
            cv=self.config['model_training']['cross_validation_folds'],
            scoring='accuracy',
            n_jobs=self.config['model_training']['n_jobs'],
            verbose=self.config['model_training']['verbose']
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _optimize_with_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                            model_name: str, param_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Optimizing {model_name} with Optuna...")
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param, (param_type, *values) in param_space.items():
                if param_type == 'int':
                    params[param] = trial.suggest_int(param, values[0], values[1])
                elif param_type == 'float':
                    params[param] = trial.suggest_float(param, values[0], values[1])
                elif param_type == 'categorical':
                    params[param] = trial.suggest_categorical(param, values[0])
            
            # Create model with sampled parameters
            model_class = self.model_configs[model_name]['model']
            model = model_class(**params, random_state=42)
            
            # Perform cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config['model_training']['cross_validation_folds'],
                scoring='accuracy',
                n_jobs=1
            )
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config['hyperparameter_optimization']['n_trials'],
            timeout=self.config['hyperparameter_optimization']['timeout'],
            n_jobs=1
        )
        
        logger.info(f"Best trial: {study.best_trial.params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def _evaluate_model(self, model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str) -> Dict[str, Any]:
        """Evaluate trained model"""
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {}
        metrics['model_name'] = model_name
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC (if probability predictions available)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                metrics['log_loss'] = log_loss(y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = None
                metrics['log_loss'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Store performance metrics
        self.performance_metrics[model_name] = metrics
        
        # Feature importance (if available)
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            self.feature_importance[model_name] = model.named_steps['model'].feature_importances_.tolist()
        elif hasattr(model.named_steps['model'], 'coef_'):
            self.feature_importance[model_name] = np.abs(model.named_steps['model'].coef_[0]).tolist()
        
        return metrics
    
    def train_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            trained_models: Dict[str, Pipeline]) -> Dict[str, Any]:
        """Train ensemble models"""
        logger.info("Training ensemble models...")
        
        ensemble_results = {}
        
        # Voting Classifier
        if self.config['ensemble_methods']['enable_voting']:
            voting_result = self._train_voting_classifier(
                X_train, y_train, X_test, y_test, trained_models
            )
            ensemble_results['voting_classifier'] = voting_result
        
        # Stacking Classifier
        if self.config['ensemble_methods']['enable_stacking']:
            stacking_result = self._train_stacking_classifier(
                X_train, y_train, X_test, y_test, trained_models
            )
            ensemble_results['stacking_classifier'] = stacking_result
        
        return ensemble_results
    
    def _train_voting_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               trained_models: Dict[str, Pipeline]) -> Dict[str, Any]:
        """Train voting classifier ensemble"""
        logger.info("Training voting classifier...")
        
        # Select best individual models for voting
        best_models = self._select_best_models(trained_models, n_models=5)
        
        # Create voting classifier
        estimators = [(name, model) for name, model in best_models.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=self.config['model_training']['n_jobs']
        )
        
        # Train voting classifier
        voting_clf.fit(X_train, y_train)
        
        # Evaluate
        results = self._evaluate_model(voting_clf, X_test, y_test, 'voting_classifier')
        
        # Store model
        self.models['voting_classifier'] = voting_clf
        
        return results
    
    def _train_stacking_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 trained_models: Dict[str, Pipeline]) -> Dict[str, Any]:
        """Train stacking classifier ensemble"""
        logger.info("Training stacking classifier...")
        
        from sklearn.ensemble import StackingClassifier
        
        # Select best individual models as base estimators
        base_estimators = [(name, model) for name, model in trained_models.items()]
        
        # Use logistic regression as meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=self.config['model_training']['cross_validation_folds'],
            n_jobs=self.config['model_training']['n_jobs']
        )
        
        # Train stacking classifier
        stacking_clf.fit(X_train, y_train)
        
        # Evaluate
        results = self._evaluate_model(stacking_clf, X_test, y_test, 'stacking_classifier')
        
        # Store model
        self.models['stacking_classifier'] = stacking_clf
        
        return results
    
    def _select_best_models(self, trained_models: Dict[str, Pipeline], 
                          n_models: int = 5) -> Dict[str, Pipeline]:
        """Select best performing models"""
        # Sort models by accuracy
        sorted_models = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        # Select top N models
        best_models = {}
        for model_name, _ in sorted_models[:n_models]:
            if model_name in trained_models:
                best_models[model_name] = trained_models[model_name]
        
        return best_models
    
    def perform_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str]) -> Dict[str, Any]:
        """Perform comprehensive feature selection"""
        logger.info("Performing feature selection...")
        
        selection_results = {}
        
        # Univariate feature selection
        univariate_results = self._univariate_feature_selection(X, y, feature_names)
        selection_results['univariate'] = univariate_results
        
        # Recursive feature elimination
        rfe_results = self._recursive_feature_elimination(X, y, feature_names)
        selection_results['rfe'] = rfe_results
        
        # Feature importance from models
        importance_results = self._model_based_feature_selection(X, y, feature_names)
        selection_results['model_based'] = importance_results
        
        # Correlation analysis
        correlation_results = self._correlation_analysis(X, y, feature_names)
        selection_results['correlation'] = correlation_results
        
        return selection_results
    
    def _univariate_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str]) -> Dict[str, Any]:
        """Perform univariate feature selection"""
        logger.info("Univariate feature selection...")
        
        # SelectKBest with f_classif
        selector = SelectKBest(score_func=f_classif, k='all')
        X_selected = selector.fit_transform(X, y)
        
        # Get scores and p-values
        scores = selector.scores_
        pvalues = selector.pvalues_
        
        # Create feature ranking
        feature_ranking = []
        for i, (name, score, pvalue) in enumerate(zip(feature_names, scores, pvalues)):
            feature_ranking.append({
                'feature': name,
                'score': score,
                'p_value': pvalue,
                'rank': i + 1
            })
        
        # Sort by score
        feature_ranking.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'method': 'univariate_f_classif',
            'feature_ranking': feature_ranking,
            'selected_features': [item['feature'] for item in feature_ranking[:20]],
            'scores': scores.tolist(),
            'p_values': pvalues.tolist()
        }
    
    def _recursive_feature_elimination(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str]) -> Dict[str, Any]:
        """Perform recursive feature elimination"""
        logger.info("Recursive feature elimination...")
        
        from sklearn.feature_selection import RFE
        
        # Use Random Forest as base estimator
        base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create RFE selector
        rfe = RFE(estimator=base_estimator, n_features_to_select=20, step=1)
        X_selected = rfe.fit_transform(X, y)
        
        # Get selected features
        selected_features = [name for name, selected in zip(feature_names, rfe.support_) if selected]
        
        # Get feature rankings
        feature_ranking = []
        for name, rank in zip(feature_names, rfe.ranking_):
            feature_ranking.append({
                'feature': name,
                'ranking': rank
            })
        
        feature_ranking.sort(key=lambda x: x['ranking'])
        
        return {
            'method': 'rfe_random_forest',
            'selected_features': selected_features,
            'feature_ranking': feature_ranking,
            'n_features_selected': len(selected_features)
        }
    
    def _model_based_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str]) -> Dict[str, Any]:
        """Perform model-based feature selection"""
        logger.info("Model-based feature selection...")
        
        # Train Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_scores = rf.feature_importances_
        
        # Create feature ranking
        feature_ranking = []
        for name, importance in zip(feature_names, importance_scores):
            feature_ranking.append({
                'feature': name,
                'importance': importance,
                'rank': 0  # Will be filled after sorting
            })
        
        # Sort by importance
        feature_ranking.sort(key=lambda x: x['importance'], reverse=True)
        
        # Update ranks
        for i, item in enumerate(feature_ranking):
            item['rank'] = i + 1
        
        return {
            'method': 'random_forest_importance',
            'feature_ranking': feature_ranking,
            'selected_features': [item['feature'] for item in feature_ranking[:20]],
            'importance_scores': importance_scores.tolist()
        }
    
    def _correlation_analysis(self, X: np.ndarray, y: np.ndarray,
                            feature_names: List[str]) -> Dict[str, Any]:
        """Perform correlation analysis"""
        logger.info("Correlation analysis...")
        
        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))  # Use absolute correlation
        
        # Create feature ranking
        feature_ranking = []
        for name, corr in zip(feature_names, correlations):
            feature_ranking.append({
                'feature': name,
                'correlation': corr,
                'rank': 0  # Will be filled after sorting
            })
        
        # Sort by correlation
        feature_ranking.sort(key=lambda x: x['correlation'], reverse=True)
        
        # Update ranks
        for i, item in enumerate(feature_ranking):
            item['rank'] = i + 1
        
        return {
            'method': 'correlation_with_target',
            'feature_ranking': feature_ranking,
            'selected_features': [item['feature'] for item in feature_ranking[:20]],
            'correlations': correlations
        }
    
    def validate_model(self, model: Pipeline, X_val: np.ndarray, y_val: np.ndarray,
                      model_name: str) -> Dict[str, Any]:
        """Comprehensive model validation"""
        logger.info(f"Validating {model_name}...")
        
        validation_results = {}
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_val, y_val,
            cv=self.config['model_training']['cross_validation_folds'],
            scoring='accuracy',
            n_jobs=self.config['model_training']['n_jobs']
        )
        
        validation_results['cross_validation'] = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        # Learning curves
        learning_curves = self._plot_learning_curves(model, X_val, y_val, model_name)
        validation_results['learning_curves'] = learning_curves
        
        # Validation curves
        validation_curves = self._plot_validation_curves(model, X_val, y_val, model_name)
        validation_results['validation_curves'] = validation_curves
        
        # Confusion matrix analysis
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        validation_results['confusion_matrix_analysis'] = self._analyze_confusion_matrix(cm)
        
        # Classification metrics by class
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)
            validation_results['probability_analysis'] = self._analyze_probabilities(y_val, y_pred_proba)
        
        return validation_results
    
    def _plot_learning_curves(self, model: Pipeline, X: np.ndarray, y: np.ndarray,
                            model_name: str) -> Dict[str, Any]:
        """Generate learning curves"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            cv=self.config['model_training']['cross_validation_folds'],
            n_jobs=self.config['model_training']['n_jobs'],
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist()
        }
    
    def _plot_validation_curves(self, model: Pipeline, X: np.ndarray, y: np.ndarray,
                              model_name: str) -> Dict[str, Any]:
        """Generate validation curves"""
        # This would typically generate actual plots
        # For now, return placeholder data
        return {
            'parameter_name': 'max_depth',
            'parameter_range': list(range(1, 21)),
            'train_scores': np.random.random(20).tolist(),
            'val_scores': np.random.random(20).tolist()
        }
    
    def _analyze_confusion_matrix(self, cm: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix"""
        if cm.shape[0] != cm.shape[1]:
            return {'error': 'Confusion matrix must be square'}
        
        n_classes = cm.shape[0]
        
        analysis = {
            'matrix': cm.tolist(),
            'true_positives': [],
            'false_positives': [],
            'true_negatives': [],
            'false_negatives': [],
            'class_metrics': []
        }
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            analysis['true_positives'].append(int(tp))
            analysis['false_positives'].append(int(fp))
            analysis['true_negatives'].append(int(tn))
            analysis['false_negatives'].append(int(fn))
            
            # Calculate class-specific metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            analysis['class_metrics'].append({
                'class': i,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        return analysis
    
    def _analyze_probabilities(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze probability predictions"""
        analysis = {
            'mean_probabilities': y_pred_proba.mean(axis=0).tolist(),
            'std_probabilities': y_pred_proba.std(axis=0).tolist(),
            'min_probabilities': y_pred_proba.min(axis=0).tolist(),
            'max_probabilities': y_pred_proba.max(axis=0).tolist(),
            'calibration': self._calculate_calibration(y_true, y_pred_proba)
        }
        
        return analysis
    
    def _calculate_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate prediction calibration"""
        from sklearn.calibration import calibration_curve
        
        calibration_data = []
        
        for i in range(y_pred_proba.shape[1]):
            prob_true, prob_pred = calibration_curve(
                y_true == i, y_pred_proba[:, i], n_bins=10
            )
            
            calibration_data.append({
                'class': i,
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            })
        
        return calibration_data
    
    def save_models(self, save_path: str):
        """Save all trained models"""
        logger.info(f"Saving models to {save_path}")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save individual models
            for model_name, model in self.models.items():
                model_path = os.path.join(save_path, f"{model_name}.pkl")
                joblib.dump(model, model_path)
            
            # Save performance metrics
            metrics_path = os.path.join(save_path, 'performance_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            # Save feature importance
            importance_path = os.path.join(save_path, 'feature_importance.json')
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            # Save configuration
            config_path = os.path.join(save_path, 'training_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, load_path: str):
        """Load trained models"""
        logger.info(f"Loading models from {load_path}")
        
        try:
            # Load individual models
            for filename in os.listdir(load_path):
                if filename.endswith('.pkl') and filename != 'performance_metrics.json':
                    model_name = filename[:-4]  # Remove .pkl
                    model_path = os.path.join(load_path, filename)
                    self.models[model_name] = joblib.load(model_path)
            
            # Load performance metrics
            metrics_path = os.path.join(load_path, 'performance_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.performance_metrics = json.load(f)
            
            # Load feature importance
            importance_path = os.path.join(load_path, 'feature_importance.json')
            if os.path.exists(importance_path):
                with open(importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def compare_models(self, model_names: List[str] = None) -> pd.DataFrame:
        """Compare performance of multiple models"""
        if model_names is None:
            model_names = list(self.performance_metrics.keys())
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name in self.performance_metrics:
                metrics = self.performance_metrics[model_name]
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0)
                })
        
        return pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, Pipeline]:
        """Get the best performing model"""
        if not self.performance_metrics:
            raise ValueError("No models have been trained yet")
        
        best_model_name = max(
            self.performance_metrics.keys(),
            key=lambda x: self.performance_metrics[x].get(metric, 0)
        )
        
        if best_model_name not in self.models:
            raise ValueError(f"Best model {best_model_name} not found in trained models")
        
        return best_model_name, self.models[best_model_name]

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.random((1000, 50))
    y = np.random.randint(0, 3, 1000)  # 3 classes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train individual model
    results = trainer.train_individual_model(
        X_train, y_train, X_test, y_test, 'random_forest', optimize_hyperparameters=True
    )
    
    print(f"Random Forest Results: {json.dumps(results, indent=2)}")
    
    # Compare models
    comparison = trainer.compare_models()
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))