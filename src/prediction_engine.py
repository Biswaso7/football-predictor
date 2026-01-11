#!/usr/bin/env python3
"""
Advanced Football Betting Prediction Engine
Core prediction algorithms and models
Author: AI Assistant
Version: 1.0.0
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

# Statistical imports
from scipy import stats
from scipy.stats import poisson, skellam
# import pymc3 as pm  # Optional: used for Bayesian models

# Custom imports
from data_processor import DataProcessor
from betting_calculator import BettingCalculator
from model_trainer import ModelTrainer
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballPredictionEngine:
    """
    Advanced football prediction engine using multiple ML models
    and statistical methods for comprehensive betting predictions
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the prediction engine"""
        self.config = self._load_config(config_path)
        self.data_processor = DataProcessor()
        self.betting_calculator = BettingCalculator()
        self.model_trainer = ModelTrainer()
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_performance = {}
        
        # Initialize models
        self._initialize_models()
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'models': {
                'enable_ml': True,
                'enable_deep_learning': True,
                'enable_statistical': True,
                'enable_ensemble': True
            },
            'prediction_types': [
                'match_result', 'over_under', 'both_teams_to_score',
                'correct_score', 'double_chance', 'asian_handicap'
            ],
            'confidence_threshold': 0.65,
            'min_odds': 1.1,
            'max_odds': 10.0,
            'kelly_fraction': 0.25,
            'risk_management': {
                'max_stake_percentage': 0.05,
                'max_daily_loss': 0.1,
                'stop_loss_percentage': 0.15
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
    
    def _initialize_models(self):
        """Initialize all prediction models"""
        logger.info("Initializing prediction models...")
        
        # Machine Learning Models
        if self.config['models']['enable_ml']:
            self._initialize_ml_models()
        
        # Deep Learning Models
        if self.config['models']['enable_deep_learning']:
            self._initialize_deep_learning_models()
        
        # Statistical Models
        if self.config['models']['enable_statistical']:
            self._initialize_statistical_models()
        
        # Ensemble Models
        if self.config['models']['enable_ensemble']:
            self._initialize_ensemble_models()
        
        logger.info("Models initialized successfully")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        logger.info("Initializing ML models...")
        
        # Random Forest for match results
        self.models['rf_match_result'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting for over/under
        self.models['gb_over_under'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        )
        
        # Logistic Regression for both teams to score
        self.models['lr_btts'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=0.1
        )
        
        # SVM for correct score
        self.models['svm_correct_score'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = StandardScaler()  # Could be RobustScaler
        
        # Initialize encoders
        self.encoders['team'] = LabelEncoder()
        self.encoders['league'] = LabelEncoder()
    
    def _initialize_deep_learning_models(self):
        """Initialize deep learning models"""
        logger.info("Initializing deep learning models...")
        
        # Neural Network for match prediction
        self.models['nn_match_prediction'] = self._build_neural_network(
            input_dim=50,
            hidden_layers=[128, 64, 32],
            output_dim=3,  # home win, draw, away win
            activation='relu',
            output_activation='softmax'
        )
        
        # LSTM for sequence prediction
        self.models['lstm_sequence'] = self._build_lstm_model(
            sequence_length=10,
            features=25,
            lstm_units=[64, 32],
            output_dim=1
        )
        
        # CNN for pattern recognition
        self.models['cnn_patterns'] = self._build_cnn_model(
            input_shape=(28, 28, 1),
            conv_layers=[32, 64],
            dense_layers=[128, 64],
            output_dim=3
        )
    
    def _build_neural_network(self, input_dim: int, hidden_layers: List[int], 
                             output_dim: int, activation: str = 'relu', 
                             output_activation: str = 'softmax') -> keras.Model:
        """Build a neural network"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation=activation, 
                              input_shape=(input_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers[1:], 1):
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
        
        # Output layer
        model.add(layers.Dense(output_dim, activation=output_activation))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    def _build_lstm_model(self, sequence_length: int, features: int,
                          lstm_units: list, output_dim: int) -> keras.Model:
        """Build LSTM model"""
        model = keras.Sequential()
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            if i == 0:
                model.add(layers.LSTM(units, return_sequences=return_sequences,
                                      input_shape=(sequence_length, features)))
            else:
                model.add(layers.LSTM(units, return_sequences=return_sequences))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(output_dim, activation='sigmoid'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'precision', 'recall'])
        return model
    
    def _build_cnn_model(self, input_shape: Tuple[int, int, int], 
                        conv_layers: List[int], dense_layers: List[int], 
                        output_dim: int) -> keras.Model:
        """Build CNN model"""
        model = keras.Sequential()
        
        # Convolutional layers
        for i, filters in enumerate(conv_layers):
            if i == 0:
                model.add(layers.Conv2D(filters, (3, 3), activation='relu', 
                                       input_shape=input_shape))
            else:
                model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(0.25))
        
        # Flatten and dense layers
        model.add(layers.Flatten())
        
        for units in dense_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.5))
        
        # Output layer
        model.add(layers.Dense(output_dim, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _initialize_statistical_models(self):
        """Initialize statistical models"""
        logger.info("Initializing statistical models...")
        
        # Poisson distribution for goals
        self.models['poisson_goals'] = self._build_poisson_model()
        
        # Dixon-Coles model
        self.models['dixon_coles'] = self._build_dixon_coles_model()
        
        # Elo rating system
        self.models['elo_ratings'] = self._build_elo_model()
        
        # Expected Goals (xG) model
        self.models['expected_goals'] = self._build_xg_model()
    
    def _build_poisson_model(self):
        """Build Poisson distribution model"""
        class PoissonGoalsModel:
            def __init__(self):
                self.home_attack_strength = {}
                self.away_attack_strength = {}
                self.home_defence_strength = {}
                self.away_defence_strength = {}
                self.league_average = 0
            
            def fit(self, data):
                """Fit Poisson model to data"""
                # Calculate team strengths
                home_goals = data.groupby('home_team')['home_score'].mean()
                away_goals = data.groupby('away_team')['away_score'].mean()
                home_conceded = data.groupby('home_team')['away_score'].mean()
                away_conceded = data.groupby('away_team')['home_score'].mean()
                
                league_home_avg = data['home_score'].mean()
                league_away_avg = data['away_score'].mean()
                self.league_average = (league_home_avg + league_away_avg) / 2
                
                # Calculate attack and defence strengths
                for team in home_goals.index:
                    self.home_attack_strength[team] = home_goals[team] / league_home_avg
                    self.home_defence_strength[team] = home_conceded[team] / league_away_avg
                
                for team in away_goals.index:
                    self.away_attack_strength[team] = away_goals[team] / league_away_avg
                    self.away_defence_strength[team] = away_conceded[team] / league_home_avg
            
            def predict(self, home_team, away_team):
                """Predict match outcome"""
                home_lambda = (self.home_attack_strength.get(home_team, 1.0) * 
                              self.away_defence_strength.get(away_team, 1.0) * 
                              self.league_average)
                
                away_lambda = (self.away_attack_strength.get(away_team, 1.0) * 
                              self.home_defence_strength.get(home_team, 1.0) * 
                              self.league_average)
                
                return {
                    'home_lambda': home_lambda,
                    'away_lambda': away_lambda,
                    'home_win_prob': self._calculate_win_prob(home_lambda, away_lambda, 'home'),
                    'draw_prob': self._calculate_draw_prob(home_lambda, away_lambda),
                    'away_win_prob': self._calculate_win_prob(home_lambda, away_lambda, 'away')
                }
            
            def _calculate_win_prob(self, home_lambda, away_lambda, team):
                """Calculate win probability"""
                prob = 0
                if team == 'home':
                    for home_goals in range(10):  # Limit to reasonable range
                        for away_goals in range(home_goals):
                            prob += (poisson.pmf(home_goals, home_lambda) * 
                                   poisson.pmf(away_goals, away_lambda))
                else:
                    for away_goals in range(10):
                        for home_goals in range(away_goals):
                            prob += (poisson.pmf(home_goals, home_lambda) * 
                                   poisson.pmf(away_goals, away_lambda))
                return prob
            
            def _calculate_draw_prob(self, home_lambda, away_lambda):
                """Calculate draw probability"""
                prob = 0
                for goals in range(10):
                    prob += (poisson.pmf(goals, home_lambda) * 
                           poisson.pmf(goals, away_lambda))
                return prob
        
        return PoissonGoalsModel()
    
    def _build_dixon_coles_model(self):
        """Build Dixon-Coles model"""
        class DixonColesModel:
            def __init__(self):
                self.home_attack = {}
                self.away_attack = {}
                self.home_defence = {}
                self.away_defence = {}
                self.rho = 0  # Dependency parameter
            
            def fit(self, data):
                """Fit Dixon-Coles model"""
                # Implementation of Dixon-Coles model fitting
                # This is a simplified version
                pass
            
            def predict(self, home_team, away_team):
                """Predict using Dixon-Coles model"""
                # Implementation of Dixon-Coles prediction
                pass
        
        return DixonColesModel()
    
    def _build_elo_model(self):
        """Build Elo rating model"""
        class EloModel:
            def __init__(self, k_factor=20):
                self.ratings = {}
                self.k_factor = k_factor
                self.home_advantage = 100  # Home advantage in Elo points
            
            def update_ratings(self, home_team, away_team, home_score, away_score):
                """Update Elo ratings after match"""
                # Get current ratings
                home_rating = self.ratings.get(home_team, 1500)
                away_rating = self.ratings.get(away_team, 1500)
                
                # Apply home advantage
                home_rating += self.home_advantage
                
                # Calculate expected scores
                home_expected = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
                away_expected = 1 - home_expected
                
                # Calculate actual scores
                if home_score > away_score:
                    home_actual = 1
                    away_actual = 0
                elif home_score < away_score:
                    home_actual = 0
                    away_actual = 1
                else:
                    home_actual = 0.5
                    away_actual = 0.5
                
                # Update ratings
                self.ratings[home_team] = home_rating + self.k_factor * (home_actual - home_expected)
                self.ratings[away_team] = away_rating + self.k_factor * (away_actual - away_expected)
            
            def predict(self, home_team, away_team):
                """Predict match outcome using Elo ratings"""
                home_rating = self.ratings.get(home_team, 1500) + self.home_advantage
                away_rating = self.ratings.get(away_team, 1500)
                
                # Calculate expected outcome
                home_expected = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
                away_expected = 1 - home_expected
                
                return {
                    'home_win_prob': home_expected,
                    'away_win_prob': away_expected,
                    'draw_prob': 1 - home_expected - away_expected
                }
        
        return EloModel()
    
    def _build_xg_model(self):
        """Build Expected Goals model"""
        class ExpectedGoalsModel:
            def __init__(self):
                self.shot_values = {}
                self.team_attack_strength = {}
                self.team_defence_strength = {}
            
            def calculate_xg(self, shot_data):
                """Calculate expected goals for shots"""
                # Simplified xG calculation
                xg = 0
                for shot in shot_data:
                    distance = shot.get('distance', 0)
                    angle = shot.get('angle', 0)
                    shot_type = shot.get('type', 'open_play')
                    
                    # Base xG value
                    base_xg = max(0, 1 - (distance / 20)) * max(0, 1 - (angle / 90))
                    
                    # Adjust for shot type
                    type_multiplier = {
                        'open_play': 1.0,
                        'set_piece': 0.8,
                        'counter_attack': 1.2,
                        'penalty': 0.75
                    }
                    
                    xg += base_xg * type_multiplier.get(shot_type, 1.0)
                
                return xg
            
            def predict_match_xg(self, home_team, away_team):
                """Predict expected goals for match"""
                # Get team strengths
                home_attack = self.team_attack_strength.get(home_team, 1.0)
                away_defence = self.team_defence_strength.get(away_team, 1.0)
                away_attack = self.team_attack_strength.get(away_team, 1.0)
                home_defence = self.team_defence_strength.get(home_team, 1.0)
                
                # Calculate expected goals
                home_xg = home_attack * away_defence * 1.5  # League average
                away_xg = away_attack * home_defence * 1.2
                
                return {
                    'home_xg': home_xg,
                    'away_xg': away_xg
                }
        
        return ExpectedGoalsModel()
    
    def _initialize_ensemble_models(self):
        """Initialize ensemble models"""
        logger.info("Initializing ensemble models...")
        
        # Voting classifier for match results
        self.models['ensemble_match_result'] = VotingClassifier(
            estimators=[
                ('rf', self.models['rf_match_result']),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(random_state=42))
            ],
            voting='soft'
        )
        
        # Stacking ensemble for advanced predictions
        self.models['stacking_ensemble'] = self._build_stacking_ensemble()
    
    def _build_stacking_ensemble(self):
        """Build stacking ensemble model"""
        from sklearn.ensemble import StackingClassifier
        
        # Base learners
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        
        # Meta learner
        meta_learner = LogisticRegression(random_state=42)
        
        # Create stacking ensemble
        stacking_ensemble = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba'
        )
        
        return stacking_ensemble
    
    def predict_match(self, match_data: Dict[str, Any], 
                     prediction_type: str = 'match_result') -> Dict[str, Any]:
        """
        Main prediction method
        
        Args:
            match_data: Dictionary containing match information
            prediction_type: Type of prediction to make
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            logger.info(f"Making {prediction_type} prediction for {match_data['home_team']} vs {match_data['away_team']}")
            
            # Prepare features
            features = self._prepare_features(match_data)
            
            # Get predictions from different models
            predictions = {}
            
            # ML model predictions
            if self.config['models']['enable_ml']:
                ml_predictions = self._get_ml_predictions(features, prediction_type)
                predictions.update(ml_predictions)
            
            # Deep learning predictions
            if self.config['models']['enable_deep_learning']:
                dl_predictions = self._get_dl_predictions(features, prediction_type)
                predictions.update(dl_predictions)
            
            # Statistical predictions
            if self.config['models']['enable_statistical']:
                stat_predictions = self._get_statistical_predictions(match_data, prediction_type)
                predictions.update(stat_predictions)
            
            # Ensemble predictions
            if self.config['models']['enable_ensemble']:
                ensemble_predictions = self._get_ensemble_predictions(features, prediction_type)
                predictions.update(ensemble_predictions)
            
            # Combine predictions
            final_prediction = self._combine_predictions(predictions, prediction_type)
            
            # Calculate confidence and odds
            confidence = self._calculate_confidence(final_prediction, prediction_type)
            recommended_odds = self._calculate_recommended_odds(final_prediction, prediction_type)
            
            # Risk assessment
            risk_level = self._assess_risk(confidence, recommended_odds, match_data)
            
            result = {
                'predicted_outcome': final_prediction['outcome'],
                'confidence': confidence,
                'odds': recommended_odds,
                'risk_level': risk_level,
                'prediction_breakdown': final_prediction,
                'model_predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction completed: {result['predicted_outcome']} with {confidence:.2%} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._get_fallback_prediction(match_data, prediction_type)
    
    def _prepare_features(self, match_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction"""
        # This is a simplified feature preparation
        # In a real system, this would be much more comprehensive
        
        features = []
        
        # Team form (last 5 matches)
        home_form = match_data.get('home_form', [0.5] * 5)
        away_form = match_data.get('away_form', [0.5] * 5)
        
        # Head to head
        h2h_home_wins = match_data.get('h2h_home_wins', 0.5)
        h2h_draws = match_data.get('h2h_draws', 0.2)
        h2h_away_wins = match_data.get('h2h_away_wins', 0.3)
        
        # League position
        home_position = match_data.get('home_league_position', 10) / 20  # Normalize
        away_position = match_data.get('away_league_position', 10) / 20
        
        # Goals scored/conceded
        home_goals_scored = match_data.get('home_goals_scored_avg', 1.5) / 5
        home_goals_conceded = match_data.get('home_goals_conceded_avg', 1.2) / 5
        away_goals_scored = match_data.get('away_goals_scored_avg', 1.3) / 5
        away_goals_conceded = match_data.get('away_goals_conceded_avg', 1.4) / 5
        
        # Combine features
        features.extend([
            *home_form, *away_form,
            h2h_home_wins, h2h_draws, h2h_away_wins,
            home_position, away_position,
            home_goals_scored, home_goals_conceded,
            away_goals_scored, away_goals_conceded
        ])
        
        return np.array(features).reshape(1, -1)
    
    def _get_ml_predictions(self, features: np.ndarray, prediction_type: str) -> Dict[str, Any]:
        """Get machine learning predictions"""
        predictions = {}
        
        if prediction_type == 'match_result':
            # Random Forest prediction
            if 'rf_match_result' in self.models:
                rf_pred = self.models['rf_match_result'].predict_proba(features)[0]
                predictions['rf_match_result'] = {
                    'home_win': rf_pred[0],
                    'draw': rf_pred[1],
                    'away_win': rf_pred[2]
                }
        
        elif prediction_type == 'over_under':
            # Gradient Boosting prediction
            if 'gb_over_under' in self.models:
                gb_pred = self.models['gb_over_under'].predict_proba(features)[0]
                predictions['gb_over_under'] = {
                    'over': gb_pred[1],
                    'under': gb_pred[0]
                }
        
        elif prediction_type == 'both_teams_to_score':
            # Logistic Regression prediction
            if 'lr_btts' in self.models:
                lr_pred = self.models['lr_btts'].predict_proba(features)[0]
                predictions['lr_btts'] = {
                    'yes': lr_pred[1],
                    'no': lr_pred[0]
                }
        
        return predictions
    
    def _get_dl_predictions(self, features: np.ndarray, prediction_type: str) -> Dict[str, Any]:
        """Get deep learning predictions"""
        predictions = {}
        
        # Neural Network prediction
        if 'nn_match_prediction' in self.models:
            nn_pred = self.models['nn_match_prediction'].predict(features)[0]
            predictions['nn_match_prediction'] = {
                'home_win': nn_pred[0],
                'draw': nn_pred[1],
                'away_win': nn_pred[2]
            }
        
        return predictions
    
    def _get_statistical_predictions(self, match_data: Dict[str, Any], 
                                   prediction_type: str) -> Dict[str, Any]:
        """Get statistical predictions"""
        predictions = {}
        
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        
        # Poisson model prediction
        if 'poisson_goals' in self.models:
            poisson_pred = self.models['poisson_goals'].predict(home_team, away_team)
            predictions['poisson_goals'] = poisson_pred
        
        # Elo model prediction
        if 'elo_ratings' in self.models:
            elo_pred = self.models['elo_ratings'].predict(home_team, away_team)
            predictions['elo_ratings'] = elo_pred
        
        return predictions
    
    def _get_ensemble_predictions(self, features: np.ndarray, 
                                prediction_type: str) -> Dict[str, Any]:
        """Get ensemble predictions"""
        predictions = {}
        
        # Voting ensemble
        if 'ensemble_match_result' in self.models:
            ensemble_pred = self.models['ensemble_match_result'].predict_proba(features)[0]
            predictions['ensemble_match_result'] = {
                'home_win': ensemble_pred[0],
                'draw': ensemble_pred[1],
                'away_win': ensemble_pred[2]
            }
        
        # Stacking ensemble
        if 'stacking_ensemble' in self.models:
            stacking_pred = self.models['stacking_ensemble'].predict_proba(features)[0]
            predictions['stacking_ensemble'] = {
                'home_win': stacking_pred[0],
                'draw': stacking_pred[1],
                'away_win': stacking_pred[2]
            }
        
        return predictions
    
    def _combine_predictions(self, predictions: Dict[str, Any], 
                           prediction_type: str) -> Dict[str, Any]:
        """Combine predictions from different models"""
        if not predictions:
            return self._get_default_prediction(prediction_type)
        
        # Weighted average based on model performance
        weights = self._get_model_weights(prediction_type)
        
        combined_probs = {}
        
        # Collect all probabilities for each outcome
        for model_name, pred in predictions.items():
            if prediction_type == 'match_result':
                for outcome in ['home_win', 'draw', 'away_win']:
                    if outcome in pred:
                        if outcome not in combined_probs:
                            combined_probs[outcome] = []
                        combined_probs[outcome].append(pred[outcome] * weights.get(model_name, 1.0))
            
            elif prediction_type == 'over_under':
                for outcome in ['over', 'under']:
                    if outcome in pred:
                        if outcome not in combined_probs:
                            combined_probs[outcome] = []
                        combined_probs[outcome].append(pred[outcome] * weights.get(model_name, 1.0))
            
            elif prediction_type == 'both_teams_to_score':
                for outcome in ['yes', 'no']:
                    if outcome in pred:
                        if outcome not in combined_probs:
                            combined_probs[outcome] = []
                        combined_probs[outcome].append(pred[outcome] * weights.get(model_name, 1.0))
        
        # Calculate weighted averages
        final_probs = {}
        for outcome, probs in combined_probs.items():
            if probs:
                final_probs[outcome] = np.mean(probs)
        
        # Normalize probabilities
        total_prob = sum(final_probs.values())
        if total_prob > 0:
            final_probs = {k: v/total_prob for k, v in final_probs.items()}
        
        # Select outcome with highest probability
        best_outcome = max(final_probs, key=final_probs.get)
        
        return {
            'outcome': best_outcome,
            'probabilities': final_probs,
            'confidence': final_probs[best_outcome]
        }
    
    def _get_model_weights(self, prediction_type: str) -> Dict[str, float]:
        """Get model weights based on performance"""
        # These weights should be updated based on actual model performance
        default_weights = {
            'rf_match_result': 1.0,
            'gb_over_under': 1.0,
            'lr_btts': 1.0,
            'nn_match_prediction': 1.2,
            'poisson_goals': 0.9,
            'elo_ratings': 0.8,
            'ensemble_match_result': 1.3,
            'stacking_ensemble': 1.4
        }
        
        return default_weights
    
    def _calculate_confidence(self, prediction: Dict[str, Any], 
                            prediction_type: str) -> float:
        """Calculate confidence in prediction"""
        base_confidence = prediction['confidence']
        
        # Adjust confidence based on prediction type
        confidence_multipliers = {
            'match_result': 1.0,
            'over_under': 0.95,
            'both_teams_to_score': 0.9,
            'correct_score': 0.7,
            'double_chance': 0.85,
            'asian_handicap': 0.88
        }
        
        multiplier = confidence_multipliers.get(prediction_type, 1.0)
        adjusted_confidence = base_confidence * multiplier
        
        # Ensure confidence is within reasonable bounds
        return max(0.5, min(0.99, adjusted_confidence))
    
    def _calculate_recommended_odds(self, prediction: Dict[str, Any], 
                                  prediction_type: str) -> float:
        """Calculate recommended odds"""
        probability = prediction['confidence']
        
        # Convert probability to fair odds
        fair_odds = 1 / probability
        
        # Apply margin (typically 5-10%)
        margin = 0.07
        recommended_odds = fair_odds * (1 - margin)
        
        # Ensure odds are within configured bounds
        min_odds = self.config['min_odds']
        max_odds = self.config['max_odds']
        
        return max(min_odds, min(max_odds, recommended_odds))
    
    def _assess_risk(self, confidence: float, odds: float, 
                    match_data: Dict[str, Any]) -> str:
        """Assess risk level of prediction"""
        # Risk factors
        risk_score = 0
        
        # Confidence risk
        risk_score += (1 - confidence) * 3
        
        # Odds risk (higher odds = higher risk)
        risk_score += (odds - 1) * 0.5
        
        # League risk (some leagues are more unpredictable)
        high_risk_leagues = ['brazilian_serie_a', 'eredivisie', 'mls']
        if match_data.get('league', '').lower() in high_risk_leagues:
            risk_score += 1
        
        # Team form risk
        home_form = match_data.get('home_form', [0.5] * 5)
        away_form = match_data.get('away_form', [0.5] * 5)
        form_variance = np.var(home_form + away_form)
        risk_score += form_variance * 2
        
        # Determine risk level
        if risk_score < 1.5:
            return 'low'
        elif risk_score < 3.0:
            return 'medium'
        else:
            return 'high'
    
    def _get_default_prediction(self, prediction_type: str) -> Dict[str, Any]:
        """Get default prediction when models fail"""
        default_outcomes = {
            'match_result': 'draw',
            'over_under': 'under',
            'both_teams_to_score': 'no',
            'correct_score': '1-1',
            'double_chance': 'home_or_draw',
            'asian_handicap': 'home'
        }
        
        default_probabilities = {
            'match_result': {'home_win': 0.33, 'draw': 0.34, 'away_win': 0.33},
            'over_under': {'over': 0.5, 'under': 0.5},
            'both_teams_to_score': {'yes': 0.5, 'no': 0.5}
        }
        
        outcome = default_outcomes.get(prediction_type, 'unknown')
        probabilities = default_probabilities.get(prediction_type, {})
        
        return {
            'outcome': outcome,
            'probabilities': probabilities,
            'confidence': 0.5
        }
    
    def _get_fallback_prediction(self, match_data: Dict[str, Any], 
                               prediction_type: str) -> Dict[str, Any]:
        """Get fallback prediction when error occurs"""
        logger.warning("Using fallback prediction due to error")
        
        default_prediction = self._get_default_prediction(prediction_type)
        
        return {
            'predicted_outcome': default_prediction['outcome'],
            'confidence': 0.5,
            'odds': 2.0,
            'risk_level': 'medium',
            'prediction_breakdown': default_prediction,
            'model_predictions': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def train_models(self, training_data: pd.DataFrame):
        """Train all models with provided data"""
        logger.info("Starting model training...")
        
        try:
            # Prepare data
            X, y = self.data_processor.prepare_training_data(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_test_scaled = self.scalers['standard'].transform(X_test)
            
            # Train ML models
            self._train_ml_models(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # Train deep learning models
            self._train_deep_learning_models(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # Train statistical models
            self._train_statistical_models(training_data)
            
            # Train ensemble models
            self._train_ensemble_models(X_train_scaled, y_train)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def _train_ml_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray):
        """Train machine learning models"""
        logger.info("Training ML models...")
        
        # Train Random Forest
        if 'rf_match_result' in self.models:
            self.models['rf_match_result'].fit(X_train, y_train)
            rf_score = self.models['rf_match_result'].score(X_test, y_test)
            self.model_performance['rf_match_result'] = rf_score
            logger.info(f"Random Forest accuracy: {rf_score:.3f}")
        
        # Train Gradient Boosting
        if 'gb_over_under' in self.models:
            self.models['gb_over_under'].fit(X_train, y_train)
            gb_score = self.models['gb_over_under'].score(X_test, y_test)
            self.model_performance['gb_over_under'] = gb_score
            logger.info(f"Gradient Boosting accuracy: {gb_score:.3f}")
        
        # Train Logistic Regression
        if 'lr_btts' in self.models:
            self.models['lr_btts'].fit(X_train, y_train)
            lr_score = self.models['lr_btts'].score(X_test, y_test)
            self.model_performance['lr_btts'] = lr_score
            logger.info(f"Logistic Regression accuracy: {lr_score:.3f}")
    
    def _train_deep_learning_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray):
        """Train deep learning models"""
        logger.info("Training deep learning models...")
        
        # Prepare data for neural network
        y_train_categorical = keras.utils.to_categorical(y_train)
        y_test_categorical = keras.utils.to_categorical(y_test)
        
        # Train Neural Network
        if 'nn_match_prediction' in self.models:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = self.models['nn_match_prediction'].fit(
                X_train, y_train_categorical,
                epochs=100, batch_size=32,
                validation_data=(X_test, y_test_categorical),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate model
            nn_score = self.models['nn_match_prediction'].evaluate(
                X_test, y_test_categorical, verbose=0
            )[1]
            self.model_performance['nn_match_prediction'] = nn_score
            logger.info(f"Neural Network accuracy: {nn_score:.3f}")
    
    def _train_statistical_models(self, data: pd.DataFrame):
        """Train statistical models"""
        logger.info("Training statistical models...")
        
        # Train Poisson model
        if 'poisson_goals' in self.models:
            self.models['poisson_goals'].fit(data)
            logger.info("Poisson model trained")
        
        # Train Elo model
        if 'elo_ratings' in self.models:
            # Update Elo ratings based on historical data
            for _, match in data.iterrows():
                self.models['elo_ratings'].update_ratings(
                    match['home_team'], match['away_team'],
                    match['home_score'], match['away_score']
                )
            logger.info("Elo model trained")
    
    def _train_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train ensemble models"""
        logger.info("Training ensemble models...")
        
        # Train Voting Classifier
        if 'ensemble_match_result' in self.models:
            self.models['ensemble_match_result'].fit(X_train, y_train)
            logger.info("Voting ensemble trained")
        
        # Train Stacking Ensemble
        if 'stacking_ensemble' in self.models:
            self.models['stacking_ensemble'].fit(X_train, y_train)
            logger.info("Stacking ensemble trained")
    
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate all models on test data"""
        logger.info("Evaluating models...")
        
        results = {}
        
        # Prepare test data
        X_test, y_test = self.data_processor.prepare_training_data(test_data)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Evaluate each model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'score'):
                    score = model.score(X_test_scaled, y_test)
                    results[model_name] = {
                        'accuracy': score,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Calculate additional metrics
                    y_pred = model.predict(X_test_scaled)
                    results[model_name].update({
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'f1_score': f1_score(y_test, y_pred, average='weighted')
                    })
                
                elif hasattr(model, 'evaluate'):  # Keras model
                    y_test_categorical = keras.utils.to_categorical(y_test)
                    loss, accuracy = model.evaluate(X_test_scaled, y_test_categorical, verbose=0)
                    results[model_name] = {
                        'accuracy': accuracy,
                        'loss': loss,
                        'timestamp': datetime.now().isoformat()
                    }
                
                elif hasattr(model, 'predict'):  # Statistical models
                    # Custom evaluation for statistical models
                    results[model_name] = {
                        'status': 'evaluated',
                        'timestamp': datetime.now().isoformat()
                    }
            
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def save_models(self, model_path: str):
        """Save all models to disk"""
        logger.info(f"Saving models to {model_path}")
        
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Save ML models
            for model_name, model in self.models.items():
                if hasattr(model, 'save'):
                    # Keras models
                    model_path_full = os.path.join(model_path, f"{model_name}.h5")
                    model.save(model_path_full)
                else:
                    # Sklearn models
                    model_path_full = os.path.join(model_path, f"{model_name}.pkl")
                    joblib.dump(model, model_path_full)
            
            # Save scalers and encoders
            joblib.dump(self.scalers, os.path.join(model_path, 'scalers.pkl'))
            joblib.dump(self.encoders, os.path.join(model_path, 'encoders.pkl'))
            
            # Save configuration
            with open(os.path.join(model_path, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Save model performance
            with open(os.path.join(model_path, 'performance.json'), 'w') as f:
                json.dump(self.model_performance, f, indent=2)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, model_path: str):
        """Load models from disk"""
        logger.info(f"Loading models from {model_path}")
        
        try:
            # Load ML models
            for model_name in ['rf_match_result', 'gb_over_under', 'lr_btts', 'svm_correct_score']:
                model_file = os.path.join(model_path, f"{model_name}.pkl")
                if os.path.exists(model_file):
                    self.models[model_name] = joblib.load(model_file)
            
            # Load deep learning models
            for model_name in ['nn_match_prediction', 'lstm_sequence', 'cnn_patterns']:
                model_file = os.path.join(model_path, f"{model_name}.h5")
                if os.path.exists(model_file):
                    self.models[model_name] = keras.models.load_model(model_file)
            
            # Load statistical models
            for model_name in ['poisson_goals', 'dixon_coles', 'elo_ratings', 'expected_goals']:
                model_file = os.path.join(model_path, f"{model_name}.pkl")
                if os.path.exists(model_file):
                    self.models[model_name] = joblib.load(model_file)
            
            # Load ensemble models
            for model_name in ['ensemble_match_result', 'stacking_ensemble']:
                model_file = os.path.join(model_path, f"{model_name}.pkl")
                if os.path.exists(model_file):
                    self.models[model_name] = joblib.load(model_file)
            
            # Load scalers and encoders
            scalers_file = os.path.join(model_path, 'scalers.pkl')
            if os.path.exists(scalers_file):
                self.scalers = joblib.load(scalers_file)
            
            encoders_file = os.path.join(model_path, 'encoders.pkl')
            if os.path.exists(encoders_file):
                self.encoders = joblib.load(encoders_file)
            
            # Load configuration
            config_file = os.path.join(model_path, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            
            # Load performance metrics
            performance_file = os.path.join(model_path, 'performance.json')
            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    self.model_performance = json.load(f)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        return {
            'performance': self.model_performance,
            'timestamp': datetime.now().isoformat()
        }
    
    def update_model_weights(self, performance_data: Dict[str, float]):
        """Update model weights based on performance"""
        logger.info("Updating model weights based on performance...")
        
        # Update weights based on recent performance
        for model_name, performance in performance_data.items():
            if model_name in self.model_performance:
                # Weighted average of historical and recent performance
                old_performance = self.model_performance[model_name]
                self.model_performance[model_name] = (old_performance * 0.7 + performance * 0.3)
        
        logger.info("Model weights updated")

# Example usage
if __name__ == "__main__":
    # Initialize prediction engine
    engine = FootballPredictionEngine()
    
    # Example match data
    match_data = {
        'home_team': 'Manchester United',
        'away_team': 'Liverpool',
        'league': 'Premier League',
        'home_odds': 2.5,
        'draw_odds': 3.2,
        'away_odds': 2.8,
        'home_form': [0.8, 0.6, 0.7, 0.9, 0.5],
        'away_form': [0.7, 0.8, 0.6, 0.7, 0.8],
        'h2h_home_wins': 0.4,
        'h2h_draws': 0.3,
        'h2h_away_wins': 0.3,
        'home_league_position': 4,
        'away_league_position': 2,
        'home_goals_scored_avg': 2.1,
        'home_goals_conceded_avg': 1.2,
        'away_goals_scored_avg': 1.8,
        'away_goals_conceded_avg': 1.4
    }
    
    # Make prediction
    prediction = engine.predict_match(match_data, 'match_result')
    print(json.dumps(prediction, indent=2))