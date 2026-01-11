#!/usr/bin/env python3
"""
Advanced Football Data Processor
Handles data collection, cleaning, and feature engineering
Author: AI Assistant
Version: 1.0.0
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class FootballData(Base):
    """Football match data model"""
    __tablename__ = 'football_matches'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(String(50), unique=True, nullable=False)
    date = Column(DateTime, nullable=False)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    league = Column(String(100), nullable=False)
    season = Column(String(20))
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)
    over_under = Column(Float)
    both_teams_to_score = Column(Float)
    attendance = Column(Integer)
    referee = Column(String(100))
    weather = Column(String(50))
    temperature = Column(Float)
    humidity = Column(Float)
    pitch_condition = Column(String(50))
    home_possession = Column(Float)
    away_possession = Column(Float)
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TeamStats(Base):
    """Team statistics model"""
    __tablename__ = 'team_stats'
    
    id = Column(Integer, primary_key=True)
    team_name = Column(String(100), nullable=False, index=True)
    league = Column(String(100), nullable=False)
    season = Column(String(20))
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_scored = Column(Integer, default=0)
    goals_conceded = Column(Integer, default=0)
    goal_difference = Column(Integer, default=0)
    points = Column(Integer, default=0)
    home_form = Column(String(500))  # JSON array
    away_form = Column(String(500))  # JSON array
    overall_form = Column(String(500))  # JSON array
    last_5_matches = Column(String(500))  # JSON array
    attack_strength = Column(Float, default=1.0)
    defence_strength = Column(Float, default=1.0)
    form_rating = Column(Float, default=0.5)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DataProcessor:
    """Advanced football data processor"""
    
    def __init__(self, config_path: str = None):
        """Initialize data processor"""
        self.config = self._load_config(config_path)
        self.db_path = os.path.abspath(self.config.get('database_path', 'data/football_data.db'))
        self.api_keys = self.config.get('api_keys', {})
        self.data_sources = self.config.get('data_sources', {})
        
        # Initialize database
                # database engine
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.warning("DB not reachable – using in-memory fallback: %s", e)
            self.engine = create_engine('sqlite:///:memory:', echo=False)
            Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

        # data holders
        self.matches_df = pd.DataFrame()
        self.teams_df = pd.DataFrame()
        self.leagues_df = pd.DataFrame()

        logger.info("DataProcessor initialized successfully")
        
        # Feature engineering
        self.feature_encoders = {}
        self.feature_scalers = {}
        
        logger.info("DataProcessor initialized successfully")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'database_path': 'data/football_data.db',
            'api_keys': {
                'football_api': 'your-api-key-here',
                'odds_api': 'your-odds-api-key-here',
                'weather_api': 'your-weather-api-key-here'
            },
            'data_sources': {
                'football_data': {
                    'url': 'https://api.football-data.org/v4',
                    'rate_limit': 100,
                    'timeout': 30
                },
                'odds_data': {
                    'url': 'https://api.the-odds-api.com/v4',
                    'rate_limit': 500,
                    'timeout': 30
                }
            },
            'feature_engineering': {
                'enable_form_features': True,
                'enable_head_to_head': True,
                'enable_team_stats': True,
                'enable_weather_features': True,
                'enable_odds_features': True
            },
            'data_quality': {
                'min_matches_per_team': 5,
                'max_missing_data_percentage': 0.1,
                'outlier_threshold': 3.0
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
    
    async def collect_data_async(self, start_date: str, end_date: str, 
                                leagues: List[str] = None) -> pd.DataFrame:
        """Collect data asynchronously from multiple sources"""
        logger.info(f"Collecting data from {start_date} to {end_date}")
        
        tasks = []
        
        # Collect from multiple sources
        if self.data_sources.get('football_data'):
            tasks.append(self._collect_football_data_async(start_date, end_date, leagues))
        
        if self.data_sources.get('odds_data'):
            tasks.append(self._collect_odds_data_async(start_date, end_date, leagues))
        
        if self.data_sources.get('weather_data'):
            tasks.append(self._collect_weather_data_async(start_date, end_date))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine data
        combined_data = self._combine_data_sources(results)
        
        # Clean and process data
        cleaned_data = self.clean_data(combined_data)
        
        # Store in database
        self.store_data(cleaned_data)
        
        return cleaned_data
    
    async def _collect_football_data_async(self, start_date: str, end_date: str, 
                                         leagues: List[str] = None) -> pd.DataFrame:
        """Collect football data asynchronously"""
        logger.info("Collecting football data...")
        
        data = []
        
        # API configuration
        api_config = self.data_sources['football_data']
        headers = {'X-Auth-Token': self.api_keys.get('football_api', '')}
        
        # List of leagues to collect
        if not leagues:
            leagues = ['PL', 'CL', 'PD', 'BL1', 'SA', 'FL1']  # Premier League, Champions League, etc.
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for league in leagues:
                task = self._fetch_league_data_async(session, league, start_date, end_date, headers)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, pd.DataFrame):
                    data.append(result)
                else:
                    logger.warning(f"Error collecting league data: {result}")
        
        if data:
            return pd.concat(data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    async def _fetch_league_data_async(self, session: aiohttp.ClientSession, 
                                     league: str, start_date: str, end_date: str, 
                                     headers: Dict[str, str]) -> pd.DataFrame:
        """Fetch league data asynchronously"""
        try:
            url = f"{self.data_sources['football_data']['url']}/competitions/{league}/matches"
            params = {
                'dateFrom': start_date,
                'dateTo': end_date
            }
            
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    matches = data.get('matches', [])
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(matches)
                    
                    # Process match data
                    if not df.empty:
                        df = self._process_match_data(df)
                    
                    return df
                else:
                    logger.warning(f"API request failed for league {league}: {response.status}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error fetching league {league} data: {e}")
            return pd.DataFrame()
    
    def _process_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw match data"""
        # Rename columns to standard format
        column_mapping = {
            'id': 'match_id',
            'utcDate': 'date',
            'homeTeam': 'home_team',
            'awayTeam': 'away_team',
            'competition': 'league',
            'score': 'score',
            'status': 'status'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Extract team names
        df['home_team'] = df['home_team'].apply(lambda x: x.get('name', '') if isinstance(x, dict) else x)
        df['away_team'] = df['away_team'].apply(lambda x: x.get('name', '') if isinstance(x, dict) else x)
        df['league'] = df['league'].apply(lambda x: x.get('name', '') if isinstance(x, dict) else x)
        
        # Process score
        if 'score' in df.columns:
            df['home_score'] = df['score'].apply(lambda x: x.get('fullTime', {}).get('home', 0) if isinstance(x, dict) else 0)
            df['away_score'] = df['score'].apply(lambda x: x.get('fullTime', {}).get('away', 0) if isinstance(x, dict) else 0)
            df = df.drop('score', axis=1)
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Add season
        df['season'] = df['date'].apply(self._get_season)
        
        return df
    
    def _get_season(self, date: datetime) -> str:
        """Get football season from date"""
        year = date.year
        if date.month >= 8:  # August starts new season
            return f"{year}-{year+1}"
        else:
            return f"{year-1}-{year}"
    
    async def _collect_odds_data_async(self, start_date: str, end_date: str, 
                                     leagues: List[str] = None) -> pd.DataFrame:
        """Collect odds data asynchronously"""
        logger.info("Collecting odds data...")
        
        # Implementation would depend on odds API
        # This is a placeholder implementation
        return pd.DataFrame()
    
    async def _collect_weather_data_async(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect weather data asynchronously"""
        logger.info("Collecting weather data...")
        
        # Implementation would depend on weather API
        # This is a placeholder implementation
        return pd.DataFrame()
    
    def _combine_data_sources(self, data_sources: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple sources"""
        logger.info("Combining data sources...")
        
        # Start with the largest dataset (usually match data)
        combined_df = pd.DataFrame()
        
        for df in data_sources:
            if not df.empty:
                if combined_df.empty:
                    combined_df = df.copy()
                else:
                    # Merge on common columns
                    common_cols = list(set(combined_df.columns) & set(df.columns))
                    if common_cols:
                        combined_df = pd.merge(combined_df, df, on=common_cols, how='left', suffixes=('', '_dup'))
                        # Remove duplicate columns
                        combined_df = combined_df.loc[:, ~combined_df.columns.str.endswith('_dup')]
        
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        logger.info("Cleaning data...")
        
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['match_id'], keep='last')
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Standardize formats
        df = self._standardize_formats(df)
        
        # Validate data quality
        df = self._validate_data_quality(df)
        
        logger.info(f"Data cleaning completed. Shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Numeric columns - fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        # Categorical columns - fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if df[col].mode().empty:
                    mode_value = 'Unknown'
                else:
                    mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers"""
        logger.info("Removing outliers...")
        
        # Define numeric columns that shouldn't have extreme values
        numeric_cols = ['home_score', 'away_score', 'home_odds', 'draw_odds', 'away_odds']
        
        for col in numeric_cols:
            if col in df.columns:
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive filtering
                upper_bound = Q3 + 3 * IQR
                
                # Count outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not outliers.empty:
                    logger.info(f"Found {len(outliers)} outliers in {col}")
                    
                    # For odds, cap instead of remove
                    if 'odds' in col:
                        df[col] = df[col].clip(lower_bound, upper_bound)
                    else:
                        # For scores, investigate before removing
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def _standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats"""
        logger.info("Standardizing formats...")
        
        # Team names - capitalize and strip whitespace
        team_cols = ['home_team', 'away_team']
        for col in team_cols:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title()
        
        # League names - standardize
        if 'league' in df.columns:
            df['league'] = df['league'].str.strip().str.title()
        
        # Date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality"""
        logger.info("Validating data quality...")
        
        # Check minimum matches per team
        min_matches = self.config['data_quality']['min_matches_per_team']
        
        team_counts = pd.concat([df['home_team'], df['away_team']]).value_counts()
        valid_teams = team_counts[team_counts >= min_matches].index
        
        # Filter to include only teams with sufficient data
        df = df[(df['home_team'].isin(valid_teams)) & (df['away_team'].isin(valid_teams))]
        
        # Check for reasonable score ranges
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df = df[(df['home_score'] >= 0) & (df['away_score'] >= 0)]
            df = df[(df['home_score'] <= 15) & (df['away_score'] <= 15)]  # Cap at 15 goals
        
        # Check for reasonable odds ranges
        odds_cols = ['home_odds', 'draw_odds', 'away_odds']
        for col in odds_cols:
            if col in df.columns:
                df = df[(df[col] >= 1.0) & (df[col] <= 100.0)]  # Reasonable odds range
        
        # Check missing data percentage
        max_missing = self.config['data_quality']['max_missing_data_percentage']
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > max_missing:
                logger.warning(f"Column {col} has {missing_pct:.2%} missing data")
        
        return df
    
    def store_data(self, df: pd.DataFrame):
        """Store data in database"""
        logger.info("Storing data in database...")
        
        try:
            session = self.Session()
            
            # Store matches
            for _, row in df.iterrows():
                match = FootballData(
                    match_id=row.get('match_id', ''),
                    date=row.get('date', datetime.now()),
                    home_team=row.get('home_team', ''),
                    away_team=row.get('away_team', ''),
                    league=row.get('league', ''),
                    season=row.get('season', ''),
                    home_score=row.get('home_score'),
                    away_score=row.get('away_score'),
                    home_odds=row.get('home_odds'),
                    draw_odds=row.get('draw_odds'),
                    away_odds=row.get('away_odds'),
                    over_under=row.get('over_under'),
                    both_teams_to_score=row.get('both_teams_to_score'),
                    attendance=row.get('attendance'),
                    referee=row.get('referee'),
                    weather=row.get('weather'),
                    temperature=row.get('temperature'),
                    humidity=row.get('humidity'),
                    pitch_condition=row.get('pitch_condition'),
                    home_possession=row.get('home_possession'),
                    away_possession=row.get('away_possession'),
                    home_shots=row.get('home_shots'),
                    away_shots=row.get('away_shots'),
                    home_shots_on_target=row.get('home_shots_on_target'),
                    away_shots_on_target=row.get('away_shots_on_target'),
                    home_corners=row.get('home_corners'),
                    away_corners=row.get('away_corners'),
                    home_fouls=row.get('home_fouls'),
                    away_fouls=row.get('away_fouls'),
                    home_yellow_cards=row.get('home_yellow_cards'),
                    away_yellow_cards=row.get('away_yellow_cards'),
                    home_red_cards=row.get('home_red_cards'),
                    away_red_cards=row.get('away_red_cards')
                )
                
                session.merge(match)  # Use merge to handle duplicates
            
            session.commit()
            logger.info(f"Stored {len(df)} matches in database")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing data: {e}")
            raise
        finally:
            session.close()
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for machine learning"""
        logger.info("Preparing training data...")
        
        # Feature engineering
        features_df = self._engineer_features(df)
        
        # Select relevant features
        feature_cols = self._select_features(features_df)
        X = features_df[feature_cols].values
        
        # Target variable (assuming we're predicting match result)
        y = self._create_target_variable(df)
        
        # Handle missing values
        X = self._handle_missing_values_ml(X)
        
        # Scale features
        X = self._scale_features(X)
        
        logger.info(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for machine learning"""
        logger.info("Engineering features...")
        
        features_df = df.copy()
        
        # Form features
        if self.config['feature_engineering']['enable_form_features']:
            features_df = self._add_form_features(features_df)
        
        # Head-to-head features
        if self.config['feature_engineering']['enable_head_to_head']:
            features_df = self._add_head_to_head_features(features_df)
        
        # Team statistics features
        if self.config['feature_engineering']['enable_team_stats']:
            features_df = self._add_team_stats_features(features_df)
        
        # Weather features
        if self.config['feature_engineering']['enable_weather_features']:
            features_df = self._add_weather_features(features_df)
        
        # Odds-based features
        if self.config['feature_engineering']['enable_odds_features']:
            features_df = self._add_odds_features(features_df)
        
        # Time-based features
        features_df = self._add_time_features(features_df)
        
        return features_df
    
    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team form features"""
        logger.info("Adding form features...")
        
        # Calculate form for each team
        team_form = {}
        
        for _, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            date = row['date']
            
            # Get recent matches for each team
            home_recent = df[
                ((df['home_team'] == home_team) | (df['away_team'] == home_team)) &
                (df['date'] < date)
            ].tail(5)
            
            away_recent = df[
                ((df['home_team'] == away_team) | (df['away_team'] == away_team)) &
                (df['date'] < date)
            ].tail(5)
            
            # Calculate form (win rate in last 5 matches)
            home_form = self._calculate_team_form(home_recent, home_team)
            away_form = self._calculate_team_form(away_recent, away_team)
            
            # Add to dataframe
            idx = row.name
            df.loc[idx, 'home_form'] = json.dumps(home_form)
            df.loc[idx, 'away_form'] = json.dumps(away_form)
            df.loc[idx, 'home_form_rating'] = np.mean(home_form) if home_form else 0.5
            df.loc[idx, 'away_form_rating'] = np.mean(away_form) if away_form else 0.5
        
        return df
    
    def _calculate_team_form(self, recent_matches: pd.DataFrame, team: str) -> List[float]:
        """Calculate team form from recent matches"""
        form = []
        
        for _, match in recent_matches.iterrows():
            if match['home_team'] == team:
                if match['home_score'] > match['away_score']:
                    form.append(1.0)  # Win
                elif match['home_score'] == match['away_score']:
                    form.append(0.5)  # Draw
                else:
                    form.append(0.0)  # Loss
            elif match['away_team'] == team:
                if match['away_score'] > match['home_score']:
                    form.append(1.0)  # Win
                elif match['away_score'] == match['home_score']:
                    form.append(0.5)  # Draw
                else:
                    form.append(0.0)  # Loss
        
        return form
    
    def _add_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head features"""
        logger.info("Adding head-to-head features...")
        
        for _, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            date = row['date']
            
            # Get previous matches between these teams
            h2h_matches = df[
                (
                    ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                    ((df['home_team'] == away_team) & (df['away_team'] == home_team))
                ) &
                (df['date'] < date)
            ].tail(10)  # Last 10 matches
            
            if not h2h_matches.empty:
                # Calculate H2H statistics
                home_wins = 0
                away_wins = 0
                draws = 0
                total_goals = 0
                
                for _, match in h2h_matches.iterrows():
                    if match['home_team'] == home_team:
                        if match['home_score'] > match['away_score']:
                            home_wins += 1
                        elif match['home_score'] < match['away_score']:
                            away_wins += 1
                        else:
                            draws += 1
                    else:
                        if match['away_score'] > match['home_score']:
                            home_wins += 1
                        elif match['away_score'] < match['home_score']:
                            away_wins += 1
                        else:
                            draws += 1
                    
                    total_goals += match['home_score'] + match['away_score']
                
                total_matches = len(h2h_matches)
                
                idx = row.name
                df.loc[idx, 'h2h_home_win_rate'] = home_wins / total_matches
                df.loc[idx, 'h2h_away_win_rate'] = away_wins / total_matches
                df.loc[idx, 'h2h_draw_rate'] = draws / total_matches
                df.loc[idx, 'h2h_avg_goals'] = total_goals / total_matches
            else:
                # No H2H history - use neutral values
                idx = row.name
                df.loc[idx, 'h2h_home_win_rate'] = 0.33
                df.loc[idx, 'h2h_away_win_rate'] = 0.33
                df.loc[idx, 'h2h_draw_rate'] = 0.34
                df.loc[idx, 'h2h_avg_goals'] = 2.5
        
        return df
    
    def _add_team_stats_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team statistics features"""
        logger.info("Adding team statistics features...")
        
        # Calculate team statistics up to each match
        for _, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            date = row['date']
            
            # Get all previous matches for each team
            home_prev = df[
                ((df['home_team'] == home_team) | (df['away_team'] == home_team)) &
                (df['date'] < date)
            ]
            
            away_prev = df[
                ((df['home_team'] == away_team) | (df['away_team'] == away_team)) &
                (df['date'] < date)
            ]
            
            # Calculate statistics
            home_stats = self._calculate_team_stats(home_prev, home_team)
            away_stats = self._calculate_team_stats(away_prev, away_team)
            
            # Add to dataframe
            idx = row.name
            for stat, value in home_stats.items():
                df.loc[idx, f'home_{stat}'] = value
            
            for stat, value in away_stats.items():
                df.loc[idx, f'away_{stat}'] = value
        
        return df
    
    def _calculate_team_stats(self, matches: pd.DataFrame, team: str) -> Dict[str, float]:
        """Calculate team statistics from matches"""
        if matches.empty:
            return self._get_default_team_stats()
        
        stats = {}
        
        # Basic stats
        matches_played = len(matches)
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        
        for _, match in matches.iterrows():
            if match['home_team'] == team:
                goals_scored += match['home_score']
                goals_conceded += match['away_score']
                
                if match['home_score'] > match['away_score']:
                    wins += 1
                elif match['home_score'] == match['away_score']:
                    draws += 1
                else:
                    losses += 1
            else:
                goals_scored += match['away_score']
                goals_conceded += match['home_score']
                
                if match['away_score'] > match['home_score']:
                    wins += 1
                elif match['away_score'] == match['home_score']:
                    draws += 1
                else:
                    losses += 1
        
        stats['win_rate'] = wins / matches_played
        stats['draw_rate'] = draws / matches_played
        stats['loss_rate'] = losses / matches_played
        stats['goals_scored_avg'] = goals_scored / matches_played
        stats['goals_conceded_avg'] = goals_conceded / matches_played
        stats['goal_difference_avg'] = (goals_scored - goals_conceded) / matches_played
        stats['points_avg'] = (wins * 3 + draws) / matches_played
        
        return stats
    
    def _get_default_team_stats(self) -> Dict[str, float]:
        """Get default team statistics"""
        return {
            'win_rate': 0.33,
            'draw_rate': 0.34,
            'loss_rate': 0.33,
            'goals_scored_avg': 1.3,
            'goals_conceded_avg': 1.3,
            'goal_difference_avg': 0.0,
            'points_avg': 1.33
        }
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather-related features"""
        logger.info("Adding weather features...")
        
        # Temperature categories
        if 'temperature' in df.columns:
            df['temp_category'] = pd.cut(df['temperature'], 
                                       bins=[-float('inf'), 0, 15, 25, float('inf')],
                                       labels=['cold', 'cool', 'mild', 'warm'])
        
        # Humidity categories
        if 'humidity' in df.columns:
            df['humidity_category'] = pd.cut(df['humidity'],
                                           bins=[0, 30, 60, 80, 100],
                                           labels=['low', 'moderate', 'high', 'very_high'])
        
        # Weather conditions
        if 'weather' in df.columns:
            # Simplify weather conditions
            weather_mapping = {
                'sunny': 'clear',
                'clear': 'clear',
                'partly_cloudy': 'cloudy',
                'cloudy': 'cloudy',
                'overcast': 'cloudy',
                'rain': 'rain',
                'rainy': 'rain',
                'drizzle': 'rain',
                'snow': 'snow',
                'fog': 'fog',
                'windy': 'windy'
            }
            
            df['weather_simplified'] = df['weather'].map(weather_mapping).fillna('unknown')
        
        return df
    
    def _add_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add odds-based features"""
        logger.info("Adding odds features...")
        
        # Implied probabilities
        if all(col in df.columns for col in ['home_odds', 'draw_odds', 'away_odds']):
            # Calculate implied probabilities
            total_prob = 1/df['home_odds'] + 1/df['draw_odds'] + 1/df['away_odds']
            
            df['home_implied_prob'] = (1/df['home_odds']) / total_prob
            df['draw_implied_prob'] = (1/df['draw_odds']) / total_prob
            df['away_implied_prob'] = (1/df['away_odds']) / total_prob
            
            # Bookmaker margin
            df['bookmaker_margin'] = total_prob - 1
            
            # Favorite identification
            df['favorite'] = np.where(df['home_odds'] < df['away_odds'], 'home', 'away')
            df['favorite'] = np.where(df['draw_odds'] < np.minimum(df['home_odds'], df['away_odds']), 'draw', df['favorite'])
            
            # Odds difference
            df['odds_diff'] = np.abs(df['home_odds'] - df['away_odds'])
            
            # Total goals odds
            if 'over_under' in df.columns:
                df['over_under_binary'] = np.where(df['over_under'] > 2.5, 'over', 'under')
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        logger.info("Adding time features...")
        
        if 'date' in df.columns:
            # Day of week
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            
            # Month
            df['month'] = df['date'].dt.month
            
            # Season
            df['season'] = df['date'].dt.month.map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            })
            
            # Time of day (if time information is available)
            if pd.api.types.is_datetime64_any_dtype(df['date']):
                df['hour'] = df['date'].dt.hour
                df['time_of_day'] = pd.cut(df['hour'], 
                                         bins=[0, 6, 12, 18, 24],
                                         labels=['night', 'morning', 'afternoon', 'evening'])
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for modeling"""
        # Remove non-feature columns
        exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'league', 
                       'season', 'home_score', 'away_score', 'status']
        
        # Also exclude any columns that contain 'score' or are targets
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and 'score' not in col.lower()]
        
        return feature_cols
    
    def _handle_missing_values_ml(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values for machine learning"""
        # Simple imputation with median
        from sklearn.impute import SimpleImputer
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        return X_imputed
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features for machine learning"""
        # Use StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store scaler for later use
        self.feature_scalers['standard'] = scaler
        
        return X_scaled
    
    def _create_target_variable(self, df: pd.DataFrame) -> np.ndarray:
        """Create target variable for classification"""
        # Default to match result (0: home win, 1: draw, 2: away win)
        y = np.where(df['home_score'] > df['away_score'], 0,
                    np.where(df['home_score'] < df['away_score'], 2, 1))
        
        return y
    
    def get_team_statistics(self, team_name: str, league: str = None) -> Dict[str, Any]:
        """Get comprehensive team statistics"""
        logger.info(f"Getting statistics for {team_name}")
        
        try:
            session = self.Session()
            
            # Get team matches
            query = session.query(FootballData).filter(
                ((FootballData.home_team == team_name) | (FootballData.away_team == team_name)) &
                (FootballData.status == 'FINISHED')
            )
            
            if league:
                query = query.filter(FootballData.league == league)
            
            matches = query.all()
            
            if not matches:
                return self._get_default_team_stats()
            
            # Calculate statistics
            stats = self._calculate_comprehensive_stats(matches, team_name)
            
            session.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting team statistics: {e}")
            return self._get_default_team_stats()
    
    def _calculate_comprehensive_stats(self, matches: List[FootballData], 
                                     team_name: str) -> Dict[str, Any]:
        """Calculate comprehensive team statistics"""
        stats = {
            'team_name': team_name,
            'total_matches': len(matches),
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'clean_sheets': 0,
            'matches_with_goals': 0,
            'home_matches': 0,
            'away_matches': 0,
            'home_wins': 0,
            'away_wins': 0,
            'recent_form': [],
            'league_performance': {},
            'goals_distribution': {'scored': [], 'conceded': []},
            'cards_stats': {'yellow': 0, 'red': 0},
            'disciplinary_points': 0
        }
        
        recent_results = []
        
        for match in matches:
            is_home = match.home_team == team_name
            
            if is_home:
                stats['home_matches'] += 1
                goals_scored = match.home_score
                goals_conceded = match.away_score
                stats['cards_stats']['yellow'] += match.home_yellow_cards or 0
                stats['cards_stats']['red'] += match.home_red_cards or 0
            else:
                stats['away_matches'] += 1
                goals_scored = match.away_score
                goals_conceded = match.home_score
                stats['cards_stats']['yellow'] += match.away_yellow_cards or 0
                stats['cards_stats']['red'] += match.away_red_cards or 0
            
            # Match result
            if goals_scored > goals_conceded:
                stats['wins'] += 1
                if is_home:
                    stats['home_wins'] += 1
                recent_results.append('W')
            elif goals_scored == goals_conceded:
                stats['draws'] += 1
                recent_results.append('D')
            else:
                stats['losses'] += 1
                recent_results.append('L')
            
            # Goals
            stats['goals_scored'] += goals_scored
            stats['goals_conceded'] += goals_conceded
            stats['goals_distribution']['scored'].append(goals_scored)
            stats['goals_distribution']['conceded'].append(goals_conceded)
            
            # Clean sheets and goals scored
            if goals_conceded == 0:
                stats['clean_sheets'] += 1
            if goals_scored > 0:
                stats['matches_with_goals'] += 1
            
            # League performance
            league = match.league
            if league not in stats['league_performance']:
                stats['league_performance'][league] = {
                    'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                    'goals_scored': 0, 'goals_conceded': 0
                }
            
            stats['league_performance'][league]['matches'] += 1
            if goals_scored > goals_conceded:
                stats['league_performance'][league]['wins'] += 1
            elif goals_scored == goals_conceded:
                stats['league_performance'][league]['draws'] += 1
            else:
                stats['league_performance'][league]['losses'] += 1
            
            stats['league_performance'][league]['goals_scored'] += goals_scored
            stats['league_performance'][league]['goals_conceded'] += goals_conceded
        
        # Calculate derived statistics
        if stats['total_matches'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_matches']
            stats['draw_rate'] = stats['draws'] / stats['total_matches']
            stats['loss_rate'] = stats['losses'] / stats['total_matches']
            stats['clean_sheet_rate'] = stats['clean_sheets'] / stats['total_matches']
            stats['goals_per_match'] = stats['goals_scored'] / stats['total_matches']
            stats['goals_conceded_per_match'] = stats['goals_conceded'] / stats['total_matches']
            stats['goal_difference'] = stats['goals_scored'] - stats['goals_conceded']
            stats['goal_difference_per_match'] = stats['goal_difference'] / stats['total_matches']
            
            # Points (3 for win, 1 for draw)
            stats['points'] = stats['wins'] * 3 + stats['draws']
            stats['points_per_match'] = stats['points'] / stats['total_matches']
            
            # Home/away performance
            if stats['home_matches'] > 0:
                stats['home_win_rate'] = stats['home_wins'] / stats['home_matches']
            if stats['away_matches'] > 0:
                stats['away_win_rate'] = (stats['wins'] - stats['home_wins']) / stats['away_matches']
        
        # Recent form (last 5 matches)
        stats['recent_form'] = recent_results[-5:] if len(recent_results) >= 5 else recent_results
        stats['recent_form_rating'] = np.mean([1 if r == 'W' else 0.5 if r == 'D' else 0 
                                              for r in stats['recent_form']]) if stats['recent_form'] else 0.5
        
        # Goals statistics
        if stats['goals_distribution']['scored']:
            stats['avg_goals_scored'] = np.mean(stats['goals_distribution']['scored'])
            stats['std_goals_scored'] = np.std(stats['goals_distribution']['scored'])
            stats['max_goals_scored'] = max(stats['goals_distribution']['scored'])
            stats['min_goals_scored'] = min(stats['goals_distribution']['scored'])
        
        if stats['goals_distribution']['conceded']:
            stats['avg_goals_conceded'] = np.mean(stats['goals_distribution']['conceded'])
            stats['std_goals_conceded'] = np.std(stats['goals_distribution']['conceded'])
            stats['max_goals_conceded'] = max(stats['goals_distribution']['conceded'])
            stats['min_goals_conceded'] = min(stats['goals_distribution']['conceded'])
        
        # Cards statistics
        stats['disciplinary_points'] = stats['cards_stats']['yellow'] + stats['cards_stats']['red'] * 3
        if stats['total_matches'] > 0:
            stats['cards_per_match'] = stats['disciplinary_points'] / stats['total_matches']
        
        return stats
    
    def update_data(self, start_date: str, end_date: str) -> bool:
        """Update data for specified date range"""
        logger.info(f"Updating data from {start_date} to {end_date}")
        
        try:
            # Collect new data
            new_data = asyncio.run(self.collect_data_async(start_date, end_date))
            
            if not new_data.empty:
                # Clean and store
                cleaned_data = self.clean_data(new_data)
                self.store_data(cleaned_data)
                
                logger.info(f"Data updated successfully. Added {len(cleaned_data)} new matches.")
                return True
            else:
                logger.warning("No new data found")
                return False
                
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return False
    
    def get_league_statistics(self, league_name: str) -> Dict[str, Any]:
        """Get league-wide statistics"""
        logger.info(f"Getting statistics for {league_name}")
        
        try:
            session = self.Session()
            
            # Get all matches for the league
            matches = session.query(FootballData).filter(
                FootballData.league == league_name,
                FootballData.status == 'FINISHED'
            ).all()
            
            if not matches:
                return {}
            
            # Calculate league statistics
            stats = {
                'league_name': league_name,
                'total_matches': len(matches),
                'avg_home_goals': np.mean([m.home_score for m in matches]),
                'avg_away_goals': np.mean([m.away_score for m in matches]),
                'avg_total_goals': np.mean([m.home_score + m.away_score for m in matches]),
                'home_win_rate': len([m for m in matches if m.home_score > m.away_score]) / len(matches),
                'draw_rate': len([m for m in matches if m.home_score == m.away_score]) / len(matches),
                'away_win_rate': len([m for m in matches if m.home_score < m.away_score]) / len(matches),
                'avg_home_odds': np.mean([m.home_odds for m in matches if m.home_odds]),
                'avg_draw_odds': np.mean([m.draw_odds for m in matches if m.draw_odds]),
                'avg_away_odds': np.mean([m.away_odds for m in matches if m.away_odds])
            }
            
            session.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting league statistics: {e}")
            return {}
    
    def export_data(self, format: str = 'csv', file_path: str = None) -> str:
        """Export data to various formats"""
        logger.info(f"Exporting data in {format} format")
        
        try:
            # Get all data
            session = self.Session()
            matches = session.query(FootballData).all()
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'match_id': m.match_id,
                'date': m.date,
                'home_team': m.home_team,
                'away_team': m.away_team,
                'league': m.league,
                'season': m.season,
                'home_score': m.home_score,
                'away_score': m.away_score,
                'home_odds': m.home_odds,
                'draw_odds': m.draw_odds,
                'away_odds': m.away_odds,
                'over_under': m.over_under,
                'both_teams_to_score': m.both_teams_to_score,
                'attendance': m.attendance,
                'referee': m.referee,
                'weather': m.weather,
                'temperature': m.temperature,
                'humidity': m.humidity
            } for m in matches])
            
            session.close()
            
            # Export based on format
            if format.lower() == 'csv':
                if not file_path:
                    file_path = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(file_path, index=False)
                
            elif format.lower() == 'json':
                if not file_path:
                    file_path = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                df.to_json(file_path, orient='records', indent=2)
                
            elif format.lower() == 'excel':
                if not file_path:
                    file_path = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                df.to_excel(file_path, index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data exported to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize data processor
    processor = DataProcessor()
    
    # Example: Get team statistics
    stats = processor.get_team_statistics("Manchester United", "Premier League")
    print(json.dumps(stats, indent=2, default=str))