#!/usr/bin/env python3
"""
Advanced Betting Calculator
Handles betting calculations, bankroll management, and profit analysis
Author: AI Assistant
Version: 1.0.0
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class BetStatus(Enum):
    """Bet status enumeration"""
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    VOID = "void"
    HALF_WON = "half_won"
    HALF_LOST = "half_lost"

class BetType(Enum):
    """Bet type enumeration"""
    SINGLE = "single"
    DOUBLE = "double"
    TREBLE = "treble"
    ACCUMULATOR = "accumulator"
    SYSTEM = "system"

class StakingStrategy(Enum):
    """Staking strategy enumeration"""
    FLAT = "flat"
    PERCENTAGE = "percentage"
    KELLY = "kelly"
    MARTINGALE = "martingale"
    FIBONACCI = "fibonacci"
    PROPORTIONAL = "proportional"

@dataclass
class Bet:
    """Bet data class"""
    bet_id: str
    user_id: int
    prediction_id: int
    bet_type: BetType
    stake: float
    odds: float
    potential_win: float
    actual_win: float
    status: BetStatus
    created_at: datetime
    settled_at: Optional[datetime] = None

@dataclass
class BettingStats:
    """Betting statistics data class"""
    total_bets: int
    won_bets: int
    lost_bets: int
    void_bets: int
    total_stake: float
    total_profit: float
    roi: float
    average_odds: float
    win_rate: float
    average_stake: float
    longest_winning_streak: int
    longest_losing_streak: int
    current_streak: int
    max_drawdown: float
    sharpe_ratio: float

class BetRecord(Base):
    """Bet record database model"""
    __tablename__ = 'bet_records'
    
    id = Column(Integer, primary_key=True)
    bet_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    prediction_id = Column(Integer, nullable=False, index=True)
    bet_type = Column(String(20), nullable=False)
    stake = Column(Float, nullable=False)
    odds = Column(Float, nullable=False)
    potential_win = Column(Float, nullable=False)
    actual_win = Column(Float, default=0.0)
    status = Column(String(20), nullable=False, default=BetStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime)
    notes = Column(String(500))

class BettingHistory(Base):
    """Betting history database model"""
    __tablename__ = 'betting_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    starting_bankroll = Column(Float, nullable=False)
    ending_bankroll = Column(Float, nullable=False)
    daily_profit = Column(Float, default=0.0)
    daily_roi = Column(Float, default=0.0)
    bets_placed = Column(Integer, default=0)
    bets_won = Column(Integer, default=0)
    total_stake = Column(Float, default=0.0)
    notes = Column(String(500))

class BettingCalculator:
    """Advanced betting calculator with comprehensive analysis"""
    
    def __init__(self, config_path: str = None):
        """Initialize betting calculator"""
        self.config = self._load_config(config_path)
        self.db_path = self.config.get('database_path', 'data/betting_data.db')
        
        # Initialize database
                # database engine
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.warning("Betting DB not reachable – using in-memory fallback: %s", e)
            self.engine = create_engine('sqlite:///:memory:', echo=False)
            Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

        logger.info("BettingCalculator initialized successfully")
        # Betting parameters
        self.min_odds = self.config.get('min_odds', 1.1)
        self.max_odds = self.config.get('max_odds', 50.0)
        self.max_stake_percentage = self.config.get('max_stake_percentage', 0.05)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.25)
        
        logger.info("BettingCalculator initialized successfully")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'database_path': 'data/betting_data.db',
            'min_odds': 1.1,
            'max_odds': 50.0,
            'max_stake_percentage': 0.05,
            'kelly_fraction': 0.25,
            'staking_strategies': {
                'flat_stake': 10.0,
                'percentage_stake': 0.02,
                'kelly_multiplier': 0.25,
                'martingale_multiplier': 2.0,
                'fibonacci_sequence': [1, 1, 2, 3, 5, 8, 13]
            },
            'risk_management': {
                'max_daily_loss': 0.1,
                'max_weekly_loss': 0.2,
                'max_monthly_loss': 0.3,
                'stop_loss_percentage': 0.15,
                'take_profit_percentage': 0.25
            },
            'bankroll_management': {
                'initial_bankroll': 1000.0,
                'min_bankroll': 100.0,
                'max_stake_percentage': 0.05,
                'min_stake': 1.0,
                'max_stake': 100.0
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
    
    def calculate_stake(self, bankroll: float, confidence: float, odds: float,
                       strategy: StakingStrategy = StakingStrategy.PERCENTAGE,
                       **kwargs) -> float:
        """
        Calculate optimal stake based on strategy and risk parameters
        
        Args:
            bankroll: Current bankroll amount
            confidence: Prediction confidence (0-1)
            odds: Betting odds
            strategy: Staking strategy to use
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Recommended stake amount
        """
        try:
            # Validate inputs
            if bankroll <= 0:
                raise ValueError("Bankroll must be positive")
            if not (0 <= confidence <= 1):
                raise ValueError("Confidence must be between 0 and 1")
            if odds < self.min_odds or odds > self.max_odds:
                raise ValueError(f"Odds must be between {self.min_odds} and {self.max_odds}")
            
            # Calculate stake based on strategy
            if strategy == StakingStrategy.FLAT:
                stake = self._calculate_flat_stake(bankroll, **kwargs)
            
            elif strategy == StakingStrategy.PERCENTAGE:
                stake = self._calculate_percentage_stake(bankroll, confidence, **kwargs)
            
            elif strategy == StakingStrategy.KELLY:
                stake = self._calculate_kelly_stake(bankroll, confidence, odds, **kwargs)
            
            elif strategy == StakingStrategy.MARTINGALE:
                stake = self._calculate_martingale_stake(bankroll, **kwargs)
            
            elif strategy == StakingStrategy.FIBONACCI:
                stake = self._calculate_fibonacci_stake(bankroll, **kwargs)
            
            elif strategy == StakingStrategy.PROPORTIONAL:
                stake = self._calculate_proportional_stake(bankroll, confidence, **kwargs)
            
            else:
                raise ValueError(f"Unknown staking strategy: {strategy}")
            
            # Apply risk management constraints
            stake = self._apply_risk_constraints(stake, bankroll)
            
            logger.info(f"Calculated stake: {stake:.2f} using {strategy.value} strategy")
            return max(0, stake)  # Ensure non-negative stake
            
        except Exception as e:
            logger.error(f"Error calculating stake: {e}")
            return 0.0
    
    def _calculate_flat_stake(self, bankroll: float, **kwargs) -> float:
        """Calculate flat stake amount"""
        flat_amount = kwargs.get('flat_amount', self.config['staking_strategies']['flat_stake'])
        return min(flat_amount, bankroll * self.max_stake_percentage)
    
    def _calculate_percentage_stake(self, bankroll: float, confidence: float, **kwargs) -> float:
        """Calculate percentage-based stake"""
        base_percentage = kwargs.get('base_percentage', self.config['staking_strategies']['percentage_stake'])
        adjusted_percentage = base_percentage * confidence
        return bankroll * adjusted_percentage
    
    def _calculate_kelly_stake(self, bankroll: float, confidence: float, odds: float, **kwargs) -> float:
        """
        Calculate Kelly Criterion stake
        
        Kelly formula: f = (bp - q) / b
        where:
        - f is the fraction of bankroll to bet
        - b is the net odds received (odds - 1)
        - p is the probability of winning
        - q is the probability of losing (1 - p)
        """
        try:
            # Get Kelly parameters
            kelly_multiplier = kwargs.get('kelly_multiplier', self.kelly_fraction)
            probability = confidence  # Use confidence as win probability
            
            # Validate probability
            if probability <= 0 or probability >= 1:
                logger.warning(f"Invalid probability for Kelly: {probability}")
                return 0.0
            
            # Calculate Kelly fraction
            b = odds - 1  # Net odds
            p = probability  # Win probability
            q = 1 - p  # Loss probability
            
            kelly_fraction = (b * p - q) / b
            
            # Apply Kelly multiplier and constraints
            kelly_fraction = kelly_fraction * kelly_multiplier
            
            # Ensure Kelly fraction is reasonable
            kelly_fraction = max(0, min(kelly_fraction, self.max_stake_percentage))
            
            return bankroll * kelly_fraction
            
        except Exception as e:
            logger.error(f"Error in Kelly calculation: {e}")
            return 0.0
    
    def _calculate_martingale_stake(self, bankroll: float, **kwargs) -> float:
        """Calculate Martingale stake (for recovery after losses)"""
        try:
            # Get loss streak information
            user_id = kwargs.get('user_id')
            loss_streak = kwargs.get('loss_streak', 0)
            base_stake = kwargs.get('base_stake', self.config['staking_strategies']['flat_stake'])
            multiplier = kwargs.get('multiplier', self.config['staking_strategies']['martingale_multiplier'])
            
            if loss_streak == 0:
                return base_stake
            
            # Calculate Martingale stake
            martingale_stake = base_stake * (multiplier ** loss_streak)
            
            # Apply safety constraints
            max_stake = bankroll * self.max_stake_percentage
            return min(martingale_stake, max_stake)
            
        except Exception as e:
            logger.error(f"Error in Martingale calculation: {e}")
            return 0.0
    
    def _calculate_fibonacci_stake(self, bankroll: float, **kwargs) -> float:
        """Calculate Fibonacci stake"""
        try:
            # Get Fibonacci sequence position
            position = kwargs.get('fibonacci_position', 0)
            fib_sequence = self.config['staking_strategies']['fibonacci_sequence']
            
            if position >= len(fib_sequence):
                position = len(fib_sequence) - 1
            
            base_stake = kwargs.get('base_stake', self.config['staking_strategies']['flat_stake'])
            fib_stake = base_stake * fib_sequence[position]
            
            # Apply constraints
            max_stake = bankroll * self.max_stake_percentage
            return min(fib_stake, max_stake)
            
        except Exception as e:
            logger.error(f"Error in Fibonacci calculation: {e}")
            return 0.0
    
    def _calculate_proportional_stake(self, bankroll: float, confidence: float, **kwargs) -> float:
        """Calculate stake proportional to confidence and edge"""
        try:
            # Get edge (expected value)
            edge = kwargs.get('edge', 0.0)
            min_edge = kwargs.get('min_edge', 0.05)  # Minimum edge to bet
            
            if edge < min_edge:
                return 0.0
            
            # Calculate proportional stake
            base_percentage = self.config['staking_strategies']['percentage_stake']
            proportional_stake = bankroll * base_percentage * (1 + edge) * confidence
            
            return proportional_stake
            
        except Exception as e:
            logger.error(f"Error in proportional calculation: {e}")
            return 0.0
    
    def _apply_risk_constraints(self, stake: float, bankroll: float) -> float:
        """Apply risk management constraints to stake"""
        # Maximum stake percentage of bankroll
        max_stake = bankroll * self.max_stake_percentage
        stake = min(stake, max_stake)
        
        # Minimum stake constraint
        min_stake = self.config['bankroll_management']['min_stake']
        if stake < min_stake:
            return 0.0  # Don't bet if stake is too small
        
        # Maximum stake constraint
        max_stake_limit = self.config['bankroll_management']['max_stake']
        stake = min(stake, max_stake_limit)
        
        return stake
    
    def calculate_expected_value(self, probability: float, odds: float) -> float:
        """
        Calculate expected value of a bet
        
        EV = (Probability × Odds) - (1 - Probability)
        
        Args:
            probability: Probability of winning (0-1)
            odds: Betting odds
            
        Returns:
            Expected value (positive = profitable, negative = unprofitable)
        """
        if not (0 <= probability <= 1):
            raise ValueError("Probability must be between 0 and 1")
        if odds <= 1:
            raise ValueError("Odds must be greater than 1")
        
        return (probability * odds) - (1 - probability)
    
    def calculate_implied_probability(self, odds: float, margin: float = 0.0) -> float:
        """
        Calculate implied probability from odds
        
        Args:
            odds: Betting odds
            margin: Bookmaker margin (0-1)
            
        Returns:
            Implied probability
        """
        if odds <= 0:
            raise ValueError("Odds must be positive")
        
        fair_probability = 1 / odds
        adjusted_probability = fair_probability * (1 - margin)
        
        return adjusted_probability
    
    def calculate_parlay_odds(self, odds_list: List[float]) -> float:
        """
        Calculate combined odds for parlay/accumulator bet
        
        Args:
            odds_list: List of individual odds
            
        Returns:
            Combined parlay odds
        """
        if not odds_list:
            return 1.0
        
        # Validate odds
        for odds in odds_list:
            if odds < 1:
                raise ValueError("All odds must be >= 1")
        
        # Calculate combined odds
        combined_odds = 1.0
        for odds in odds_list:
            combined_odds *= odds
        
        return combined_odds
    
    def calculate_each_way_terms(self, win_odds: float, place_terms: str = "1/4") -> Dict[str, float]:
        """
        Calculate each-way bet terms
        
        Args:
            win_odds: Win odds
            place_terms: Place terms (e.g., "1/4", "1/5")
            
        Returns:
            Dictionary with win and place odds
        """
        try:
            # Parse place terms
            numerator, denominator = map(int, place_terms.split('/'))
            place_fraction = numerator / denominator
            
            # Calculate place odds
            place_odds = 1 + (win_odds - 1) * place_fraction
            
            return {
                'win_odds': win_odds,
                'place_odds': place_odds,
                'place_terms': place_terms,
                'place_fraction': place_fraction
            }
            
        except Exception as e:
            logger.error(f"Error calculating each-way terms: {e}")
            return {
                'win_odds': win_odds,
                'place_odds': 1.0,
                'place_terms': place_terms,
                'place_fraction': 0.0
            }
    
    def calculate_asian_handicap(self, home_odds: float, away_odds: float, 
                               handicap: float) -> Dict[str, Any]:
        """
        Calculate Asian handicap outcomes
        
        Args:
            home_odds: Home team odds
            away_odds: Away team odds
            handicap: Handicap value (e.g., -0.5, +1.25)
            
        Returns:
            Dictionary with handicap outcomes
        """
        try:
            outcomes = {}
            
            # Handle different handicap types
            if handicap == 0:
                # Level ball (0)
                outcomes = {
                    'home_win': {'odds': home_odds, 'result': 'home_win'},
                    'away_win': {'odds': away_odds, 'result': 'away_win'},
                    'void': {'odds': 1.0, 'result': 'void'}
                }
                
            elif handicap % 0.25 == 0:
                # Quarter-ball handicap (e.g., -0.25, +0.75)
                outcomes = self._calculate_quarter_handicap(home_odds, away_odds, handicap)
                
            elif handicap % 0.5 == 0:
                # Half-ball handicap (e.g., -0.5, +1.5)
                outcomes = self._calculate_half_handicap(home_odds, away_odds, handicap)
                
            else:
                # Full-ball handicap with half win/loss
                outcomes = self._calculate_full_handicap(home_odds, away_odds, handicap)
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Error calculating Asian handicap: {e}")
            return {}
    
    def _calculate_quarter_handicap(self, home_odds: float, away_odds: float, 
                                  handicap: float) -> Dict[str, Any]:
        """Calculate quarter-ball Asian handicap"""
        outcomes = {}
        
        # Split handicap into two parts
        lower_handicap = handicap - 0.25
        upper_handicap = handicap + 0.25        
        outcomes['home_half_win'] = {
            'odds': (home_odds + 1.0) / 2,  # Average with void odds
            'handicap': lower_handicap,
            'result': 'half_win'
        }
        
        outcomes['home_half_lose'] = {
            'odds': 0.5,  # Half stake lost
            'handicap': upper_handicap,
            'result': 'half_lose'
        }
        
        outcomes['away_half_win'] = {
            'odds': (away_odds + 1.0) / 2,
            'handicap': -lower_handicap,
            'result': 'half_win'
        }
        
        outcomes['away_half_lose'] = {
            'odds': 0.5,
            'handicap': -upper_handicap,
            'result': 'half_lose'
        }
        
        return outcomes
    
    def _calculate_half_handicap(self, home_odds: float, away_odds: float, 
                               handicap: float) -> Dict[str, Any]:
        """Calculate half-ball Asian handicap"""
        return {
            'home_win': {'odds': home_odds, 'handicap': handicap, 'result': 'win'},
            'away_win': {'odds': away_odds, 'handicap': -handicap, 'result': 'win'}
        }
    
    def _calculate_full_handicap(self, home_odds: float, away_odds: float, 
                               handicap: float) -> Dict[str, Any]:
        """Calculate full-ball Asian handicap with half win/loss"""
        return {
            'home_win': {'odds': home_odds, 'handicap': handicap, 'result': 'win'},
            'home_half_win': {'odds': (home_odds + 1.0) / 2, 'handicap': handicap - 0.25, 'result': 'half_win'},
            'away_win': {'odds': away_odds, 'handicap': -handicap, 'result': 'win'},
            'away_half_win': {'odds': (away_odds + 1.0) / 2, 'handicap': -handicap + 0.25, 'result': 'half_win'}
        }
    
    def calculate_bankroll_growth(self, initial_bankroll: float, 
                                bets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate bankroll growth over time
        
        Args:
            initial_bankroll: Starting bankroll amount
            bets: List of bet results with dates and profits
            
        Returns:
            Dictionary with growth statistics
        """
        try:
            if not bets:
                return {
                    'initial_bankroll': initial_bankroll,
                    'final_bankroll': initial_bankroll,
                    'total_profit': 0.0,
                    'total_roi': 0.0,
                    'growth_curve': []
                }
            
            # Sort bets by date
            sorted_bets = sorted(bets, key=lambda x: x['date'])
            
            bankroll = initial_bankroll
            growth_curve = []
            cumulative_profit = 0.0
            
            for bet in sorted_bets:
                stake = bet.get('stake', 0.0)
                profit = bet.get('profit', 0.0)
                
                # Update bankroll
                bankroll += profit
                cumulative_profit += profit
                
                # Record growth point
                growth_curve.append({
                    'date': bet['date'],
                    'bankroll': bankroll,
                    'cumulative_profit': cumulative_profit,
                    'roi': (cumulative_profit / initial_bankroll) * 100
                })
            
            # Calculate statistics
            final_bankroll = bankroll
            total_profit = final_bankroll - initial_bankroll
            total_roi = (total_profit / initial_bankroll) * 100
            
            # Calculate maximum drawdown
            peak = initial_bankroll
            max_drawdown = 0.0
            
            for point in growth_curve:
                if point['bankroll'] > peak:
                    peak = point['bankroll']
                
                drawdown = (peak - point['bankroll']) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'initial_bankroll': initial_bankroll,
                'final_bankroll': final_bankroll,
                'total_profit': total_profit,
                'total_roi': total_roi,
                'max_drawdown': max_drawdown,
                'growth_curve': growth_curve,
                'profit_factor': self._calculate_profit_factor(bets),
                'sharpe_ratio': self._calculate_sharpe_ratio(bets)
            }
            
        except Exception as e:
            logger.error(f"Error calculating bankroll growth: {e}")
            return {}
    
    def _calculate_profit_factor(self, bets: List[Dict[str, Any]]) -> float:
        """Calculate profit factor (ratio of total wins to total losses)"""
        total_wins = sum(bet.get('profit', 0) for bet in bets if bet.get('profit', 0) > 0)
        total_losses = abs(sum(bet.get('profit', 0) for bet in bets if bet.get('profit', 0) < 0))
        
        if total_losses == 0:
            return float('inf') if total_wins > 0 else 0.0
        
        return total_wins / total_losses
    
    def _calculate_sharpe_ratio(self, bets: List[Dict[str, Any]], 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not bets:
            return 0.0
        
        # Calculate returns
        returns = []
        for bet in bets:
            stake = bet.get('stake', 0)
            profit = bet.get('profit', 0)
            if stake > 0:
                return_pct = profit / stake
                returns.append(return_pct)
        
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / 365)  # Daily risk-free rate
        
        if len(excess_returns) < 2:
            return 0.0
        
        # Calculate Sharpe ratio (annualized)
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily betting)
        sharpe_ratio = mean_return / std_return * np.sqrt(365)
        
        return sharpe_ratio
    
    def calculate_streaks(self, bets: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate winning and losing streaks
        
        Args:
            bets: List of bet results
            
        Returns:
            Dictionary with streak information
        """
        if not bets:
            return {
                'longest_winning_streak': 0,
                'longest_losing_streak': 0,
                'current_streak': 0,
                'current_streak_type': 'none'
            }
        
        # Sort bets by date
        sorted_bets = sorted(bets, key=lambda x: x['date'])
        
        current_streak = 0
        current_type = 'none'
        longest_win = 0
        longest_loss = 0
        
        for bet in sorted_bets:
            profit = bet.get('profit', 0)
            
            if profit > 0:  # Win
                if current_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'win'
                longest_win = max(longest_win, current_streak)
                
            elif profit < 0:  # Loss
                if current_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'loss'
                longest_loss = max(longest_loss, current_streak)
                
            else:  # Push/void
                current_streak = 0
                current_type = 'none'
        
        return {
            'longest_winning_streak': longest_win,
            'longest_losing_streak': longest_loss,
            'current_streak': current_streak,
            'current_streak_type': current_type
        }
    
    def analyze_betting_performance(self, user_id: int, start_date: str = None, 
                                  end_date: str = None) -> BettingStats:
        """
        Comprehensive betting performance analysis
        
        Args:
            user_id: User ID to analyze
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            
        Returns:
            Comprehensive betting statistics
        """
        try:
            session = self.Session()
            
            # Build query
            query = session.query(BetRecord).filter(BetRecord.user_id == user_id)
            
            if start_date:
                query = query.filter(BetRecord.created_at >= start_date)
            if end_date:
                query = query.filter(BetRecord.created_at <= end_date)
            
            bets = query.all()
            
            if not bets:
                return BettingStats(
                    total_bets=0, won_bets=0, lost_bets=0, void_bets=0,
                    total_stake=0.0, total_profit=0.0, roi=0.0, average_odds=0.0,
                    win_rate=0.0, average_stake=0.0, longest_winning_streak=0,
                    longest_losing_streak=0, current_streak=0, max_drawdown=0.0,
                    sharpe_ratio=0.0
                )
            
            # Calculate basic statistics
            total_bets = len(bets)
            won_bets = len([b for b in bets if b.status == BetStatus.WON.value])
            lost_bets = len([b for b in bets if b.status == BetStatus.LOST.value])
            void_bets = len([b for b in bets if b.status == BetStatus.VOID.value])
            
            total_stake = sum(b.stake for b in bets)
            total_profit = sum(b.actual_win - b.stake for b in bets)
            roi = (total_profit / total_stake * 100) if total_stake > 0 else 0.0
            
            average_odds = np.mean([b.odds for b in bets]) if bets else 0.0
            win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0.0
            average_stake = total_stake / total_bets if total_bets > 0 else 0.0
            
            # Calculate streaks
            bet_results = [{
                'date': b.created_at,
                'profit': b.actual_win - b.stake
            } for b in bets]
            
            streaks = self.calculate_streaks(bet_results)
            
            # Calculate bankroll growth and drawdown
            bankroll_data = self.calculate_bankroll_growth(
                self.config['bankroll_management']['initial_bankroll'],
                bet_results
            )
            
            session.close()
            
            return BettingStats(
                total_bets=total_bets,
                won_bets=won_bets,
                lost_bets=lost_bets,
                void_bets=void_bets,
                total_stake=total_stake,
                total_profit=total_profit,
                roi=roi,
                average_odds=average_odds,
                win_rate=win_rate,
                average_stake=average_stake,
                longest_winning_streak=streaks['longest_winning_streak'],
                longest_losing_streak=streaks['longest_losing_streak'],
                current_streak=streaks['current_streak'],
                max_drawdown=bankroll_data.get('max_drawdown', 0.0),
                sharpe_ratio=bankroll_data.get('sharpe_ratio', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing betting performance: {e}")
            session.close()
            return BettingStats(
                total_bets=0, won_bets=0, lost_bets=0, void_bets=0,
                total_stake=0.0, total_profit=0.0, roi=0.0, average_odds=0.0,
                win_rate=0.0, average_stake=0.0, longest_winning_streak=0,
                longest_losing_streak=0, current_streak=0, max_drawdown=0.0,
                sharpe_ratio=0.0
            )
    
    def record_bet(self, bet: Bet) -> bool:
        """Record a new bet"""
        try:
            session = self.Session()
            
            bet_record = BetRecord(
                bet_id=bet.bet_id,
                user_id=bet.user_id,
                prediction_id=bet.prediction_id,
                bet_type=bet.bet_type.value,
                stake=bet.stake,
                odds=bet.odds,
                potential_win=bet.potential_win,
                actual_win=bet.actual_win,
                status=bet.status.value
            )
            
            session.add(bet_record)
            session.commit()
            
            logger.info(f"Bet recorded: {bet.bet_id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording bet: {e}")
            return False
        finally:
            session.close()
    
    def settle_bet(self, bet_id: str, actual_win: float, status: BetStatus) -> bool:
        """Settle a bet with actual result"""
        try:
            session = self.Session()
            
            bet_record = session.query(BetRecord).filter_by(bet_id=bet_id).first()
            
            if not bet_record:
                logger.error(f"Bet not found: {bet_id}")
                return False
            
            bet_record.actual_win = actual_win
            bet_record.status = status.value
            bet_record.settled_at = datetime.utcnow()
            
            session.commit()
            
            logger.info(f"Bet settled: {bet_id} - Status: {status.value}, Win: {actual_win}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error settling bet: {e}")
            return False
        finally:
            session.close()
    
    def get_betting_recommendations(self, user_id: int, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate betting recommendations based on predictions and user history
        
        Args:
            user_id: User ID
            predictions: List of predictions with confidence and odds
            
        Returns:
            List of betting recommendations
        """
        try:
            # Get user betting history
            user_stats = self.analyze_betting_performance(user_id)
            
            recommendations = []
            
            for prediction in predictions:
                # Calculate expected value
                ev = self.calculate_expected_value(
                    prediction['confidence'],
                    prediction['odds']
                )
                
                # Calculate optimal stake
                bankroll = self.config['bankroll_management']['initial_bankroll']  # Should get from user profile
                stake = self.calculate_stake(
                    bankroll,
                    prediction['confidence'],
                    prediction['odds'],
                    StakingStrategy.KELLY
                )
                
                # Risk assessment
                risk_level = self._assess_bet_risk(prediction, user_stats)
                
                # Recommendation logic
                if ev > 0.05 and stake > 0:  # Positive expected value
                    recommendation = {
                        'prediction_id': prediction['id'],
                        'recommended_action': 'BET',
                        'stake': stake,
                        'expected_value': ev,
                        'confidence': prediction['confidence'],
                        'risk_level': risk_level,
                        'reasoning': f"Positive EV ({ev:.3f}) and sufficient confidence ({prediction['confidence']:.2%})"
                    }
                elif ev > 0.02 and stake > 0:
                    recommendation = {
                        'prediction_id': prediction['id'],
                        'recommended_action': 'SMALL_BET',
                        'stake': stake * 0.5,  # Reduce stake for marginal EV
                        'expected_value': ev,
                        'confidence': prediction['confidence'],
                        'risk_level': risk_level,
                        'reasoning': f"Marginal EV ({ev:.3f}) - consider smaller stake"
                    }
                else:
                    recommendation = {
                        'prediction_id': prediction['id'],
                        'recommended_action': 'SKIP',
                        'stake': 0.0,
                        'expected_value': ev,
                        'confidence': prediction['confidence'],
                        'risk_level': risk_level,
                        'reasoning': f"Negative or insufficient EV ({ev:.3f})"
                    }
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating betting recommendations: {e}")
            return []
    
    def _assess_bet_risk(self, prediction: Dict[str, Any], user_stats: BettingStats) -> str:
        """Assess risk level of a bet"""
        risk_score = 0
        
        # Confidence risk
        risk_score += (1 - prediction['confidence']) * 3
        
        # Odds risk (higher odds = higher risk)
        risk_score += (prediction['odds'] - 1) * 0.5
        
        # User performance risk
        if user_stats.win_rate < 40:
            risk_score += 1
        elif user_stats.win_rate > 60:
            risk_score -= 0.5
        
        # Recent performance risk
        if user_stats.current_streak < 0 and user_stats.current_streak_type == 'loss':
            risk_score += min(abs(user_stats.current_streak) * 0.2, 2)
        
        # Determine risk level
        if risk_score < 1.5:
            return 'low'
        elif risk_score < 3.0:
            return 'medium'
        else:
            return 'high'
    
    def calculate_compound_growth(self, initial_bankroll: float, 
                                monthly_roi: float, months: int) -> Dict[str, Any]:
        """
        Calculate compound growth projections
        
        Args:
            initial_bankroll: Starting amount
            monthly_roi: Monthly return on investment (as decimal)
            months: Number of months to project
            
        Returns:
            Dictionary with growth projections
        """
        try:
            projections = []
            current_bankroll = initial_bankroll
            
            for month in range(1, months + 1):
                monthly_profit = current_bankroll * monthly_roi
                current_bankroll += monthly_profit
                
                projections.append({
                    'month': month,
                    'bankroll': current_bankroll,
                    'monthly_profit': monthly_profit,
                    'cumulative_profit': current_bankroll - initial_bankroll,
                    'total_roi': (current_bankroll - initial_bankroll) / initial_bankroll * 100
                })
            
            return {
                'initial_bankroll': initial_bankroll,
                'final_bankroll': current_bankroll,
                'total_profit': current_bankroll - initial_bankroll,
                'total_roi': (current_bankroll - initial_bankroll) / initial_bankroll * 100,
                'monthly_roi': monthly_roi * 100,
                'projections': projections
            }
            
        except Exception as e:
            logger.error(f"Error calculating compound growth: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Initialize calculator
    calculator = BettingCalculator()
    
    # Example calculations
    bankroll = 1000.0
    confidence = 0.75
    odds = 2.5
    
    # Calculate stake using Kelly criterion
    kelly_stake = calculator.calculate_stake(bankroll, confidence, odds, StakingStrategy.KELLY)
    print(f"Kelly stake: ${kelly_stake:.2f}")
    
    # Calculate expected value
    ev = calculator.calculate_expected_value(confidence, odds)
    print(f"Expected value: {ev:.3f}")
    
    # Calculate parlay odds
    parlay_odds = calculator.calculate_parlay_odds([1.8, 2.2, 1.9])
    print(f"Parlay odds: {parlay_odds:.2f}")
    
    # Asian handicap calculation
    asian_handicap = calculator.calculate_asian_handicap(2.1, 1.8, -0.75)
    print(f"Asian handicap: {json.dumps(asian_handicap, indent=2)}")