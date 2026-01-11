import sys, os, pandas as pd, numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'web_app'))
from src.model_trainer import ModelTrainer

# 50 fake rows
df = pd.DataFrame({
    'home_team': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City'], 50),
    'away_team': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City'], 50),
    'home_score': np.random.randint(0, 4, 50),
    'away_score': np.random.randint(0, 4, 50),
    'home_odds': np.random.uniform(1.5, 3.5, 50),
    'away_odds': np.random.uniform(1.5, 3.5, 50),
    'league': 'Premier League'
})
df['result'] = np.where(df.home_score > df.away_score, 0,
                        np.where(df.home_score < df.away_score, 2, 1))

# add dummy features
df['home_form'] = np.random.random(50)
df['away_form'] = np.random.random(50)
df['h2h_home_win_rate'] = np.random.random(50)
df['league_avg_goals'] = np.random.random(50)

trainer = ModelTrainer()
# disable feature selection so all 6 columns are used
trainer.config['feature_selection']['enable_selection'] = False

X = df[['home_odds', 'away_odds', 'home_form', 'away_form', 'h2h_home_win_rate', 'league_avg_goals']].values
y = df['result'].values

trainer.train_individual_model(X, y, X, y, 'random_forest', optimize_hyperparameters=False)
trainer.save_models('models')
print('✅ Model trained & saved – refresh homepage to see stats!')