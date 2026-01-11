import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'web_app'))
from src.data_processor import DataProcessor
from datetime import datetime, timedelta
import pandas as pd

dp = DataProcessor()
matches = pd.DataFrame([
    {'match_id':'DUMMY1','date':datetime.now()+timedelta(hours=2),
     'home_team':'Arsenal','away_team':'Chelsea','league':'Premier League',
     'home_odds':2.5,'draw_odds':3.2,'away_odds':2.8},
    {'match_id':'DUMMY2','date':datetime.now()+timedelta(hours=4),
     'home_team':'Liverpool','away_team':'Man City','league':'Premier League',
     'home_odds':2.1,'draw_odds':3.4,'away_odds':3.1}
])
dp.store_data(matches)
print('✅ 2 dummy matches injected – refresh browser!')