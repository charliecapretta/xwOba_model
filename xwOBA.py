import pandas as pd
import numpy as np
import glob
from sklearn.neighbors import KNeighborsRegressor

WOBA_WEIGHTS_BIP = {
    'Out': 0.0,
    'Sacrifice': 0.0,
    'Error': 0.0,
    'FieldersChoice': 0.0,
    'Single': 0.96,
    'Double': 1.37,
    'Triple': 1.68,
    'HomeRun': 2.12,
}
WALK_WEIGHT = 0.74
HBP_WEIGHT = 0.76
PA_ENDING_RESULTS = set(WOBA_WEIGHTS_BIP.keys()) | {'Walk', 'HitByPitch', 'Strikeout'}


def _resolve_play_result(df):
    """Return a Series with PlayResult overridden for walks, strikeouts, and HBP."""
    resolved = df['PlayResult'].copy()
    resolved[df['KorBB'] == 'Walk'] = 'Walk'
    resolved[df['KorBB'] == 'Strikeout'] = 'Strikeout'
    resolved[df['PitchCall'] == 'HitByPitch'] = 'HitByPitch'
    return resolved


class XwOBAModel:
    def __init__(self, n_neighbors=10):
        self._model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self._trained = False

    def train(self, historical_csv_path):
        df = pd.read_csv(historical_csv_path)
        df = df[['ExitSpeed', 'Angle', 'PlayResult', 'KorBB', 'PitchCall']].copy()

        df['resolved_result'] = _resolve_play_result(df)

        # Only train on balls in play with valid batted ball data
        bip = df[df['PitchCall'] == 'InPlay'].copy()
        bip = bip.dropna(subset=['ExitSpeed', 'Angle'])
        bip['woba_val'] = bip['resolved_result'].map(WOBA_WEIGHTS_BIP)
        bip = bip.dropna(subset=['woba_val'])

        X = bip[['ExitSpeed', 'Angle']].values
        y = bip['woba_val'].values
        self._model.fit(X, y)
        self._trained = True

    def getXwOBA(self, exit_speeds, launch_angles):
        """Predict mean xwOBA for a set of batted balls (BIP only, ignores NaN rows)."""
        combined = pd.DataFrame({'ExitSpeed': exit_speeds, 'Angle': launch_angles})
        valid = combined.dropna()
        if valid.empty:
            return 0.0
        preds = self._model.predict(valid[['ExitSpeed', 'Angle']].values)
        return float(np.mean(preds))

    def get_batter_xwoba(self, batter_df):
        """Calculate xwOBA for a batter over all PA-ending events."""
        df = batter_df[['ExitSpeed', 'Angle', 'PlayResult', 'KorBB', 'PitchCall']].copy()
        df['resolved_result'] = _resolve_play_result(df)

        pa_events = df[df['resolved_result'].isin(PA_ENDING_RESULTS)].copy()
        if pa_events.empty:
            return 0.0

        xwoba_values = []
        for _, row in pa_events.iterrows():
            result = row['resolved_result']
            if result == 'Walk':
                xwoba_values.append(WALK_WEIGHT)
            elif result == 'HitByPitch':
                xwoba_values.append(HBP_WEIGHT)
            elif result == 'Strikeout':
                xwoba_values.append(0.0)
            else:
                # Ball in play
                ev, la = row['ExitSpeed'], row['Angle']
                if pd.isna(ev) or pd.isna(la):
                    xwoba_values.append(0.0)
                else:
                    pred = self._model.predict([[ev, la]])[0]
                    xwoba_values.append(float(pred))

        return sum(xwoba_values) / len(xwoba_values)

    def get_players_xwoba(self, season_df, team_code=None):
        """Return dict of {batter_name: xwoba} for all batters in season_df."""
        df = season_df.copy()
        if team_code is not None:
            df = df[df['BatterTeam'] == team_code]

        results = {}
        for batter, batter_df in df.groupby('Batter'):
            results[batter] = round(self.get_batter_xwoba(batter_df), 3)
        return results

#Input CSV
HISTORICAL_CSV = 'Trackman.csv'
xwoba_model = XwOBAModel(n_neighbors=10)
xwoba_model.train(HISTORICAL_CSV)


def get_players_xwOBA(season_df=None, team_code='GEO_COL1'):
    if season_df is None:
        files = glob.glob('Trackman*.csv')
        season_df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return xwoba_model.get_players_xwoba(season_df, team_code=team_code)

