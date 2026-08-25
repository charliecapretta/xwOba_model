"""Expected wOBA (xwOBA) from TrackMan batted-ball data.

Reference formula
-----------------
wOBA is a scaled linear-weights rate stat:

    wOBA = (wBB*uBB + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR)
           / (AB + BB - IBB + SF + HBP)

Two things about the denominator are easy to get wrong and are handled
explicitly below:

  * Sacrifice *flies* (SF) count in the denominator (numerator 0).
    Sacrifice *bunts* (SH) are excluded from wOBA entirely.
  * Intentional walks (IBB) are excluded from both numerator and
    denominator; only unintentional walks (uBB) are credited.

Reached-on-error and fielder's choice are plain outs for wOBA purposes:
they are at-bats, so they sit in the denominator with a numerator of 0.

xwOBA keeps that structure and changes exactly one thing: for a ball in
play, the actual outcome is replaced by the value the batted ball was
*expected* to produce given its exit velocity and launch angle. Walks,
hit-by-pitches and strikeouts are not modeled -- they carry their actual
wOBA weights, because the batter's contact quality has nothing to do with
them.
"""

import glob

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Scaled linear weights. These must all come from a SINGLE league-season
# weight table -- mixing vintages (e.g. a 2015 wHR with a 2023 w1B) silently
# distorts every number the model produces. The values below are the
# college (NCAA D1) run environment the Trackman exports come from, which
# runs hotter than MLB; swap in the matching table if you change leagues.
WOBA_WEIGHTS = {
    'Walk': 0.74,          # unintentional walks only
    'HitByPitch': 0.76,
    'Single': 0.96,
    'Double': 1.37,
    'Triple': 1.68,
    'HomeRun': 2.12,
    # Zero-value events that still occupy a denominator slot.
    'Out': 0.0,
    'Strikeout': 0.0,
    'Error': 0.0,
    'FieldersChoice': 0.0,
    'Sacrifice': 0.0,
}

# Outcomes of a ball in play, i.e. the ones the EV/LA model predicts.
BIP_RESULTS = frozenset(
    ['Out', 'Sacrifice', 'Error', 'FieldersChoice',
     'Single', 'Double', 'Triple', 'HomeRun']
)

# Everything that ends a plate appearance and belongs in the wOBA denominator.
PA_ENDING_RESULTS = BIP_RESULTS | {'Walk', 'HitByPitch', 'Strikeout'}

REQUIRED_COLUMNS = ['ExitSpeed', 'Angle', 'PlayResult', 'KorBB', 'PitchCall']
# Present in a standard TrackMan export but tolerated as missing.
OPTIONAL_COLUMNS = ['TaggedHitType']


def _validate_weights(weights):
    """Guard against a typo'd or mixed-vintage weight table."""
    ordered = ['Walk', 'HitByPitch', 'Single', 'Double', 'Triple', 'HomeRun']
    values = [weights[k] for k in ordered]
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(
            'wOBA weights must increase monotonically from Walk to HomeRun; '
            'got {}'.format(dict(zip(ordered, values)))
        )


_validate_weights(WOBA_WEIGHTS)


def _column(df, name, default):
    """Return df[name], or a constant Series if the column is absent."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _prepare(df):
    """Normalize a raw TrackMan frame into the fields the model reasons about.

    Adds:
      resolved_result -- PlayResult with walks / strikeouts / HBP folded in
      is_bunt         -- batted ball tagged as a bunt
      is_ibb          -- walk whose final pitch was intentional
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError('missing required TrackMan columns: {}'.format(missing))

    out = df[REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]].copy()

    korbb = out['KorBB']
    pitch_call = out['PitchCall']

    resolved = out['PlayResult'].astype('object').copy()
    resolved[korbb == 'Walk'] = 'Walk'
    resolved[korbb == 'Strikeout'] = 'Strikeout'
    resolved[pitch_call == 'HitByPitch'] = 'HitByPitch'
    out['resolved_result'] = resolved

    out['is_bunt'] = _column(out, 'TaggedHitType', None) == 'Bunt'
    # A TrackMan PA ends on the pitch that produced it, so an intentional
    # ball as the walk-ending pitch identifies an IBB.
    out['is_ibb'] = (resolved == 'Walk') & (pitch_call == 'BallIntentional')

    return out


def _denominator_mask(df):
    """Rows that count as (AB + BB - IBB + SF + HBP)."""
    result = df['resolved_result']
    is_bip = result.isin(BIP_RESULTS)

    # A stale PlayResult on a non-terminal pitch must not create a phantom PA.
    valid_bip = is_bip & (df['PitchCall'] == 'InPlay')
    valid_other = result.isin(['Walk', 'HitByPitch', 'Strikeout'])

    # Sacrifice bunts (SH) are excluded from wOBA; sacrifice flies are not.
    sac_bunt = (result == 'Sacrifice') & df['is_bunt']

    return (valid_bip | valid_other) & ~sac_bunt & ~df['is_ibb']


class XwOBAModel:
    """KNN model of expected wOBA value per batted ball, from EV and LA."""

    def __init__(self, n_neighbors=10):
        # ExitSpeed spans ~0-120 mph and Angle ~-90 to +90 degrees. On raw
        # units a Euclidean neighbor search treats 1 mph as interchangeable
        # with 1 degree, which is not a meaningful distance. Standardizing
        # first puts both features on comparable footing.
        self._model = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=n_neighbors),
        )
        self._trained = False

    def _check_trained(self):
        if not self._trained:
            raise RuntimeError('call train() before requesting predictions')

    def train(self, historical_csv_path):
        df = _prepare(pd.read_csv(historical_csv_path))

        # Train only on real, measured batted balls. Bunts are excluded: their
        # EV/LA profile overlaps weak topped grounders, so including them
        # corrupts the neighborhood for ordinary contact.
        bip = df[(df['PitchCall'] == 'InPlay') & ~df['is_bunt']].copy()
        bip = bip.dropna(subset=['ExitSpeed', 'Angle'])
        bip['woba_val'] = bip['resolved_result'].map(WOBA_WEIGHTS)
        bip = bip.dropna(subset=['woba_val'])

        if bip.empty:
            raise ValueError('no usable batted balls in {}'.format(historical_csv_path))

        self._model.fit(bip[['ExitSpeed', 'Angle']].to_numpy(), bip['woba_val'].to_numpy())
        self._trained = True
        return self

    def predict_contact(self, exit_speeds, launch_angles):
        """Expected wOBA value for each batted ball. NaN where EV/LA is missing."""
        self._check_trained()
        frame = pd.DataFrame({
            'ExitSpeed': pd.Series(exit_speeds, dtype='float64').reset_index(drop=True),
            'Angle': pd.Series(launch_angles, dtype='float64').reset_index(drop=True),
        })
        preds = pd.Series(np.nan, index=frame.index, dtype='float64')
        usable = frame.notna().all(axis=1)
        if usable.any():
            preds[usable] = self._model.predict(frame.loc[usable].to_numpy())
        return preds

    def getXwOBA(self, exit_speeds, launch_angles):
        """Mean expected wOBA *on contact* (xwOBAcon) for a set of batted balls.

        This is not xwOBA: it has no walks, hit-by-pitches or strikeouts in it.
        Use get_batter_xwoba() for the full rate stat.
        """
        preds = self.predict_contact(exit_speeds, launch_angles).dropna()
        if preds.empty:
            return 0.0
        return float(preds.mean())

    def _expected_values(self, pa_events):
        """Numerator contribution of each plate appearance."""
        # Default to the actual outcome's weight. This covers walks, HBP and
        # strikeouts, and is also the right fallback for a ball in play with
        # no tracked EV/LA -- crediting those 0.0 would book untracked hits
        # as outs and bias a batter's xwOBA downward.
        values = pa_events['resolved_result'].map(WOBA_WEIGHTS).astype('float64')

        modelable = (
            pa_events['resolved_result'].isin(BIP_RESULTS)
            & ~pa_events['is_bunt']
            & pa_events[['ExitSpeed', 'Angle']].notna().all(axis=1)
        )
        if modelable.any():
            predicted = self._model.predict(
                pa_events.loc[modelable, ['ExitSpeed', 'Angle']].to_numpy()
            )
            values[modelable] = predicted

        return values.fillna(0.0)

    def get_batter_xwoba(self, batter_df):
        """xwOBA for one batter over every plate appearance in batter_df."""
        self._check_trained()
        df = _prepare(batter_df)
        pa_events = df[_denominator_mask(df)]
        if pa_events.empty:
            return 0.0
        return float(self._expected_values(pa_events).sum() / len(pa_events))

    def get_players_xwoba(self, season_df, team_code=None):
        """Return {batter_name: xwoba} for every batter in season_df."""
        self._check_trained()
        df = season_df
        if team_code is not None:
            df = df[df['BatterTeam'] == team_code]

        return {
            batter: round(self.get_batter_xwoba(batter_df), 3)
            for batter, batter_df in df.groupby('Batter')
        }


# Input CSV
HISTORICAL_CSV = 'Trackman.csv'

_model_cache = {}


def get_model(historical_csv_path=HISTORICAL_CSV, n_neighbors=10):
    """Train (once) and return the shared model."""
    key = (historical_csv_path, n_neighbors)
    if key not in _model_cache:
        _model_cache[key] = XwOBAModel(n_neighbors=n_neighbors).train(historical_csv_path)
    return _model_cache[key]


def get_players_xwOBA(season_df=None, team_code='GEO_COL1'):
    if season_df is None:
        files = glob.glob('Trackman*.csv')
        if not files:
            raise FileNotFoundError('no Trackman*.csv files found')
        season_df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return get_model().get_players_xwoba(season_df, team_code=team_code)
