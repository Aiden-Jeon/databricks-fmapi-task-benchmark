"""Feature engineering for PUBG winPlacePerc prediction."""
import pandas as pd
import numpy as np


# Numeric features that represent a player's in-game actions
NUMERIC_FEATURES = [
    'assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals',
    'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill',
    'matchDuration', 'maxPlace', 'numGroups', 'rankPoints', 'revives',
    'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
    'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints',
]


def _match_type_clean(mt: pd.Series) -> pd.Series:
    """Collapse many matchType variants into core modes (solo/duo/squad/other)."""
    mapping = {
        'solo-fpp': 'solo', 'solo': 'solo',
        'duo-fpp': 'duo', 'duo': 'duo',
        'squad-fpp': 'squad', 'squad': 'squad',
    }
    return mt.map(mapping).fillna('other')


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Build engineered features. Returns (train_feat, test_feat, target).

    Features include:
      - raw player stats
      - within-match normalized stats (per-match mean/std scaling)
      - within-group (team) aggregated stats (sum/max/mean) since teammates share target
      - killPlace normalized by match size (proxy for placement)
      - distance totals and ratios
      - matchType categorical
    """
    train = train_df.copy()
    test = test_df.copy()
    target = train['winPlacePerc'].copy()
    train = train.drop(columns=['winPlacePerc'])

    train['__is_train'] = 1
    test['__is_train'] = 0
    full = pd.concat([train, test], ignore_index=True, sort=False)

    # ---- matchType cleaning ----
    full['matchTypeClean'] = _match_type_clean(full['matchType'])

    # ---- distance totals ----
    full['totalDistance'] = (
        full['walkDistance'] + full['rideDistance'] + full['swimDistance']
    )
    # Avoid divide-by-zero
    full['walkPct'] = full['walkDistance'] / (full['totalDistance'] + 1.0)
    full['ridePct'] = full['rideDistance'] / (full['totalDistance'] + 1.0)

    # ---- headshot ratio ----
    full['headshotRate'] = full['headshotKills'] / (full['kills'] + 1.0)

    # ---- per-player kill efficiency ----
    full['killsPerDamage'] = full['kills'] / (full['damageDealt'] + 1.0)

    # ---- within-match normalization ----
    # matchId-based groupby for match-level stats
    match_cols = ['walkDistance', 'damageDealt', 'kills', 'boosts', 'heals',
                 'rideDistance', 'totalDistance', 'longestKill', 'killPlace']
    # Use transform for match-level mean/std to broadcast back to each player
    for c in match_cols:
        m_mean = full.groupby('matchId')[c].transform('mean')
        m_std = full.groupby('matchId')[c].transform('std').replace(0, 1)
        full[f'{c}_match_mean'] = m_mean
        full[f'{c}_match_z'] = (full[c] - m_mean) / m_std

    # ---- killPlace normalized by match size (strong proxy for placement) ----
    # killPlace is a within-match ranking (1 = most kills). Normalize to [0,1].
    match_size = full.groupby('matchId')['Id'].transform('count')
    full['killPlace_pct'] = (full['killPlace'] - 1) / (match_size - 1).clip(lower=1)
    # Higher killPlace_pct = lower kills = worse. So flip sign sense:
    # we keep as is; model can learn direction. Also create 1 - pct.
    full['killPlace_pct_inv'] = 1 - full['killPlace_pct']

    # ---- within-group (team) aggregations ----
    # Teammates share winPlacePerc, so team aggregates are very predictive.
    team_cols = ['walkDistance', 'damageDealt', 'kills', 'boosts', 'heals',
                 'rideDistance', 'swimDistance', 'totalDistance', 'assists',
                 'DBNOs', 'revives', 'weaponsAcquired', 'longestKill']
    for c in team_cols:
        tg = full.groupby(['matchId', 'groupId'])[c]
        full[f'{c}_team_sum'] = tg.transform('sum')
        full[f'{c}_team_max'] = tg.transform('max')
        full[f'{c}_team_mean'] = tg.transform('mean')

    # team size
    full['teamSize'] = full.groupby(['matchId', 'groupId'])['Id'].transform('count')

    # ---- group normalized by match: player's stat as fraction of team total ----
    for c in ['walkDistance', 'damageDealt', 'kills']:
        full[f'{c}_team_frac'] = full[c] / (full[f'{c}_team_sum'] + 1.0)

    # ---- match-level count features ----
    full['matchPlayerCount'] = match_size

    # ---- drop raw ids ----
    drop_cols = ['Id', 'groupId', 'matchId', '__is_train']
    feat = full.drop(columns=drop_cols)

    # ---- encode matchType categorically ----
    feat['matchType'] = feat['matchType'].astype('category')
    feat['matchTypeClean'] = feat['matchTypeClean'].astype('category')

    train_feat = feat[full['__is_train'] == 1].reset_index(drop=True)
    test_feat = feat[full['__is_train'] == 0].reset_index(drop=True)
    return train_feat, test_feat, target
