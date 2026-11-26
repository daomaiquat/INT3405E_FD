"""Fold-local feature families for previously untested transaction relationships."""

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

GOAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOAL))
from eda_from_scratch import centered_pair, group_totals
from residual_models import fit_codes


DIGITS = {'amount_usd': 100, 'hours_since_last_txn': 100, 'distance_from_home_km': 100,
          'account_balance_usd': 100, 'velocity_score': 10, 'merchant_risk_score': 10}
PEERS = (
    ('velocity_score', ('txn_count_last_24h', 'elapsed_group', 'auth_method')),
    ('hours_since_last_txn', ('txn_count_last_24h', 'velocity_score', 'channel')),
    ('amount_usd', ('merchant_category', 'channel', 'account_balance_usd')),
    ('distance_from_home_km', ('channel', 'is_foreign_transaction', 'used_vpn')),
    ('merchant_risk_score', ('merchant_category', 'auth_method', 'billing_shipping_mismatch')),
    ('cvv_retry_count', ('auth_method', 'device_type', 'channel')),
    ('prior_disputes', ('card_age_months', 'customer_age', 'card_type')),
)


def physical(frame):
    amount = frame['amount_usd'].to_numpy()
    balance = frame['account_balance_usd'].to_numpy()
    hours = frame['hours_since_last_txn'].to_numpy()
    count = frame['txn_count_last_24h'].to_numpy()
    age = frame['customer_age'].to_numpy()
    months = frame['card_age_months'].to_numpy()
    velocity = frame['velocity_score'].to_numpy()
    hour = frame['time_of_day_hour'].to_numpy() + .5
    weekday = frame['day_of_week'].to_numpy()
    past_hour = np.remainder(hour - hours, 24)
    last_gap = np.minimum(hours / 24., .999)
    previous_phase = 2*np.pi*past_hour / 24
    return {
        'log_amount_balance_ratio': np.log1p(amount) - np.log1p(balance),
        'balance_net_scaled': (balance - amount) / (1 + balance + amount),
        'spend_exceeds_balance': (amount > balance).astype(float),
        'short_interval_empty_count': ((hours < 24) & (count == 0)).astype(float),
        'long_interval_positive_count': ((hours >= 24) & (count > 0)).astype(float),
        'count_times_relative_gap': np.minimum(count * -np.log1p(-last_gap), 20),
        'prior_events_per_available_hour': np.log1p(np.maximum(count - 1., 0.) / np.maximum(24 - hours, .25)),
        'velocity_per_transaction': np.log1p(velocity) - np.log1p(count),
        'velocity_times_gap': np.log1p(velocity) + np.log1p(hours),
        'implied_age_when_card_started': age - months / 12.,
        'card_predates_adulthood': (age - months / 12. < 18).astype(float),
        'previous_hour_sin': np.sin(previous_phase),
        'previous_hour_cos': np.cos(previous_phase),
        'previous_week_phase': np.remainder(24 * weekday + hour - hours, 168.) / 168.,
        'previous_day_crossed': np.floor((hours - hour) / 24.) + 1.,
        'ip_flag_disagreement': (frame['is_foreign_transaction'] != frame['ip_country_mismatch']).to_numpy(float),
    }


def physical_features(frame):
    return np.column_stack(tuple(physical(frame).values())).astype(np.float32)


def digits_for(frame):
    return {column: np.rint(frame[column].to_numpy() * scale).astype(np.int64) % 10
            for column, scale in DIGITS.items()}


def digit_tables(left, right, labels, offsets):
    probabilities = expit(offsets)
    g, h, n = group_totals(left * 10 + right, labels, probabilities, 100)
    cells = n.reshape(10, 10)
    expected = np.outer(cells.sum(axis=1), cells.sum(axis=0)) / len(labels)
    rarity = np.log((cells + 30) / (expected + 30))
    centered = centered_pair(g.reshape(10, 10), h.reshape(10, 10))
    contrast = np.clip(centered / (h.reshape(10, 10) + 25), -.45, .45)
    return contrast, rarity


def fingerprint_features(fit, labels, offsets, query):
    reference, scoring = digits_for(fit), digits_for(query)
    result = []
    for a, b in combinations(DIGITS, 2):
        tables = digit_tables(reference[a], reference[b], labels, offsets)
        result.extend(table[scoring[a], scoring[b]] for table in tables)
    return np.column_stack(result).astype(np.float32)


def peer_codes(frame, scoring):
    names = {column for variable, ctx in PEERS for column in (variable, *ctx)}
    edges = np.array([.25, 1, 3, 8, 16, 24])
    derived = (np.searchsorted(edges, frame['hours_since_last_txn']),
               np.searchsorted(edges, scoring['hours_since_last_txn']), len(edges) + 1)
    return {name: derived if name == 'elapsed_group' else fit_codes(frame[name], scoring[name]) for name in names}


def pack_context(columns, codes):
    fit, query, size = codes[columns[0]]
    for column in columns[1:]:
        left, right, width = codes[column]
        fit, query, size = fit * width + left, query * width + right, size * width
    return fit, query, size


def peer_features(fit, y, query):
    coded = peer_codes(fit, query)
    outputs = []
    for name, group in PEERS:
        value, score_value, width = coded[name]
        sub, score_sub, n_sub = pack_context(group, coded)
        parents = n_sub // coded[group[-1]][2]
        key = sub * width + value
        child_counts = [np.bincount(key[y == k], minlength=n_sub * width).reshape(n_sub, width) for k in (0, 1)]
        parent_group = np.arange(n_sub) // (n_sub // parents)
        prior_ratios, refined = [], []
        for counts, alpha in zip(child_counts, (200., 45.), strict=True):
            global_density = (counts.sum(axis=0) + .5) / (counts.sum() + .5 * width)
            parent_counts = counts.reshape(parents, -1, width).sum(axis=1)
            parent = (parent_counts + alpha * global_density) / (parent_counts.sum(axis=1, keepdims=True) + alpha)
            prior = parent[parent_group]
            child = (counts + alpha * prior) / (counts.sum(axis=1, keepdims=True) + alpha)
            prior_ratios.append(np.log(parent[score_sub // (n_sub // parents), score_value] / global_density[score_value]))
            refined.append(np.log(child[score_sub, score_value] / prior[score_sub, score_value]))
        combined = child_counts[0] + child_counts[1]
        global_density = (combined.sum(axis=0) + .5) / (combined.sum() + .5 * width)
        scoped = (combined + 80 * global_density) / (combined.sum(axis=1, keepdims=True) + 80)
        local_mass = scoped[score_sub, score_value]
        position = np.cumsum(scoped, axis=1)[score_sub, score_value] - .5 * local_mass
        outputs.extend((prior_ratios[1]-prior_ratios[0], refined[1]-refined[0],
                        np.log(local_mass / global_density[score_value]), position))
    return np.clip(np.column_stack(outputs), -4, 4).astype(np.float32)


def fold_fingerprints(fit, labels, offsets, query):
    scoring = fingerprint_features(fit, labels, offsets, query)
    training = np.empty((len(fit), scoring.shape[1]), dtype=np.float32)
    cv = StratifiedKFold(5, shuffle=True, random_state=2026)
    for a, b in cv.split(labels, labels):
        training[b] = fingerprint_features(fit.iloc[a], labels[a], offsets[a], fit.iloc[b])
    return training, scoring
