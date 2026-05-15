"""Reward constants for the ride-pooling environment."""

SOLO_REWARD = 0.0
"""Baseline reward for solo trips."""

W_VMT_SAVED = 5.0
"""Weight on vehicle miles saved when sharing."""

W_COST_SAVED = 1.0
"""Weight on cost savings to riders when sharing."""

SHARED_DISCOUNT = 0.65
"""Multiplier on the solo fare for a successful shared trip (35% discount)."""

VMT_REJECT_PEN = 1000.0
"""Penalty when a pairing would increase total fleet miles."""

INVALID_ACT_PEN = 100.0
"""Penalty for selecting an unavailable or out-of-window candidate."""

FARE_PER_MILE = 9.25
"""Per-mile fare in USD."""

TRAVEL_SPEED_MPH = 20.0
"""Assumed travel speed used to convert miles to minutes for delay stats."""
