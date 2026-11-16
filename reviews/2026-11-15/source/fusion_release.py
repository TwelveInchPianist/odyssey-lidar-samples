"""Immutable capability ceiling for each staged FCW release."""
FEATURE_LEVEL = 4
FEATURE_NAMES = ('independent_corridor','radar_fusion','camera_fusion','map_prior')

def capabilities():
    return {name:i<FEATURE_LEVEL for i,name in enumerate(FEATURE_NAMES)}
