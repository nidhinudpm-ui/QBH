import numpy as np

def compute_intervals(semitones):
    """Extract relative pitch differences."""
    if len(semitones) < 2:
        return np.array([])
    # Filter out unvoiced (0) transitions
    valid_diffs = []
    for i in range(len(semitones)-1):
        if semitones[i] > 0 and semitones[i+1] > 0:
            valid_diffs.append(semitones[i+1] - semitones[i])
    return np.clip(np.array(valid_diffs), -12, 12)

def compute_contour(intervals):
    """Directional shape (+1, -1, 0)."""
    return np.sign(intervals)

def compute_interval_histogram(intervals, bins=None, min_interval=1.0):
    """
    Transposition-invariant melodic profile.
    Filters out '0' intervals (local jitter) to focus on melody moves.
    """
    if len(intervals) == 0:
        return np.zeros(25) # -12 to +12
    
    # Filter for significant melodic movement
    moves = intervals[np.abs(intervals) >= min_interval]
    
    if len(moves) == 0:
        # Fallback: if no movement, return a peak at 0 to avoid empty histograms
        hist = np.zeros(25)
        hist[12] = 1.0 # Index 12 corresponds to interval 0
        return hist.astype(np.float32)

    hist, _ = np.histogram(moves, bins=np.arange(-12.5, 13.5))
    return hist.astype(np.float32) / (np.sum(hist) + 1e-6)

def compute_contour_histogram(contour):
    """Distribution of directional changes."""
    hist, _ = np.histogram(contour, bins=[-1.5, -0.5, 0.5, 1.5])
    return hist.astype(np.float32) / (np.sum(hist) + 1e-6)
