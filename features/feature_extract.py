import numpy as np
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import welch
from typing import Dict, List

def extract_features_single_axis(signal: np.ndarray, fs: int = 64) -> Dict[str, float]:
    N = len(signal)
    t = np.arange(N) / fs

    f, Pxx = welch(signal, fs=fs, nperseg=N)
    Pxx_norm = Pxx / np.sum(Pxx + 1e-8)  # 0으로 나누는 오류 방지

    # Freeze Index 계산용 주파수 밴드
    fog_band = (f >= 3) & (f <= 8)
    non_fog_band = (f >= 0.5) & (f < 3)
    power_fog = np.sum(Pxx[fog_band])
    power_non_fog = np.sum(Pxx[non_fog_band])

    return {
        'mean': np.mean(signal),
        'variance': np.var(signal),
        'std_dev': np.std(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'mav': np.mean(np.abs(signal)),
        'kurtosis': kurtosis(signal),
        'skewness': skew(signal),
        'sma': np.sum(np.abs(signal)),
        'slope': (signal[-1] - signal[0]) / (t[-1] - t[0] + 1e-8),
        'entropy': entropy(Pxx_norm, base=2),
        'energy': np.sum(np.abs(signal)**2),
        'peak_freq': f[np.argmax(Pxx)],
        'freeze_index': power_fog / (power_non_fog + 1e-8),
        'total_power': power_fog + power_non_fog,
    }

def extract_features_all_axes(data: Dict[str, np.ndarray], fs: int = 64) -> Dict[str, float]:
    all_features = {}
    for sensor_type in ['acc', 'gyro']:
        for axis in ['x', 'y', 'z']:
            key = f'{sensor_type}_{axis}'
            axis_feats = extract_features_single_axis(data[key], fs)
            for feat_name, value in axis_feats.items():
                all_features[f'{key}_{feat_name}'] = value
    return all_features

def extract_features_sliding_windows(
    data: Dict[str, np.ndarray], fs: int = 64
) -> List[Dict[str, float]]:
    
    window_size = 128            # 2초 = 128샘플 (fs=64)
    step_size = window_size // 2 # 50% 오버래핑 = 64샘플 슬라이딩

    length = len(data['acc_x'])  # 모든 축 동일한 길이라고 가정
    features_list = []

    for start in range(0, length - window_size + 1, step_size):
        window_data = {
            key: signal[start:start + window_size]
            for key, signal in data.items()
        }
        features = extract_features_all_axes(window_data, fs)
        features_list.append(features)

    return features_list

