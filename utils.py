import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline

def interpolate_gaps(data, fps, max_gap):
    n_keypoints, n_dims, n_frames = data.shape
    max_gap_frames = int(max_gap * fps)

    filled_data = np.copy(data)  # Preserve original data

    for kpt in range(n_keypoints):
        for dim in range(n_dims):
            signal = data[kpt, dim, :]
            nan_indices = np.where(np.isnan(signal))[0]

            if len(nan_indices) == 0:
                continue

            # Identify NaN segments
            diff = np.diff(nan_indices)
            segment_starts = np.insert(nan_indices[np.where(diff > 1)[0] + 1], 0, nan_indices[0])
            segment_ends = np.append(nan_indices[np.where(diff > 1)[0]], nan_indices[-1])

            # Interpolate gaps within the allowed size
            for start, end in zip(segment_starts, segment_ends):
                gap_size = end - start + 1
                if gap_size <= max_gap_frames:
                    valid_indices = np.where(~np.isnan(signal))[0]
                    if len(valid_indices) >= 2:
                        cs = CubicSpline(valid_indices, signal[valid_indices])
                        filled_data[kpt, dim, start:end + 1] = cs(np.arange(start, end + 1))
                    else:
                        non_nan_idx = np.where(~np.isnan(signal))[0]
                        filled_data[kpt, dim, :] = np.interp(
                            np.arange(n_frames), non_nan_idx, signal[non_nan_idx]
                        )

    return filled_data

def filter_data(data, sampling_rate, cutoff, order, gap_size):
    n_kpt, n_dims, n_frames = data.shape
    data = interpolate_gaps(data, sampling_rate, gap_size)  # Fill short gaps
    filtered_data = data.copy()

    # Design Butterworth low-pass filter
    b, a = butter(N=order, Wn=cutoff / (0.5 * sampling_rate), btype='low', analog=False)
    
    for kpt in range(n_kpt):
        for dim in range(n_dims):
            trajectory = data[kpt, dim, :]
            nans = np.isnan(trajectory)

            if np.any(nans):  # Interpolate missing values
                valid_indices = ~nans
                trajectory[nans] = np.interp(
                    np.flatnonzero(nans),
                    np.flatnonzero(valid_indices),
                    trajectory[valid_indices]
                )
            
            trajectory = filtfilt(b, a, trajectory)  # Apply filter
            trajectory[nans] = np.nan  # Restore NaNs
            
            filtered_data[kpt, dim, :] = trajectory

    return filtered_data

def step_detection(data, kpt_labels, properties):
    perspective = walk_direction(data, kpt_labels, properties['stride_min'], properties['fps'])
    events = {'left': {}, 'right': {}}

    for side in ['left', 'right']:
        _, _, velocity = bruening_ridge_detection(data, 1, side, kpt_labels, properties)
        ICs, FCs, _ = bruening_ridge_detection(data, velocity, side, kpt_labels, properties)
        events[side]['ICs'] = ICs
        events[side]['FCs'] = FCs

    IC_events = [
        {'frame': frame, 'side': side, 'perspective': perspective[frame]}
        for side in ['left', 'right']
        for frame in events[side]['ICs']
    ]
    FC_events = [
        {'frame': frame, 'side': side, 'perspective': perspective[frame]}
        for side in ['left', 'right']
        for frame in events[side]['FCs']
    ]

    return {'IC': IC_events, 'FC': FC_events}

def walk_direction(data, keypoint_mapping, min_length, fps):
    shoulder_R = data[keypoint_mapping.index('right_shoulder'), 1, :]
    shoulder_L = data[keypoint_mapping.index('left_shoulder'), 1, :]

    orientation = shoulder_R - shoulder_L
    perspective = np.where(orientation > 0, 'frontal', np.where(orientation < 0, 'saggital', None)).astype(object)

    max_gap_size = int(min_length * fps)
    valid_indices = np.where(perspective != None)[0]

    for start, end in zip(valid_indices[:-1], valid_indices[1:]):
        if end - start <= max_gap_size:
            perspective[start + 1:end] = perspective[start]

    changes = np.r_[True, perspective[:-1] != perspective[1:], True]
    segment_starts, segment_ends = np.where(changes[:-1])[0], np.where(changes[1:])[0] - 1

    for start, end in zip(segment_starts, segment_ends):
        if (end - start + 1) / fps < min_length:
            perspective[start:end + 1] = None

    return perspective

def bruening_ridge_detection(data, velocity, side, kpt_labels, properties):
    fs = properties['fps']
    heel_thr = properties['heel_thr'] * velocity
    ankle_thr = properties['heel_thr'] * velocity
    big_toe_thr = properties['toe_thr'] * velocity

    # Extract trajectories
    heel = data[kpt_labels.index(f'{side}_heel'), :, :]
    big_toe = data[kpt_labels.index(f'{side}_big_toe'), :, :]
    ankle = data[kpt_labels.index(f'{side}_ankle'), :, :]

    # Compute 3D velocities
    heel_vel = np.linalg.norm(np.diff(heel, axis=1), axis=0) * fs
    ankle_vel = np.linalg.norm(np.diff(ankle, axis=1), axis=0) * fs
    big_toe_vel = np.linalg.norm(np.diff(big_toe, axis=1), axis=0) * fs

    # Ground contact detection based on thresholds
    ground_contact = (
        (heel_vel < heel_thr) |
        (ankle_vel < ankle_thr) |
        (big_toe_vel < big_toe_thr)
    ).astype(int)

    # Remove short ground contact periods
    min_gc_duration = int(properties['stance_min'] * fs)
    min_no_gc_duration = int(properties['swing_min'] * fs)

    contact_diff = np.diff(np.r_[0, ground_contact, 0])
    starts = np.where(contact_diff == 1)[0]
    ends = np.where(contact_diff == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_gc_duration:
            ground_contact[start:end] = 0

    # Remove short no-contact periods
    contact_diff = np.diff(np.r_[0, ground_contact, 0])
    starts = np.where(contact_diff == 1)[0]
    ends = np.where(contact_diff == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_no_gc_duration:
            ground_contact[start:end] = 1

    # Identify initial contacts (ICs) and final contacts (FCs)
    ground_contact_diff = np.diff(ground_contact)
    ICs = np.where(ground_contact_diff == 1)[0] + 1
    FCs = np.where(ground_contact_diff == -1)[0] + 1

    # Compute walking velocity from stride lengths and durations
    stride_lengths = []
    stride_durations = []

    for i in range(1, len(ICs)):
        stride_duration = (ICs[i] - ICs[i - 1]) / fs
        if properties['stride_min'] <= stride_duration <= properties['stride_max']:
            stride_length = np.linalg.norm(heel[:, ICs[i]] - heel[:, ICs[i - 1]])
            stride_lengths.append(stride_length)
            stride_durations.append(stride_duration)

    velocity = (
        np.nanmean(np.array(stride_lengths) / np.array(stride_durations))
        if stride_durations else np.nan
    )

    return ICs, FCs, velocity


def gait_analysis(data, events, keypoint_mapping, properties):
    fs = properties['fps']
    stride_min = properties['stride_min']
    stride_max = properties['stride_max']

    perspectives = ['all', 'frontal', 'saggital']
    metrics = {
        perspective: {
            'left': {metric: [] for metric in ['stime', 'slen', 'vel', 'swing', 'dsupp', 'bos', 'fpa', 'arm_rom', 'knee_rom']},
            'right': {metric: [] for metric in ['stime', 'slen', 'vel', 'swing', 'dsupp', 'bos', 'fpa', 'arm_rom', 'knee_rom']}
        } for perspective in perspectives
    }

    ICs = events['IC']
    FCs = events['FC']

    def get_frame(struct_array, side, lower_bound, upper_bound):
        return [e['frame'] for e in struct_array if e['side'] == side and lower_bound < e['frame'] < upper_bound]

    for i, IC in enumerate(ICs):
        ipsi, contra = IC['side'], 'left' if IC['side'] == 'right' else 'right'
        perspective = IC.get('perspective', 'all')

        if perspective not in perspectives:
            continue

        same_foot_idx = next((j for j, event in enumerate(ICs[i + 1:], start=i + 1) if event['side'] == ipsi), None)

        if same_foot_idx is not None:
            stime = (ICs[same_foot_idx]['frame'] - IC['frame']) / fs
            if stride_min <= stime <= stride_max:
                IC0, IC2 = IC['frame'], ICs[same_foot_idx]['frame']
                IC1 = get_frame(ICs, contra, IC0, IC2)
                FC0, FC1, FC2 = None, None, None

                if IC1:
                    IC1 = IC1[0]
                    FC0 = get_frame(FCs, contra, IC0, IC1)
                    FC1 = get_frame(FCs, ipsi, IC1, IC2)
                    FC2 = get_frame(FCs, ipsi, IC2, IC2 + int(fs * stride_max))

                    FC0 = FC0[0] if FC0 else None
                    FC1 = FC1[0] if FC1 else None
                    FC2 = FC2[0] if FC2 else None

                if any(x is None for x in [IC0, IC1, IC2, FC0, FC1, FC2]):
                    continue

                HP0 = np.nanmedian(data[keypoint_mapping.index(f'{ipsi}_heel'), :, IC0:FC0], axis=1)
                HP2 = np.nanmedian(data[keypoint_mapping.index(f'{ipsi}_heel'), :, IC2:FC2], axis=1)
                TP1 = np.nanmedian(data[keypoint_mapping.index(f'{contra}_big_toe'), :, IC1:IC2], axis=1)
                HP1 = np.nanmedian(data[keypoint_mapping.index(f'{contra}_heel'), :, IC1:FC1], axis=1)

                slen = np.linalg.norm(HP2[:2] - HP0[:2])
                vel = slen / stime
                bos = np.linalg.norm(np.cross(HP2 - HP1, HP1 - HP0)) / np.linalg.norm(HP2 - HP1)

                n1 = (HP2[:2] - HP0[:2]) / np.linalg.norm(HP2[:2] - HP0[:2])
                n2 = (TP1[:2] - HP0[:2]) / np.linalg.norm(TP1[:2] - HP0[:2])
                fpa = np.degrees(np.arctan2(np.linalg.norm(np.cross(n2, n1)), np.dot(n1, n2)))

                swing = (IC2 - FC1) / fs
                dsupp = ((FC0 - IC0) + (FC1 - IC1)) / fs

                shoulder_data = data[keypoint_mapping.index(f'{contra}_shoulder'), :, IC0:IC2]
                wrist_data = data[keypoint_mapping.index(f'{contra}_wrist'), :, IC0:IC2]
                arm_rom = np.nan if np.isnan(shoulder_data).any() or np.isnan(wrist_data).any() else np.ptp(
                    np.degrees(np.arctan2(
                        np.dot(wrist_data.T - shoulder_data.T, np.array([0, 0, 1])),
                        np.dot(wrist_data.T - shoulder_data.T, np.array([1, 0, 0]))
                    ))
                )

                hip_data = data[keypoint_mapping.index(f'{ipsi}_hip'), :, IC0:IC2]
                knee_data = data[keypoint_mapping.index(f'{ipsi}_knee'), :, IC0:IC2]
                ankle_data = data[keypoint_mapping.index(f'{ipsi}_ankle'), :, IC0:IC2]
                knee_rom = np.nan if np.isnan(hip_data).any() or np.isnan(knee_data).any() or np.isnan(ankle_data).any() else \
                    np.ptp(np.degrees(np.arctan2(
                        np.linalg.norm(knee_data - hip_data, axis=0),
                        np.linalg.norm(ankle_data - knee_data, axis=0)
                    )))

                for pers in ['all', perspective]:
                    metrics[pers][ipsi]['stime'].append(stime)
                    metrics[pers][ipsi]['slen'].append(slen)
                    metrics[pers][ipsi]['vel'].append(vel)
                    metrics[pers][ipsi]['swing'].append(swing)
                    metrics[pers][ipsi]['dsupp'].append(dsupp)
                    metrics[pers][ipsi]['bos'].append(bos)
                    metrics[pers][ipsi]['fpa'].append(fpa)
                    metrics[pers][ipsi]['arm_rom'].append(arm_rom)
                    metrics[pers][ipsi]['knee_rom'].append(knee_rom)

    def compute_pooled_stats(left_values, right_values):
        pooled = np.concatenate([left_values, right_values])
        pooled = pooled[~np.isnan(pooled)]
        mean = np.mean(pooled) if len(pooled) > 0 else np.nan
        cv = 100 * np.std(pooled) / mean if mean != 0 else np.nan
        return {'mean': mean, 'CV': cv}

    def compute_asymmetry(left_values, right_values):
        left, right = map(lambda x: np.array(x)[~np.isnan(x)], [left_values, right_values])
        if len(left) == 0 or len(right) == 0:
            return np.nan
        larger, smaller = max(left.mean(), right.mean()), min(left.mean(), right.mean())
        return 100 * (1 - smaller / larger) if larger > 0 else np.nan

    parameters = {}
    for perspective, data in metrics.items():
        parameters[perspective] = {
            metric: {
                **compute_pooled_stats(data['left'][metric], data['right'][metric]),
                'asymmetry': compute_asymmetry(data['left'][metric], data['right'][metric])
            } for metric in data['left']
        }

    return parameters

def display_results(parameters):
    parameter_order = ['stime', 'slen', 'vel', 'swing', 'dsupp', 'bos', 'fpa', 'arm_rom', 'knee_rom']
    statistic_order = ['Mean', 'CV', 'Asymmetry']

    rows = []
    for perspective, metrics in parameters.items():
        for param, values in metrics.items():
            rows.extend([
                {'Parameter': param, 'Statistic': 'Mean', 'Perspective': perspective, 'Value': values['mean']},
                {'Parameter': param, 'Statistic': 'CV', 'Perspective': perspective, 'Value': values['CV']},
                {'Parameter': param, 'Statistic': 'Asymmetry', 'Perspective': perspective, 'Value': values['asymmetry']}
            ])

    df = pd.DataFrame(rows)
    df['Parameter'] = pd.Categorical(df['Parameter'], categories=parameter_order, ordered=True)
    df['Statistic'] = pd.Categorical(df['Statistic'], categories=statistic_order, ordered=True)
    df = df.sort_values(by=['Parameter', 'Statistic'])

    table = df.pivot_table(index=['Parameter', 'Statistic'], columns='Perspective', values='Value', aggfunc='mean')
    table = table.reset_index()
    table.columns.name = None

    print(table)
    return table

