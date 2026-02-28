import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


def align_frames(video: List[Image.Image], target_frame_num: int) -> List[Image.Image]:
    """
    Align the number of frames in the video to the target_frame_num by either tail-clipping or padding.

    - If len(video) > target_frame_num: clip from the tail.
    - If len(video) < target_frame_num: pad by repeating the last frame.
    - If len(video) == target_frame_num: return a shallow copy.

    Args:
        video: List of PIL.Image frames.
        target_frame_num: Desired number of frames (must be non-negative).

    Returns:
        A new list of frames with length exactly target_frame_num.
    """
    n = len(video)
    # Trivial case: target is zero, return empty list regardless of input
    if target_frame_num == 0:
        return []
    if n == 0:
        # Cannot pad to a positive number of frames without a reference frame
        raise ValueError(
            "Cannot align to a positive target_frame_num when the input video is empty."
        )
    if n == target_frame_num:
        # Already aligned; return a shallow copy
        return video.copy()
    if n > target_frame_num:
        # Tail-clipping
        return video[:target_frame_num]
    # n < target_frame_num: pad by repeating the last frame
    aligned_video = video.copy()
    last_frame = video[-1]
    pad_num = target_frame_num - n
    for _ in range(pad_num):
        aligned_video.append(last_frame)
    return aligned_video


def pad_video_by_repetition(frames: List[Optional[Image.Image]]) -> List[Image.Image]:
    """
    Pad a list of frames by repeating the nearest non-None frame.

    For each None entry, we look for the nearest non-None frame before and after it,
    and fill the None with that frame. If there are multiple Nones in a row, they
    will all be filled with the same nearest non-None frame.

    Args:
        frames: List of frames where each element is either a PIL.Image or None.

    Returns:
        A list of the same length, where all elements are PIL.Image.
    """
    if not frames:
        return []

    n = len(frames)
    result: List[Optional[Image.Image]] = [None] * n

    # Fill from left to right
    last_valid = None
    for i in range(n):
        if frames[i] is not None:
            result[i] = frames[i]
            last_valid = frames[i]
        else:
            result[i] = last_valid

    # Fill from right to left for any remaining Nones at the start
    next_valid = None
    for i in range(n - 1, -1, -1):
        if result[i] is not None:
            next_valid = result[i]
        else:
            result[i] = next_valid

    # Now result has no None; just cast the type
    return [f for f in result]  # type: ignore


def _preprocess_frame(img: Image.Image, short_side: int = 288) -> np.ndarray:
    """
    Convert a PIL image to a grayscale uint8 numpy array and downscale
    so that the shorter side is at most `short_side`.
    """
    gray = np.array(img.convert("L"), dtype=np.uint8)
    h, w = gray.shape[:2]

    scale = 1.0
    if min(h, w) > short_side:
        scale = short_side / float(min(h, w))

    if scale != 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return gray


def _compute_farnebäck_flow(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """
    Compute Farnebäck optical flow from prev_gray to next_gray.
    Returns a flow array of shape (H, W, 2), dtype float32.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def _median_flow_magnitude(flow: np.ndarray) -> float:
    """
    Compute the median L2 magnitude of the optical flow.
    """
    mag = np.linalg.norm(flow, axis=2)
    return float(np.median(mag))


def _forward_backward_consistency(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> float:
    """
    Forward-backward consistency measure:
    C_t = median_x || F^{t->t+1}(x) + F^{t+1->t}(x + F^{t->t+1}(x)) ||_2
    """
    h, w, _ = flow_fwd.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    target_x = grid_x + flow_fwd[..., 0]
    target_y = grid_y + flow_fwd[..., 1]

    bwd_x = cv2.remap(
        flow_bwd[..., 0],
        target_x,
        target_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    bwd_y = cv2.remap(
        flow_bwd[..., 1],
        target_x,
        target_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    fb_sum_x = flow_fwd[..., 0] + bwd_x
    fb_sum_y = flow_fwd[..., 1] + bwd_y
    fb_mag = np.sqrt(fb_sum_x ** 2 + fb_sum_y ** 2)

    return float(np.median(fb_mag))


def _warp_image_with_flow(src_gray: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp src_gray using a flow that maps target pixels to source pixels.
    For backward flow F^{t+1->t}, the warped image approximates I_t warped to t+1.
    """
    h, w = src_gray.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    warped = cv2.remap(
        src_gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def _motion_compensated_residual(
    gray_t: np.ndarray, gray_t1: np.ndarray, flow_bwd: np.ndarray
) -> float:
    """
    Motion-compensated residual:
    R_t = 1/(255|Ω|) * mean_x | I_hat_{t->t+1}(x) - I_{t+1}(x) |
    where I_hat_{t->t+1} is gray_t warped using backward flow F^{t+1->t}.
    """
    warped = _warp_image_with_flow(gray_t, flow_bwd)
    diff = np.abs(warped.astype(np.float32) - gray_t1.astype(np.float32))
    r = float(np.mean(diff) / 255.0)
    return r


def _frame_diff(gray_t: np.ndarray, gray_t1: np.ndarray) -> float:
    """
    Simple frame difference:
    D_t = 1/(255|Ω|) * mean_x |I_t(x) - I_{t+1}(x)|
    Very sensitive to noise frames and temporal jumps.
    """
    diff = np.abs(gray_t.astype(np.float32) - gray_t1.astype(np.float32))
    return float(np.mean(diff) / 255.0)


def _robust_z_score_local(x: np.ndarray, window: int = 31) -> np.ndarray:
    """
    Local robust Z-score using a sliding window with median + MAD.

    For each index i, statistics are computed over x[i-window/2 : i+window/2],
    which makes the normalization adaptive to local motion statistics.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return x
    window = max(3, window | 1)  # force odd window size >= 3
    half = window // 2
    z = np.zeros_like(x, dtype=np.float32)

    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        seg = x[l:r]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-9
        z[i] = 0.6745 * (x[i] - med) / mad
    return z


def _clip_z(z: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """
    Clip Z-scores to avoid extreme outliers dominating the score.
    """
    return np.clip(z, -clip, clip)


def _gaussian_smooth_1d(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    1D Gaussian smoothing of a sequence.

    A moderate sigma helps reduce jitter but still keeps sharp peaks,
    which are often good indicators of discontinuities.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 3 or sigma <= 0:
        return x
    radius = max(1, int(2 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= np.sum(kernel)

    pad = radius
    padded = np.pad(x, pad_width=pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def _hysteresis_cut_indices(
    scores: np.ndarray, score_hi: float, score_lo: float
) -> List[int]:
    """
    1D hysteresis thresholding.

    score_hi / score_lo are absolute thresholds (not std-multipliers here).
    Returns boundary indices t (0..n-2) that are selected as cuts.
    """
    n = len(scores)
    if n == 0:
        return []

    strong = scores >= score_hi
    weak = (scores >= score_lo) & ~strong

    cuts = set(np.where(strong)[0].tolist())

    # Promote weak neighbors of strong responses
    for i in range(n):
        if weak[i]:
            if (i - 1 >= 0 and strong[i - 1]) or (i + 1 < n and strong[i + 1]):
                cuts.add(i)

    return sorted(cuts)


def _merge_short_segments(
    segments: List[Tuple[int, int]],
    cut_indices: List[int],
    scores_smooth: np.ndarray,
    hard_cut_mask: np.ndarray,
    min_len: int,
    soft_keep_thr: float,
) -> List[Tuple[int, int]]:
    """
    Merge short segments across weak boundaries only.

    - A segment shorter than `min_len` is considered "short".
    - A boundary can be merged across only if:
        - it is NOT a hard cut, and
        - its score is below `soft_keep_thr`.

    Hard cuts (e.g., noise vs normal, strong temporal jumps) are never merged.
    """
    if not segments or not cut_indices:
        return segments

    merged: List[Tuple[int, int]] = [segments[0]]

    for i in range(1, len(segments)):
        prev_start, prev_end = merged[-1]
        cur_start, cur_end = segments[i]
        prev_len = prev_end - prev_start + 1

        # Boundary index between prev segment and current segment
        b_idx = cut_indices[i - 1]
        b_score = float(scores_smooth[b_idx])
        is_hard = bool(hard_cut_mask[b_idx])

        if prev_len < min_len and (not is_hard) and b_score < soft_keep_thr:
            # Previous segment is short and boundary is weak: merge
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append((cur_start, cur_end))

    return merged


def optical_flow_segmentation(
    frames: List[Image.Image],
    short_side: int = 288,
    # Hysteresis sensitivity (in units of std)
    score_hi: float = 3.0,
    score_lo: float = 2.0,
    # Temporal smoothing
    gaussian_sigma: float = 1.2,
    # Hard-cut thresholds
    hard_z_thr: float = 3.2,
    hard_R_thr: float = 0.10,
    hard_D_thr: float = 0.10,
    # Segment post-processing
    min_segment_len: int = 8,
    # If None, soft_keep_thr defaults to hi_thr (mu + score_hi * sigma).
    # Otherwise: soft_keep_thr = mu + soft_keep_factor * sigma.
    soft_keep_factor: float = None,
    # Feature weights in the discontinuity score
    w_M: float = 0.15,
    w_C: float = 0.30,
    w_R: float = 0.30,
    w_dM: float = 0.10,
    w_D: float = 0.15,
) -> List[Tuple[int, int]]:
    """
    Optical flow-based temporal segmentation.

    The function detects discontinuities in a sequence of frames by combining
    optical-flow-based features and frame differences, then applies hysteresis
    thresholding and post-processing to obtain contiguous segments.

    Args:
        frames:
            List of PIL.Image frames, indexed 0..n-1.
        short_side:
            Frames are downscaled so that min(H, W) <= short_side.
        score_hi, score_lo:
            Multipliers for std in the adaptive thresholding:
                hi_thr = mean + score_hi * std
                lo_thr = mean + score_lo * std
            Lower values => more sensitive (more cuts).
        gaussian_sigma:
            Sigma used in 1D Gaussian smoothing of the score sequence.
            Smaller => more sensitive to short spikes, larger => smoother/more conservative.
        hard_z_thr:
            Local Z-score threshold for C/R/D above which a boundary is forced
            to be a "hard cut" (never merged away).
            Lower => more sensitive.
        hard_R_thr, hard_D_thr:
            Raw thresholds on motion-compensated residual R_t and frame difference D_t
            to promote boundaries to hard cuts.
            Lower => more sensitive.
        min_segment_len:
            Minimum length of a segment before it is considered for merging.
            Smaller => more short segments survive, larger => more merging (more conservative).
        soft_keep_factor:
            Controls how strong a boundary must be to be kept when merging:
                if None:
                    soft_keep_thr = hi_thr
                else:
                    soft_keep_thr = mean + soft_keep_factor * std
            Lower factor => easier to keep boundaries (more sensitive).
        w_M, w_C, w_R, w_dM, w_D:
            Weights for the feature contributions in the discontinuity score.

    Returns:
        A list of (start_idx, end_idx) tuples representing contiguous segments.
    """
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    # 1) Preprocess frames: grayscale + downscale
    gray_frames = [_preprocess_frame(img, short_side) for img in frames]

    # 2) Compute optical flows for each adjacent pair
    flows_fwd = []
    flows_bwd = []
    for t in range(n - 1):
        g0 = gray_frames[t]
        g1 = gray_frames[t + 1]
        flow_f = _compute_farnebäck_flow(g0, g1)
        flow_b = _compute_farnebäck_flow(g1, g0)
        flows_fwd.append(flow_f)
        flows_bwd.append(flow_b)

    # 3) Compute boundary features: M, C, R, dM, D
    M = np.zeros(n - 1, dtype=np.float32)
    C = np.zeros(n - 1, dtype=np.float32)
    R = np.zeros(n - 1, dtype=np.float32)
    dM = np.zeros(n - 1, dtype=np.float32)
    D = np.zeros(n - 1, dtype=np.float32)

    for t in range(n - 1):
        flow_f = flows_fwd[t]
        flow_b = flows_bwd[t]
        g0 = gray_frames[t]
        g1 = gray_frames[t + 1]

        M[t] = _median_flow_magnitude(flow_f)
        C[t] = _forward_backward_consistency(flow_f, flow_b)
        R[t] = _motion_compensated_residual(g0, g1, flow_b)
        D[t] = _frame_diff(g0, g1)

        if t == 0:
            dM[t] = 0.0
        else:
            dM[t] = abs(M[t] - M[t - 1])

    # 4) Local robust Z-score + clipping
    z_M = _robust_z_score_local(M)
    z_C = _robust_z_score_local(C)
    z_R = _robust_z_score_local(R)
    z_dM = _robust_z_score_local(dM)
    z_D = _robust_z_score_local(D)

    z_M = _clip_z(z_M, 4.0)
    z_C = _clip_z(z_C, 4.0)
    z_R = _clip_z(z_R, 4.0)
    z_dM = _clip_z(z_dM, 4.0)
    z_D = _clip_z(z_D, 4.0)

    # 5) Discontinuity score: emphasize C/R/D, weaken pure speed terms
    scores = (
        w_M * np.abs(z_M)
        + w_C * np.maximum(z_C, 0.0)
        + w_R * np.maximum(z_R, 0.0)
        + w_dM * np.maximum(z_dM, 0.0)
        + w_D * np.maximum(z_D, 0.0)
    )

    # 6) Temporal smoothing
    scores_smooth = _gaussian_smooth_1d(scores, sigma=gaussian_sigma)

    # 7) Adaptive thresholds based on mean/std
    mu = float(np.mean(scores_smooth))
    sigma_val = float(np.std(scores_smooth) + 1e-9)
    hi_thr = mu + score_hi * sigma_val
    lo_thr = mu + score_lo * sigma_val

    # 8) Hard-cut mask: boundaries that must be preserved
    hard_cut_mask = np.zeros(n - 1, dtype=bool)

    zC_pos = np.maximum(z_C, 0.0)
    zR_pos = np.maximum(z_R, 0.0)
    zD_pos = np.maximum(z_D, 0.0)

    hard_cut_mask |= (zC_pos >= hard_z_thr)
    hard_cut_mask |= (zR_pos >= hard_z_thr)
    hard_cut_mask |= (zD_pos >= hard_z_thr)

    # Additional safety using raw R and D values
    hard_cut_mask |= (R >= hard_R_thr)
    hard_cut_mask |= (D >= hard_D_thr)

    # 9) Hysteresis-based cuts, then union with hard cuts
    cut_indices_hyst = _hysteresis_cut_indices(scores_smooth, hi_thr, lo_thr)
    hard_cut_indices = np.where(hard_cut_mask)[0].tolist()
    all_cut_indices = sorted(set(cut_indices_hyst) | set(hard_cut_indices))

    # 10) Build initial segments from cut indices
    segments: List[Tuple[int, int]] = []
    start = 0
    for ci in all_cut_indices:
        end = ci
        segments.append((start, end))
        start = ci + 1
    segments.append((start, n - 1))

    # 11) Merge short segments across weak boundaries only
    if soft_keep_factor is None:
        soft_keep_thr = hi_thr
    else:
        soft_keep_thr = mu + soft_keep_factor * sigma_val

    segments = _merge_short_segments(
        segments,
        all_cut_indices,
        scores_smooth,
        hard_cut_mask,
        min_len=min_segment_len,
        soft_keep_thr=soft_keep_thr,
    )

    return segments


# Helper to interpolate between two images
def _lerp(img_a: Image.Image, img_b: Image.Image, alpha: float) -> Image.Image:
    """
    Linear interpolation between two images:
        out = (1 - alpha) * img_a + alpha * img_b
    alpha in [0, 1]
    """
    if img_a.size != img_b.size or img_a.mode != img_b.mode:
        raise ValueError("Images to interpolate must have the same size and mode.")

    arr_a = np.asarray(img_a, dtype=np.float32)
    arr_b = np.asarray(img_b, dtype=np.float32)
    arr = (1.0 - alpha) * arr_a + alpha * arr_b
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img_a.mode)


def pad_video_by_interpolation(frames: List[Optional[Image.Image]]) -> List[Image.Image]:
    """
    Fill None entries in a frame list by linear interpolation between nearest
    non-None frames before and after.

    Rules:
        - For a run of N None frames between frame A (left) and frame B (right),
          we generate N interpolated frames that linearly transition from A to B.
        - Leading None frames (before the first non-None) are filled with the first
          non-None frame.
        - Trailing None frames (after the last non-None) are filled with the last
          non-None frame.

    Args:
        frames: List of frames where each element is either a PIL.Image or None.

    Returns:
        A list of the same length, where all elements are PIL.Image.
    """
    if not frames:
        return []

    n = len(frames)

    # Indices of all non-None frames
    valid_indices = [i for i, f in enumerate(frames) if f is not None]
    if not valid_indices:
        raise ValueError("All frames are None; cannot interpolate.")

    # Make a working copy and normalize leading/trailing Nones
    result: List[Optional[Image.Image]] = list(frames)

    # Fill leading Nones with the first valid frame
    first_valid = valid_indices[0]
    first_img = result[first_valid]
    for i in range(0, first_valid):
        result[i] = first_img

    # Fill trailing Nones with the last valid frame
    last_valid = valid_indices[-1]
    last_img = result[last_valid]
    for i in range(last_valid + 1, n):
        result[i] = last_img

    # Now handle interior runs of Nones between valid frames
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue

        # Found a run of Nones starting at i
        start = i - 1  # index of left non-None (we ensured leading Nones are filled)
        # find right non-None
        j = i
        while j < n and result[j] is None:
            j += 1
        end = j  # index of right non-None (we ensured trailing Nones are filled)

        left_img = result[start]
        right_img = result[end]
        gap = end - start  # total steps from left to right

        # Fill positions (start+1 ... end-1)
        for t, idx in enumerate(range(start + 1, end), start=1):
            alpha = t / gap  # t from 1..(gap-1)
            result[idx] = _lerp(left_img, right_img, alpha)

        i = end + 1

    # Now result has no None; just cast the type
    return [f for f in result]  # type: ignore
