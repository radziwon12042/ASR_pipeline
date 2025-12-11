"""
ASR-based EEG preprocessing pipeline for BCI applications.

This implementation builds on and interacts with several open-source tools:

- ASRpy: Python implementation of Artifact Subspace Reconstruction (ASR),
  used as the core ASR algorithm.
  https://github.com/DiGyt/asrpy

- clean_rawdata (EEGLAB plugin): Provides reference implementations and design
  ideas for EEG cleaning and ASR-based preprocessing.
  https://github.com/sccn/clean_rawdata/tree/master

- PyPREP: Python implementation of the PREP pipeline, informing some of the bad-channel
  detection strategies.
  https://github.com/sappelhoff/pyprep

- MNE-Python: EEG/MEG analysis library used for data structures, I/O,
  preprocessing utilities, and visualization.
  https://mne.tools/

For full methodological details and formal citations of these tools, see the
associated thesis document.

Note: This module is a research prototype developed for a specific BCI framework.
It may require further adaptation in order to function as a polished standalone package in other use cases,
"""

import numpy as np
import mne
from mne.utils import logger
from scipy.stats import zscore, median_abs_deviation
from asrpy import clean_windows, asr_calibrate, asr_process


def detect_flatline_channels(epochs, flatline_duration=10, verbose=True):
    flat_channels = []
    data = epochs.get_data(picks='eeg')
    data = np.concatenate(data, axis=1)
    ch_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])]

    max_flatline_samples = int(flatline_duration * epochs.info['sfreq'])

    for ch_idx, ch_name in enumerate(ch_names):
        ch_data = data[ch_idx, :]
        diffs = np.abs(np.diff(ch_data))
        zero_diffs = diffs < np.finfo(float).eps * 10

        max_flat_run = 0
        current_run = 0
        for is_flat in zero_diffs:
            if is_flat:
                current_run += 1
                max_flat_run = max(max_flat_run, current_run)
            else:
                current_run = 0

        if max_flat_run > max_flatline_samples:
            flat_channels.append(ch_name)
            if verbose:
                logger.info(f"  Flatline detected in {ch_name}: {max_flat_run / epochs.info['sfreq']:.2f}s flat")

    return flat_channels


def detect_noisy_channels_correlation(
    epochs,
    correlation_secs=1.0,
    correlation_threshold=0.4,
    frac_bad=0.01,
    sfreq=None,
    verbose=True,
):
    data = epochs.get_data(picks='eeg')
    sfreq = epochs.info['sfreq'] if sfreq is None else sfreq
    ch_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]

    n_epochs, n_chans, n_times = data.shape
    win_size = int(correlation_secs * sfreq)
    n_win_per_epoch = max(1, n_times // win_size)
    total_windows = n_epochs * n_win_per_epoch

    max_corrs = np.ones((total_windows, n_chans))
    dropouts = np.zeros((total_windows, n_chans), dtype=bool)

    w_idx = 0
    for ep in range(n_epochs):
        for w in range(n_win_per_epoch):
            start = w * win_size
            stop = start + win_size
            seg = data[ep, :, start:stop]
            seg = seg - np.mean(seg, axis=1, keepdims=True)
            amp = median_abs_deviation(seg, axis=1)
            dropouts[w_idx, :] = amp == 0

            usable_mask = amp > 0
            if np.sum(usable_mask) < 2:
                max_corrs[w_idx, :] = 0
                w_idx += 1
                continue

            seg_u = seg[usable_mask]
            cmat = np.corrcoef(seg_u)
            abs_corr = np.abs(cmat - np.diag(np.diag(cmat)))
            maxcorr_u = np.quantile(abs_corr, 0.98, axis=0)
            tmp = np.zeros(n_chans)
            tmp[usable_mask] = maxcorr_u
            max_corrs[w_idx, :] = tmp
            w_idx += 1

    frac_bad_windows = np.mean(max_corrs < correlation_threshold, axis=0)
    frac_dropout_windows = np.mean(dropouts, axis=0)

    bad_by_corr = [ch_names[i] for i in np.where(frac_bad_windows > frac_bad)[0]]
    bad_by_dropout = [ch_names[i] for i in np.where(frac_dropout_windows > frac_bad)[0]]

    if verbose:
        logger.info(f"[find_bad_by_correlation] threshold={correlation_threshold}, frac_bad={frac_bad}")
        logger.info(f"Bad-by-correlation: {bad_by_corr}")
        logger.info(f"Bad-by-dropout: {bad_by_dropout}")

    return bad_by_corr, bad_by_dropout


def detect_noisy_channels_variance(epochs, threshold=3, verbose=True):
    noisy_channels = []
    eeg_picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])

    if len(eeg_picks) == 0:
        return []

    data = epochs.get_data(picks=eeg_picks)
    ch_names = [epochs.ch_names[i] for i in eeg_picks]
    data = np.concatenate(data, axis=1)

    variances = np.var(data, axis=1)
    z_scores = np.abs(zscore(variances))
    extreme_idx = z_scores > threshold

    for ch_idx in np.where(extreme_idx)[0]:
        noisy_channels.append(ch_names[ch_idx])
        if verbose:
            logger.info(f"  Extreme variance in {ch_names[ch_idx]}: z-score = {z_scores[ch_idx]:.2f}")

    return noisy_channels


def detect_bad_channels(
    epochs,
    flatline_duration=10,
    chn_var=3,
    chn_cor=0.42,
    bad_channel_method='both',
    verbose=True
):
    if verbose:
        logger.info("Detecting bad channels...")

    bad_channels = []

    if flatline_duration is not None and flatline_duration > 0:
        flat_channels = detect_flatline_channels(epochs, verbose=verbose)
        bad_channels.extend(flat_channels)

    if bad_channel_method in ['correlation', 'both']:
        bad_by_corr, bad_by_dropout = detect_noisy_channels_correlation(epochs, correlation_threshold=chn_cor, verbose=verbose)
        bad_channels.extend(bad_by_corr)
        bad_channels.extend(bad_by_dropout)

    if bad_channel_method in ['variance', 'both']:
        bad_by_var = detect_noisy_channels_variance(epochs, threshold=chn_var, verbose=verbose)
        bad_channels.extend(bad_by_var)

    bad_channels = list(set(bad_channels))

    if verbose:
        if len(bad_channels) > 0:
            logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
        else:
            logger.info("No bad channels detected")

    return bad_channels


class ArtefactFilterASR:
    def __init__(
        self,
        cutoff=30,
        blocksize=100,
        win_len=0.5,
        win_overlap=0.66,
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
        max_bad_chans=0.2,
        zthresholds=[-3.5, 5],
        step_size=0.33,
        max_dims=0.66,
        interpolate_bads=True,
        apply_avg_reference=True,
        verbose=True,
        apply_frequency_filter=None,
        filter_low=0.5,
        filter_high=45,
    ):
        self._info = None
        self.sfreq = None
        self.cutoff = cutoff
        self.blocksize = blocksize
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.max_bad_chans = max_bad_chans
        self.zthresholds = zthresholds
        self.verbose = verbose
        self.step_size = step_size
        self.max_dims = max_dims

        self.detected_bad_channels = None
        self.interpolate_bads = interpolate_bads
        self.apply_avg_reference = apply_avg_reference
        self.apply_frequency_filter = apply_frequency_filter
        self.filter_low = filter_low
        self.filter_high = filter_high

        self.M = None
        self.T = None
        self._state = None

    def offline_filter(self, epochs, return_indexes=False, plot_clean=False):
        if self.verbose:
            logger.info("Running offline_filter...")

        if self.apply_frequency_filter:
            epochs.filter(self.filter_low, self.filter_high)

        detected_bads = detect_bad_channels(epochs, verbose=self.verbose)
        self.detected_bad_channels = detected_bads
        epochs.info['bads'].extend(detected_bads)

        eeg_picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
        n_eeg_channels = len(eeg_picks)
        n_bad_channels = len(epochs.info['bads'])
        bad_channel_fraction = n_bad_channels / n_eeg_channels if n_eeg_channels > 0 else 0

        if self.verbose:
            logger.info("Bad channel statistics:")
            logger.info(f"  Total EEG channels: {n_eeg_channels}")
            logger.info(f"  Bad channels: {n_bad_channels} ({bad_channel_fraction * 100:.1f}%)")

        if bad_channel_fraction > 0.3:
            logger.warning(f"Warning: {bad_channel_fraction * 100:.1f}% of channels are bad!")
            logger.warning("This may cause poor interpolation quality and ASR calibration issues.")

            if bad_channel_fraction > 0.5:
                raise ValueError(
                    f"{n_bad_channels}/{n_eeg_channels} ({bad_channel_fraction * 100:.1f}%) channels are bad. "
                    f"Cannot reliably interpolate or calibrate ASR with >50% bad channels."
                )

        X = epochs.get_data()
        n_epochs, n_chans, n_times = X.shape
        self._info = epochs.info
        self.sfreq = epochs.info['sfreq']
        X_cont = X.transpose(1, 0, 2).reshape(n_chans, -1)

        clean, mask = clean_windows(
            X_cont,
            sfreq=self.sfreq,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_bad_chans=self.max_bad_chans,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction,
            zthresholds=self.zthresholds,
        )

        if self.verbose:
            mask = mask.flatten()
            total_samples = len(mask)
            clean_samples = np.sum(mask)
            clean_percentage = (clean_samples / total_samples) * 100
            logger.info("Calibration data statistics:")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Clean samples: {clean_samples}")
            logger.info(f"  Clean percentage: {clean_percentage:.2f}%")

        trim = clean.shape[1] % self.blocksize
        if trim > 0:
            clean = clean[:, :-trim]

        if plot_clean:
            n_times_clean = int(4 * self.sfreq)
            n_epochs_new = clean.shape[1] // n_times_clean
            total_samples_used = n_epochs_new * n_times_clean
            clean_use = clean[:, :total_samples_used]
            clean2 = clean_use.reshape(n_chans, n_epochs_new, n_times_clean).transpose(1, 0, 2)
            clean_epochs = mne.EpochsArray(clean2, self._info)
            clean_epochs.plot(title="CLEAN Epochs", scalings={'eeg': 80e-6}, n_channels=10, n_epochs=10, block=True)

        self.M, self.T = asr_calibrate(
            clean,
            self.sfreq,
            cutoff=self.cutoff,
            blocksize=self.blocksize,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
        )

        if self.verbose:
            logger.info("ASR calibration complete. Applying process...")

        filtered_cont = asr_process(
            X_cont,
            self.sfreq,
            self.M,
            self.T,
            windowlen=self.win_len,
            lookahead=self.win_len / 2,
            stepsize=int(self.step_size * self.sfreq),
            maxdims=self.max_dims,
            ab=None,
            mem_splits=3,
        )

        filtered_data = filtered_cont.reshape(n_chans, n_epochs, n_times).transpose(1, 0, 2)
        new_epochs = mne.EpochsArray(
            filtered_data,
            self._info,
            events=epochs.events,
            event_id=epochs.event_id
        )

        if self.interpolate_bads and len(new_epochs.info['bads']) > 0:
            if self.verbose:
                logger.info(f"Interpolating {len(new_epochs.info['bads'])} bad channels...")
            new_epochs.interpolate_bads(reset_bads=False)

        if self.apply_avg_reference:
            if self.verbose:
                logger.info("Applying average reference after ASR filtering...")
            new_epochs = new_epochs.set_eeg_reference(ref_channels='average', projection=False, verbose=self.verbose)

        good_indices = np.arange(n_epochs)

        if self.verbose:
            logger.info("Offline filtering complete.")

        if return_indexes:
            return new_epochs, good_indices
        return new_epochs

    def online_filter(self, data, lookahead=None, mem_splits=1):
        if self.verbose:
            logger.info("Online filtering...")

        if self.M is None or self.T is None:
            raise RuntimeError("Run offline_filter or set calibration params first.")

        if self.apply_frequency_filter:
            data.filter(self.filter_low, self.filter_high)

        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)

        epochs = mne.EpochsArray(data, self._info)
        epochs.info['bads'] = self.detected_bad_channels.copy()

        X = epochs.get_data()
        n_epochs, n_chans, n_times = X.shape
        lookahead = lookahead if lookahead is not None else self.win_len / 2

        filtered_epochs = np.zeros_like(X)
        state = self._state

        for i in range(n_epochs):
            epoch = X[i]

            if state:
                R = state.get('R')
                Zi = state.get('Zi')
                cov = state.get('cov')
                carry = state.get('carry')
                if carry.shape[0] > epoch.shape[0]:
                    carry = carry[-epoch.shape[0]:, :]
            else:
                R = Zi = cov = carry = None

            filtered, state = asr_process(
                epoch,
                self.sfreq,
                self.M,
                self.T,
                windowlen=self.win_len,
                lookahead=lookahead,
                stepsize=int(self.step_size * self.sfreq),
                maxdims=self.max_dims,
                ab=None,
                R=R,
                Zi=Zi,
                cov=cov,
                carry=carry,
                return_states=True,
                mem_splits=mem_splits,
            )

            filtered_epochs[i] = filtered
            self._state = state

        processed_epochs = mne.EpochsArray(filtered_epochs, self._info)

        if self.interpolate_bads and len(processed_epochs.info['bads']) > 0:
            processed_epochs.interpolate_bads(reset_bads=False, verbose=False)

        if self.apply_avg_reference:
            processed_epochs = processed_epochs.set_eeg_reference(
                ref_channels='average',
                projection=False,
                verbose=self.verbose
            )

        return processed_epochs.get_data()

    def mimic_online_filter(self, epochs):
        if self.M is None or self.T is None:
            raise RuntimeError(
                "ASR model has not been fitted. Call offline_filter_asr() first to calibrate the ASR model."
            )

        if self.verbose:
            logger.info(f"Simulating online ASR filtering on {len(epochs)} epochs...")

        epochs = epochs.copy()
        epochs.load_data()

        for i in range(len(epochs)):
            data = epochs[i].get_data()
            epochs._data[i, :, :] = self.online_filter(data)[0, :, :]

        return epochs
