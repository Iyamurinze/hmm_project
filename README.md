# Human Activity Recognition with a Hidden Markov Model

Formative 2 – Hidden Markov Models. A from-scratch NumPy HMM that infers four hidden activity
states (**still, standing, walking, jumping**) from smartphone accelerometer + gyroscope signals
recorded with the Sensor Logger app.

## Repository layout

```
.
├── data/
│   ├── manifest.csv            one row per file: source recording, device, sampling rate, duration
│   ├── train/                  52 labelled clips (13 per activity, 6–11 s, 100 Hz)  e.g. walking_07.csv
│   └── test/                   unseen data, never used for training
│       ├── <activity>_long_<n>.csv   8 long recordings (40–86 s) from separate sessions
│       ├── test_session_1.csv        still → standing → walking → jumping   (211 s)
│       └── test_session_2.csv        jumping → walking → standing → still   (177 s)
├── hmm_activity_recognition.ipynb   the full pipeline (executed, outputs included)
├── hmm_activity_recognition.py      same notebook as a plain script (jupytext "percent" format)
├── results/
│   ├── figures/                all 14 figures used in the report
│   ├── metrics_table.csv       sensitivity / specificity / accuracy per state on unseen data
│   ├── confusion_matrix.csv, transition_matrix.csv, emission_means.csv, emission_stds.csv, ...
│   └── summary.txt             headline numbers
└── report.docx                 the written report
```

Every CSV has the columns `time_s, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, activity`
(accelerometer in m/s² with gravity removed, gyroscope in rad/s).

## Pipeline (see the notebook for details and justification)

1. **Pre-processing** – trim 1 s of press-start / press-stop motion from each clip; resample every
   file onto a common 50 Hz grid (handles phones with different sampling rates).
2. **Windowing** – 2 s windows (100 samples) with 1 s hop. Window length chosen from the sampling
   rate so that the FFT resolves 0.5 Hz bins and each window spans ≥ 3 gait / jump cycles.
3. **Features** – 7 time-domain (mean, std, SMA, range, gyro mean/std, inter-axis correlation)
   and 3 FFT-derived frequency-domain features (dominant frequency, spectral energy, spectral
   entropy), log-transformed where appropriate and z-scored with training statistics.
4. **HMM** – 4 hidden states, diagonal-Gaussian emissions. Forward, backward, Baum–Welch
   (log-likelihood convergence check, ε = 1e-4) and Viterbi implemented in NumPy.
5. **Evaluation** – Viterbi decoding of two unseen multi-activity sessions; confusion matrix,
   sensitivity, specificity and accuracy per state; comparison with a per-window baseline;
   robustness check on an unseen activity order; sensitivity analysis of the emission regulariser.

## Reproduce

```bash
pip install numpy pandas scipy matplotlib seaborn jupyter
jupyter nbconvert --to notebook --execute --inplace hmm_activity_recognition.ipynb
```

Runs in under a minute on a laptop.
