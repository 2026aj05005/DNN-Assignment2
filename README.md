# DNN Assignment 2

## Environment

Both notebooks were executed in **Google Colab with a GPU T4**. The notebooks use Python, PyTorch, NumPy, pandas, Matplotlib, SciPy, and scikit-learn; see [`requirements.txt`](requirements.txt) for the recorded package versions. Colab already provides a CUDA-enabled PyTorch build, so do not reinstall the pinned `torch` version there if that would replace the GPU-enabled build. The notebooks download UCI HAR and (for Problem 2) Tiny Shakespeare as needed, so an internet connection is required on the first run.

A GPU is recommended but not mandatory: a free Colab T4 is sufficient, and Problem 1 can also run on a laptop CPU. For a CPU-only **Problem 2 language-model** run, change its configuration to `D_LM = 64` and `LAYERS_LM = 2`; this is a smaller configuration and will not reproduce the full T4 language-model results.

## How to run

1. Open each notebook in Colab (or Jupyter) and select **Run all**. Use a fresh runtime and the repository's directory structure. Run Problem 1 first so it creates the root `results.csv`; Problem 2 reads it for the comparison with the best Problem 1 HAR model. Both notebooks set their device automatically and download the required datasets.
2. Run `problem1/problem1.ipynb` for the scratch RNN/LSTM/GRU checks and the HAR and adding-task experiments. It writes results to `results.csv` and figures under `problem1/figures/`.
3. Run `problem2/problem2.ipynb` for attention/encoder checks, HAR positional-encoding experiments, and the character language model and ablations. It appends results to `results.csv` and saves figures under `problem2/figures/`.

For reproducibility, keep the configured seeds `(1, 42, 123)`, run every cell in order, and do not enable `QUICK_TEST` in Problem 2 for the full experiments. Results can vary slightly across hardware/software even with fixed seeds.

## Student variant

Student ID: **2026AJ05005**. The last digit is odd, so Problem 1 selects the **forget-bias** Task 5(c) ablation. Problem 1 uses hidden size 64 and seeds `(1, 42, 123)`. Problem 2 uses `d = 128`, 4 heads, and 4 layers for its language model on the T4; its HAR encoder uses `d = 64`, 4 heads, and 2 layers.

## Runtime and known limitations

The full end-to-end wall-clock time for both notebooks was **not recorded** in the committed notebook outputs; record the elapsed time from a complete fresh Colab T4 run here before submission. Available timing evidence: the Problem 2 HAR encoders report about **0.39–0.43 seconds per epoch**, and the main 5,000-iteration language-model training reports **182 seconds**. Total time depends on all experiments, downloads, and Colab availability.

The assignment's individualisation-variants table and instructor-provided attention example were not available; the notebooks document the chosen settings and use their own hand-checkable attention example. Dataset download steps require internet access. Keep the written report within the assignment's **six-page maximum**; this README is intended as the run guide, not the report.
