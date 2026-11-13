# DNN Assignment 2 README

## Environment

Both notebooks were run in Google Colab with a T4 GPU runtime. Exact Python and package versions are recorded in `requirements.txt` at the assignment root. The work uses PyTorch, NumPy, pandas, Matplotlib, and scikit-learn; no pretrained weights are used. Internet access is required on the first run to download UCI HAR and Tiny Shakespeare. Colab already provides GPU-enabled PyTorch; avoid reinstalling a pinned CPU-only build over it.

## How to run

1. Upload and unzip the assignment folder in Colab, select **Runtime > Change runtime type > T4 GPU**, and use the submission folder structure.
2. Run `problem1/problem1.ipynb` first. Set the working directory to `problem1` (from the assignment root, run `%cd problem1`), then select **Runtime > Restart and run all**. This runs scratch-cell checks, HAR and adding experiments, the gradient plot, hidden-size sweep, and individual ablation.
3. Return to the assignment root, set the working directory to `problem2` (run `%cd problem2`), and run `problem2/problem2.ipynb` with **Runtime > Restart and run all**. This runs attention and encoder checks, Task A, the character model, ablations, and optional bonuses.

Both notebooks use relative paths and append to the root `results.csv`. Run Problem 1 before Problem 2 so the Task A comparison can read the Problem 1 HAR results. For a clean rerun, back up or clear the old results file first to avoid duplicate rows. Figures and summaries are saved in their problem folders; Problem 2 also writes `generated_samples.txt`.

## Student variant and settings

Problem 1 uses hidden size 64, seeds 1, 42, and 123, Adam at 1e-3, batch size 128, 20 HAR epochs, and 30 adding-task epochs at sequence lengths 50, 100, 200, and 400. The odd-parity individual ablation compares LSTM forget-gate bias 0 and 1 at T=200.

Problem 2 Task A uses 8-step HAR patches, d=64, four heads, two encoder blocks, and FFN width 256. Task B uses context 128, d=128, four heads, four blocks, FFN width 512, batch size 64, and 5,000 iterations. The heads and no-mask ablations use 1,500 iterations. For CPU-only Task B, use d=64 and two layers and report the changed configuration.

## Runtime and limitations

The full Problem 2 notebook reports about **18.2 minutes** on the T4; the main language-model training itself is 182 seconds. The saved outputs do not record the full Problem 1 wall-clock duration, so the combined runtime is not available here; add the elapsed time from its Colab run before submission. Context-scaling measurements are noisy. The T=200 forget-bias ablation did not solve the adding task, and cache timings at contexts 256 and 512 use untrained models for timing only.

