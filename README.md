# IPL Score Prediction Using Deep Learning

This project estimates an IPL innings' final score from the match situation at a
delivery. It includes a reproducible TensorFlow/Keras training script, a
Streamlit interface, and evaluation artifacts generated from the supplied CSV.
Predictions are estimates, not guarantees.

## Dataset

The project reads `dataset/ipl_data.csv`. The inspected file has **76,014 rows**
and **15 columns**, covering **617 matches** from **2008-04-18 through
2017-05-21**. It contains match/date and venue identifiers, batting and bowling
teams, batter/bowler names, current innings state, recent-form fields, and the
final innings `total`.

The final `total` is repeated for deliveries in a match. Inspection found no
missing values, exact duplicate rows, or matches with more than one `total`.
Training validates the required values and cricket over notation; it removed
**0 rows** from this dataset.

## Inputs and leakage prevention

The model predicts `total` using only information available at prediction time:

- Batting team, bowling team, and venue
- Completed balls, current runs, and wickets
- Runs and wickets in the last five overs

The dataset writes overs as cricket notation: `19.3` means 19 overs and 3 balls,
not 19.3 decimal overs. Training converts it to a completed-ball count before
scaling. Team-name aliases for Delhi Daredevils/Delhi Capitals, Kings XI
Punjab/Punjab Kings, Royal Challengers Bangalore/Bengaluru, and the two Rising
Pune spellings are normalized consistently.

`total`, match ID, date, batter, bowler, striker, and non-striker are not model
inputs. Names are excluded because they are high-cardinality and are not needed
for this prediction interface. The date is used only to create a
chronological evaluation split. Categorical inputs are one-hot encoded, and
numeric inputs are standardized. The fitted preprocessing object is saved with
the model.

No match is divided across train, validation, and test in either evaluation:

- The primary evaluation randomly assigns matches to 70% training, 10%
  validation, and 20% test using seed 42. The held-out sample therefore
  represents matches from across the available seasons.
- A second evaluation trains only on earlier matches (first 70%), validates on
  the next 10%, and tests on the final 20%. This checks how well the model
  transfers to later seasons.

## Model and evaluation

The regression network has Dense layers of 128, 64, and 32 ReLU units, with
dropout between hidden layers and one numerical output. It uses Adam, Huber
loss, a fixed seed, and early stopping based on a match-separated validation
set. A fitted one-hot encoder and standard scaler are stored alongside the
Keras model.

Results from running `train.py` on the supplied data:

| Evaluation | Model MAE | Model RMSE | Model R² | Training-mean baseline MAE | Baseline RMSE | Baseline R² |
|---|---:|---:|---:|---:|---:|---:|
| Random held-out matches | 16.05 | 21.92 | 0.376 | 21.14 | 27.76 | -0.001 |
| Chronologically later matches | 36.22 | 46.47 | -1.554 | 23.00 | 29.86 | -0.054 |

The random match test is better than the simple mean baseline on this dataset.
The chronological result is substantially worse than that baseline, showing
that these historical data do not support reliable future-season
generalization. The app should therefore be treated as a learning/demo project,
not a dependable live-match forecasting service.

The app also shows an approximate range based on the 90th percentile of
absolute residuals from the random held-out matches (**about ±34.62 runs** in
this run). This is an empirical reference range, not a calibrated guarantee;
the chronological evaluation shows that errors can be wider under distribution
shift.

Generated artifacts:

- `models/ipl_score_model.keras` — trained neural network
- `models/preprocessor.joblib` — preprocessing, dropdown values, and range data
- `outputs/evaluation_metrics.json` — splits, metrics, and training details
- `outputs/actual_vs_predicted.csv` — random held-out actual and predicted scores
- `outputs/actual_vs_predicted.png` — held-out scatter plot
- `outputs/training_loss.png` — training and validation loss

## Setup on Windows

Use Python **3.11**. Create a separate project environment so globally installed
packages and the existing Python 3.14 environment are left unchanged:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

In VS Code, select `.venv311\Scripts\python.exe` as the Python interpreter.

## Train the model

From the project directory:

```powershell
.\.venv311\Scripts\python.exe .\train.py
```

Training evaluates both match-level split strategies, saves the random-split
model and preprocessing artifacts, and writes the metrics and plots under
`outputs/`.

## Run the Streamlit app

Train the model first, then run:

```powershell
.\.venv311\Scripts\python.exe -m streamlit run .\app.py
```

Choose the teams and venue, enter the current score, wickets, recent-form
figures, and overs in cricket notation (for example `12.3`), then select
**Predict final score**. The app loads the saved model; it does not retrain on
startup.

## Limitations and future work

- The provided data ends in 2017. Rules, team names, venues, and scoring
  conditions have since changed.
- The chronological test was worse than a constant training-mean baseline.
  Better up-to-date seasons and additional temporal features are needed before
  relying on predictions for current matches.
- New or unseen team/venue categories are ignored by the encoder rather than
  learned from later data.
- The residual-based range comes from the random holdout and should not be
  interpreted as a guaranteed interval, especially given the chronological
  generalization result.
- Future improvements could include current seasons, explicit innings/season
  context, and evaluation by both held-out match and future season.
