# word2vec-from-scratch-numpy

Pure NumPy implementation of Word2Vec using the Skip-Gram architecture with Negative Sampling.

## What is included

- text preprocessing and vocabulary building
- skip-gram pair generation
- negative sampling with the unigram^0.75 distribution
- manual forward pass, loss, gradients, and SGD updates
- nearest-neighbor inspection with cosine similarity
- training loss visualization

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The training script expects the `text8` dataset at:

```text
data/text8
```

If the file is missing, download `text8` and place it in the `data/` directory.

## Run

From the repository root, run:

```bash
python -m scripts.run_training
```

To skip plotting:

```bash
python -m scripts.run_training --no-plot
```

## Default configuration

The repository uses the following default values from `src/config.py`:

```python
EMBEDDING_DIM = 50
WINDOW_SIZE = 2
NUM_NEGATIVE_SAMPLES = 5
LEARNING_RATE = 0.025
EPOCHS = 3

MIN_COUNT = 5
MAX_VOCAB_SIZE = None
MAX_TOKENS = 100000

SEED = 42
```

These are the default values used when no command-line overrides are provided.

## Example

```bash
python -m scripts.run_training --max-tokens 100000 --epochs 3 --embedding-dim 50 --window-size 2 --num-negative-samples 5 --no-plot
```

## Experiment configuration

An additional training experiment was launched with the following configuration:

```python
EMBEDDING_DIM = 100
WINDOW_SIZE = 3
NUM_NEGATIVE_SAMPLES = 10
LEARNING_RATE = 0.025
EPOCHS = 10

MIN_COUNT = 5
MAX_VOCAB_SIZE = None
MAX_TOKENS = 200000

SEED = 42
```

## Results

### Experiment 1: default demo configuration

Configuration:

```python
EMBEDDING_DIM = 50
WINDOW_SIZE = 2
NUM_NEGATIVE_SAMPLES = 5
LEARNING_RATE = 0.025
EPOCHS = 3

MIN_COUNT = 5
MAX_VOCAB_SIZE = None
MAX_TOKENS = 100000

SEED = 42
```

Run statistics:

- Tokens after filtering: `84870`
- Vocabulary size: `2629`
- Training pairs: `339474`
- Final loss: `2.4043`

Loss history:

```python
[2.9822805202448284, 2.4758902279839003, 2.4043116191036025]
```

Nearest neighbors:

```text
king   -> colony, policy, round, governor, dollars
queen  -> band, consisting, coastal, sterreich, heart
man    -> told, mother, clear, job, worker
woman  -> musical, score, big, brakeman, ten
city   -> parliament, treaty, federal, consists, council
love   -> minerals, accounts, developments, philadelphia, kentucky
```

### Experiment 2: larger training configuration

Configuration:

```python
EMBEDDING_DIM = 100
WINDOW_SIZE = 3
NUM_NEGATIVE_SAMPLES = 10
LEARNING_RATE = 0.025
EPOCHS = 10

MIN_COUNT = 5
MAX_VOCAB_SIZE = None
MAX_TOKENS = 200000

SEED = 42
```

Run statistics:

- Tokens after filtering: `176569`
- Vocabulary size: `4533`
- Training pairs: `1059402`
- Final loss: `2.6953`

Loss history:

```python
[3.422933107234636, 3.0255078024474575, 2.9634414975286854, 2.910398792451537, 2.8631326549965586, 2.82093195854479, 2.7833082746129234, 2.7502138986282225, 2.721030775023752, 2.6953454396206955]
```

Nearest neighbors:

```text
king   -> priam, daniel, concerto, secretary, frederick
queen  -> reserve, facto, navy, rush, seat
man    -> dollar, writes, woman, boyle, mourned
woman  -> screenwriter, bertram, knowing, scudder, dollar
city   -> downtown, port, melbourne, town, arctic
love   -> fell, instrument, priam, friend, daughter
```

## Project structure

- `src/preprocessing.py` - token loading and vocabulary creation
- `src/dataset.py` - skip-gram pairs and negative-sampling distribution
- `src/model.py` - Skip-Gram with Negative Sampling model
- `src/train.py` - training loop
- `src/evaluate.py` - nearest neighbors and loss plotting
- `scripts/run_training.py` - training entrypoint
