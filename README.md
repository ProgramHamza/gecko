# Developmental LM

A small language model that lives in a tiny world, gets curious, asks a bigger
model questions, sleeps to rewrite its own weights, can grow new capacity, and
keeps track of what it knew and when.

The child is [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M):
it speaks simple English but knows almost nothing about the world. The world is
invented (creatures like a *zibo* that hums when it rains), so everything it
learns, it learns here.

## How a day works

1. **Perceive.** The meadow produces ~10 events ("It was rainy. The zibo hummed.").
   Some follow hidden rules, some are random noise. The child scores how
   surprising each one is (its prediction error).
2. **Wonder.** Curiosity picks what to ask about: things never seen before first,
   then the things it is currently improving at fastest (learning progress).
3. **Ask.** The child guesses an answer itself, then the teacher (a bigger model)
   explains. The teacher knows the true rules; if it says something that does not
   state the rule, a scripted answer is used instead.
4. **Sleep.** New memories, some old ones (replay) and some stories its day-0 self
   wrote (so it keeps its language) are trained into the weights. If the new
   memories are still badly predicted, it can switch on a new growth module.
5. **Get tested.** Every fact is probed twice (a question and a fill-in sentence),
   plus developmental milestones: object permanence, false belief, conservation
   and causal induction. All scored by likelihood, so no instruction following
   is needed.
6. **Remember learning it.** Facts that became known today get an autobiographical
   memory: "On day 6, I learned that when it is rainy, the zibo hums."
7. **Self-knowledge.** It is asked "N days ago, did I know this?" and
   "When did I learn this?", and checked against the real test history.

Everything is logged per day, and the weights are snapshotted every night so you
can talk to any past version.

## Quick start (Ubuntu + NVIDIA GPU)

Three commands:

```bash
git clone https://github.com/ProgramHamza/gecko.git && cd gecko
```
```bash
bash setup.sh
```
```bash
./run.sh start --days 10
```

Then open `runs/first-child/report.html` in a browser.

What each step does:

1. Downloads the code.
2. `setup.sh` (takes a few minutes the first time):
   - creates a Python virtual environment in `.venv`
   - installs PyTorch (the Linux wheels include CUDA) and the other packages
   - runs the unit tests
   - downloads the child model (TinyStories-33M, about 130 MB)
   - installs [Ollama](https://ollama.com) if it is missing (it may ask for your sudo
     password) and pulls the teacher model `llama3.1:8b` (about 5 GB)
3. `start` creates the run on first use, then simulates 10 days. Run it again to
   continue from where it stopped.

**Before you start:** check that the NVIDIA driver works. `nvidia-smi` should print
your GPU. You also need `git`, `curl` and Python venv support. If any are missing:
`sudo apt install git curl python3-venv`.

If you don't want Ollama, use `SKIP_OLLAMA=1 bash setup.sh` and set
`teacher.backend: scripted` in `config.yaml`. That teacher gives template answers
and needs no LLM.

### The teacher

| `teacher.backend` | What it is | Setup |
|---|---|---|
| `ollama` (default) | Local model on your GPU, free | done by `setup.sh` |
| `anthropic` | Claude via the API (default `claude-opus-5-5`), paid per call | `WITH_CLAUDE=1 bash setup.sh`, then `cp .env.example .env` and put your key in it |
| `scripted` | No LLM, template answers | nothing |

The 33M child and an 8B teacher at Ollama's default 4-bit quantisation fit
together in 16 GB of VRAM. The Claude teacher uses low effort and
server-side refusal fallbacks; any failure falls back to the scripted answer and
is counted in the report.

## Running it

```bash
./run.sh start --days 10      # create the run if needed, then live 10 more days
./run.sh status               # where the run is
./run.sh report               # rebuild runs/first-child/report.html
```

`start` is the same as `init` (create the world, the day-0 snapshot and the day-0
tests) followed by `run --days N`. Both are also available on their own. If the
Ollama server is not running (for example after a reboot), start it with
`ollama serve &` or `sudo systemctl start ollama`.

Open `runs/first-child/report.html` in a browser for the learning curve, the
knowledge map (which fact was known on which day), learning-to-learn, the
milestones, self-knowledge, growth events and the diary. Each day also has a
readable page in `runs/first-child/diary/`.

### Talking to it

```bash
./run.sh chat --day 12                     # chat with the day-12 child
./run.sh compare "What does the zibo do when it is rainy?"
./run.sh compare "When it is rainy, the zibo" --choices "hums.|glows.|floats."
```

`compare` asks every saved day the same thing. That is the "five days ago I
wouldn't have known this" view, with the real weights from each day. Start a
chat line with `!` to let it continue text instead of answering a question.
Creature names for your run are printed by `init` and listed in `world.json`.

## Experiments to try

Each run gets its own folder. Copy `config.yaml`, change one thing, and give it
a new `run_name` (or use `--run`):

- **Rewrite vs grow:** `sleep.train_mode: full` vs `growth_only` (base frozen,
  only new modules learn) vs `base_only` with `growth.enabled: false`.
- **No sleep replay:** `replay_ratio: 0` and `rehearsal_ratio: 0`. Watch for
  forgetting in the knowledge map.
- **Curiosity:** `learning_progress` vs `surprise` vs `random`.
- **No autobiography:** `autobiographical: false`. Can it still say when it learned something?
- **Teacher:** a local 8B model vs Claude vs scripted.
- **Learning to learn:** creatures arrive on `arrival_days`. If later ones are
  learned in fewer days, the learning itself improved.
- **Bigger child:** `roneneldan/TinyStories-Instruct-33M` or another small causal LM
  (GPT-2, GPT-Neo, Llama-style layouts are supported).

The day-0 run is the baseline, and facts about creatures that have not arrived
yet are a built-in chance-level control in the learning curve.

## What to expect

- Early question-style probes will be weak: TinyStories has never seen "Q: / A:".
  It learns the format from the teacher's answers during sleep.
- Fill-in-the-sentence probes should improve within a few days of a rule being taught.
- "When did I learn this?" depends on the autobiographical memories. "N days ago,
  did I know this?" needs the model to do date arithmetic, which a 33M model will
  likely fail. That is a finding, not a bug, and the log-based answer is reported
  next to it for comparison.
- Some apparent development will be the model imitating story patterns. The
  logs (what it trained on, what changed) are there to separate real learning
  from that.

## Files

```
config.yaml          settings (copied into each run)
devlm/world.py       the meadow: invented creatures, hidden rules, daily events
devlm/child.py       the child model: surprise, scoring, generation, training, snapshots
devlm/growth.py      reserved zero-initialised modules, one per layer
devlm/curiosity.py   what to ask about
devlm/teacher.py     Ollama / Claude / scripted teacher with answer checking
devlm/sleep.py       what to consolidate each night and when to grow
devlm/probes.py      fact and milestone tests
devlm/metacog.py     "did I know this N days ago?" and ground truth
devlm/simulation.py  the daily loop
devlm/report.py      report.html
runs/<name>/         everything a run produces (gitignored)
```

Run files: `memory.jsonl` (every episode, with day and source), `sleep.jsonl`,
`probes.jsonl`, `metacog.jsonl`, `metrics.jsonl`, `snapshots/day_XXX.safetensors`
(~130 MB each; set `snapshots.keep_every` to save space), `diary/`.

If a run crashes mid-day, run it again: the unfinished day is discarded and redone.

## Tests

```bash
source .venv/bin/activate
python -m pytest -q --cov=devlm
```

The neural tests use a tiny random GPT-Neo built in memory, so they need no download.
