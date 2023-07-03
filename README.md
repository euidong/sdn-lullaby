# VM Consolidation

DQN based VM Consolidation code.

Below images show CPU, Memory load by VM on each server after each action from this model.

<div>
  <img width="49.3%" src="/resource/episode0.gif">
  <img width="49.3%" src="/resource/episode950.gif">
</div>

Left gif show episode 0 result, Right one show episode 950 result.

- \# of SRV : 4
- \# of VNF : Random
- \# of SFC : 4
- Each Server CPU Capacity : 8
- Each Server Memory Capacity : 32

## Dependency

- GPU : Quadro RTX 5000
- CUDA : 11.8
- python : 3.8.16
- torch : 2.0.1
- platform : Linux (for `torch.multiprocessing`)
- IDE : Vscode (Recommended)
- ffmpeg : (for animation rendering)

## Run

### 1. Install python packages

```bash
conda create --name vm-consolidation --file requirements.txt -c pytorch -c conda-forge -c anaconda
conda activate vm-consolidation
```

### 2. Run agent code

#### DQN Agent

```bash
python -m src.agent.dqn.py
```

#### PPO Agent

```bash
python -m src.agent.ppo.py
```

## Project Structure

```plaintext
|
|- src/       # Source Code
|- param/     # Saved model parameters
|- result/    # Result metrics & animations
`- resource/  # Resource files
```
