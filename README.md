# SDN Lullaby

<div align="center">

  ![python](https://img.shields.io/badge/python-3.8.16-brightgreen)
  ![pytorch](https://img.shields.io/badge/pytorch-2.0.1-orange)
  ![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-blueviolet)
  ![pandas](https://img.shields.io/badge/pandas-1.5.2-blue)
  ![ffmpeg](https://img.shields.io/badge/ffmpeg-4.2.2-red)

</div>

<div align="center">

  ![sdn-lullaby](/resource/sdn-lullaby.png)

</div>

VM Consolidation for SDN using Transformer-Based Deep Reinforcement Learning.

The goal of the project is to redistibute VNF (or VM) deployed inefficiently on multiple servers to maximize the performance of SFC while operating with as few servers as possible. After this process, unused machine go sleep mode. So, I named this project `SDN Lullaby`.

Below images show CPU, Memory load by VNF on each server after each action from this model.

<div align="center">

  <img width="49.3%" src="/resource/ppo_edge_load=0.2_init.gif">
  <img width="49.3%" src="/resource/ppo_edge_load=0.2_final.gif">
  
</div>

The left gif displays the results of the initial (untrained) model, while the right gif show the final results of the trained model.

- \# of SRV : 8
- \# of VNF : Random (Maximum 20)
- \# of SFC : 8
- Each Server CPU Capacity : 12
- Each Server Memory Capacity : 32

## Dependency

- GPU : Quadro RTX 5000
- CUDA : 11.8
- python : 3.8.16
- torch : 2.0.1
- platform : Linux (for `torch.multiprocessing`)
- IDE : Vscode (Recommended)
- matplotlib : 3.7.1
- pandas : 1.5.2
- ffmpeg : 4.2.2 (for animation rendering)

## Run

### 1. Install python packages

```bash
conda create --name vnf-consolidation --file requirements.txt -c pytorch -c conda-forge -c anaconda
conda activate vnf-consolidation
```

### 2. Run agent code

#### DQN Agent

```bash
python -m src.agent.dqn
```

#### PPO Agent

```bash
python -m src.agent.ppo
```

## Project Structure

```plaintext
|
|- src/       # Source Code
|- param/     # Saved model parameters
|- result/    # Result metrics & animations
`- resource/  # Resource files
```

## Architecture

<div align="center">

  <img width="500px" src="/resource/architecture.png">

</div>


## Result

### Compare with Baseline System

#### 1. Rule based System

<div>
  <img width="49.3%" src="/resource/rule_edge_load=0.2.gif">
</div>

#### 2. DQN

Left is before training result, right is after training result.

<div>
  <img width="49.3%" src="/resource/dqn_edge_load=0.2_init.gif">
  <img width="49.3%" src="/resource/dqn_edge_load=0.2_final.gif">
</div>

#### 3. PPO (Our Method)

Left is before training result, right is after training result.

<div>
  <img width="49.3%" src="/resource/ppo_edge_load=0.2_init.gif">
  <img width="49.3%" src="/resource/ppo_edge_load=0.2_final.gif">
</div>

### Metrics

Additional performance metrics are available in the following notebook: [performance_metrics.ipynb](./performance_metrics.ipynb).

## Reference

- Logo Image
  - <a href="https://www.flaticon.com/free-icons/lullaby" title="lullaby icons">Lullaby icons created by Freepik - Flaticon</a>
  - <a href="https://www.flaticon.com/free-icons/data-server" title="data server icons">Data server icons created by The Chohans Brand - Flaticon</a>
