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

The goal of this project is to redistribute VNF (or VM) deployed inefficiently on multiple servers to maximize the performance of SFC while operating with as few servers as possible. After this process, the unused machine goes to sleep mode. So, I named this project `SDN Lullaby`.

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

## Published at

<i>Eui-Dong Jeong, Jae-Hyoung Yoo, James Won-Ki Hong, ["SDN Lullaby: VM Consolidation for SDN using Transformer-Based Deep Reinforcement Learning"](https://ieeexplore.ieee.org/abstract/document/10327902), 19th International Conference on Network and Service Management (CNSM 2023), Niagara Falls, Canada, Oct. 30 – Nov. 2, 2023.</i>


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

  <img width="45%" src="/resource/architecture-1.png">
  <img width="49.2%" src="/resource/architecture-2.png">

</div>

## Further More

I also implemented PPO-based Implementation. If you are interesting in it, go to [ppo branch](https://github.com/euidong/sdn-lullaby/tree/ppo).

## Reference

- Logo Image
  - <a href="https://www.flaticon.com/free-icons/lullaby" title="lullaby icons">Lullaby icons created by Freepik - Flaticon</a>
  - <a href="https://www.flaticon.com/free-icons/data-server" title="data server icons">Data server icons created by The Chohans Brand - Flaticon</a>
