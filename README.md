# VM Consolidation

DQN based VM Consolidation code.

<div>
  <img width="49.3%" src="/doc/episode0.gif">
  <img width="49.3%" src="/doc/episode950.gif">
</div>

Left gif show episode 0 result, Right one show episode 950 result.

- \# of SRV : 4
- \# of VNF : Random
- \# of SFC : 4
- Each Server CPU Capacity : 8
- Each Server Memory Capacity : 32

## Dependency

- GPU : RTX2060
- CUDA : 10.2
- python : 3.8.16
- torch : 1.6.0
- platform : Window (Recommended)
- IDE : Vscode (Recommended)
- ffmpeg : (for animation rendering)

## Run

### 1. Install ffmpeg

<https://ffmpeg.org/download.html>

change below code in animator/animator.py

```python
plt.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg-6.0-essentials_build\\bin\\ffmpeg.exe'
```

### 2. Install python packages

```bash
conda create --name vm-consolidation --file requirements.txt -c pytorch -c conda-forge -c anaconda
conda activate vm-consolidation
```

### 3. Run main.py  

```bash
python main.py
```

## Project Structure

```plaintext
|
|- dataType.py  # Data Type code
|- env.py       # Environment code
|- main.py      # Main code
|- agent/        # Agent 
|  |- dqn.py    # DQN Agent
|  |- model.py  # DQN Model
|  `- ql.py     # Q-Learning based Agent (not working now) 
|- api/
|  |- api.py             # API Abstract Class
|  |- niTestbed.py       # NI testbed api (for real testbed) (not implemented)
|  |- simulator_test.py  # Simulator test code
|  `- simulator.py       # Simulator api (for simulating)
|- animator/
|  |- animator.py       # Animator code
|  `- animator_test.py  # Animator test code
|- result/ # Result animations
`- param/ # DQN model parameters
```
