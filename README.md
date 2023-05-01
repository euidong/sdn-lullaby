# VM Consolidation

DQN based VM Consolidation code.

## Dependency

- GPU : RTX2060
- CUDA : 10.2
- torch : 1.6.0
- platform : Window (Recommended)
- IDE : Vscode (Recommended)

## Run

```bash
conda create --name vm-consolidation --file requirements.txt
python main.py
```

## Project Structure

```plaintext
|
|- dataType.py  # Data Type code
|- env.py       # Environment code
|- main.py      # Main code
|- agent        # Agent 
|  |- dqn.py    # DQN Agent
|  |- model.py  # DQN Model
|  `- ql.py     # Q-Learning based Agent (not working now) 
|- api
|   |- api.py             # API Abstract Class
|   |- niTestbed.py       # NI testbed api (for real testbed) (not implemented)
|   |- simulator_test.py  # Simulator test code
|   `- simulator.py       # Simulator api (for simulating)
`- data # DQN model parameters
```
