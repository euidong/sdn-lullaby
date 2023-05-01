# VM Consolidation

## Dependency

- GPU : RTX2060
- CUDA : 10.2
- torch : 1.6.0

## Process

- **Host Detection**(DQL)
  - need to select over/underload host.
  - state: each host capacity and inuse (CPU, Mem)
  - action: host id[one hot]
  - reward: 1 if increase # sleeping host else 0
- **VM Selection**
  - select VM with lowest request.
  - LSTM
    - input
      - vm cpu request
      - vm memory request
    - output
      - one hot encoding of vm id[one hot]
- **VM Placement**(DQL)
  - state: each host capacity and inuse (CPU, Mem)
  - action: host id[one hot]
  - reward: 1 if increase # sleeping host else 0


## DQL

- state
  - vm
    - CPU request
    - Mem request
    - located server CPU capacity
    - located server Mem capacity
    - located server CPU request
    - located server CPU request
    - located edge CPU capacity
    - located edge Mem capacity
    - located edge CPU request
    - located edge Mem request
- action
  - moving vm id
  - sender id
  - receiver id
