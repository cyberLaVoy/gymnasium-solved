# Gymnasium Solved

Welcome to my personal project on reinforcement learning! This repository contains solutions to the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) problem set, using deep reinforcement learning and Q-learning algorithms.

The code in this repository is written in Python and includes both single-threaded and multi-threaded implementations. I created this project to learn about reinforcement learning and see its capabilities firsthand. With this repository, you can train your own agents and experiment with deep reinforcement learning and Q-learning on a variety of environments, including Atari games, classic-control, and toy-text.

## Features
- Explore deep reinforcement learning and Q-learning algorithms and their performance on various environments
- Train agents to solve complex Atari games using multi-threading
- Use the code as a starting point for your own reinforcement learning projects

## Setup

### Hardware and Drivers

1. Nvidia GPU with CUDA cores
2. Graphics driver
3. CUDA
4. cuDNN

The best way to obtain 2, 3, and 4 is to follow the instructions provided by Nvidia.

[Graphics driver (latest)](https://www.nvidia.com/download/index.aspx)

Latest driver version should be compatible with older CUDA versions. Graphics drivers are supposed to be backwards compatible with CUDA versions.

[CUDA (latest)](https://developer.nvidia.com/cuda-downloads)

[CUDA (versions)](https://developer.nvidia.com/cuda-toolkit-archive)

The CUDA version chosen seems to have a dependancy on OS version.

[cuDNN (guide)](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

[cuDNN (versions)](https://developer.nvidia.com/rdp/cudnn-archive)

Make sure to install compatible version combinations for Tensorflow. 

[Version compatibility chart](https://www.tensorflow.org/install/source#gpu)

To verify install and check versions of graphics driver and CUDA run:

```bash
nvidia-smi
```

Install Tensorflow.

```bash
pip install tensorflow
```

Install Gymnasium.

```bash
pip install gymnasium gymnasium[atari] gymnasium[accept-rom-license]
```

Install additional Python library requirements.

```bash
pip install opencv-python numpy matplotlib joblib
```

## Contributing
I welcome contributions to this repository! If you have any feedback, suggestions, or ideas for improvements, please feel free to open an issue or submit a pull request.