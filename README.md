# NeuroDRL-Driver-Alerts (NDRLDA)

**Deep Reinforcement Learning for Adaptive Driver Vigilance Control and Real-Time Alerting**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

**NeuroDRL-Driver-Alerts (NDRLDA)** is a research-oriented Brain-Computer Interface (BCI) project that introduces a **Deep Reinforcement Learning (DRL)** framework for intelligent, adaptive driver vigilance monitoring and alerting.

The system utilizes the **SEED-VIG** dataset from Shanghai Jiao Tong University (SJTU) to train a reinforcement learning agent that learns an optimal policy for issuing timely and context-aware interventions based on multimodal neurophysiological signals (EEG and EOG). 

Unlike conventional supervised learning methods that merely detect drowsiness, this project treats driver vigilance management as a **sequential decision-making problem**, enabling the agent to proactively maintain safe vigilance levels while minimizing driver distraction.

---

## ✨ Key Features

- **Multimodal Signal Processing**: Combines EEG features (Power Spectral Density & Differential Entropy across 5 frequency bands) with 36 EOG features.
- **Custom Gymnasium Environment**: Realistic simulation of vigilance dynamics using continuous PERCLOS labels as ground truth.
- **Deep Reinforcement Learning**: Proximal Policy Optimization (PPO) with actor-critic architecture.
- **Adaptive Alerting System**: Supports multiple intervention levels (No Alert, Soft, Medium, Strong).
- **Subject-Independent Evaluation**: Rigorous cross-subject validation for real-world applicability.
- **Real-Time Design**: Focus on low-latency inference suitable for edge deployment in vehicles.
- **Comprehensive Visualization**: Per-subject vigilance trajectories and performance metrics.

---

## 🎯 Research Objectives

- Develop an active vigilance control system using Deep Reinforcement Learning.
- Balance safety (preventing critical drowsiness) and user comfort (reducing unnecessary alerts).
- Investigate the effectiveness of DRL in safety-critical BCI applications for intelligent transportation.
- Bridge the gap between passive drowsiness detection and proactive intervention strategies.
- Provide a reproducible benchmark for DRL-based driver state monitoring.

---

## 📊 Dataset

- **Dataset**: SEED-VIG (SJTU Brain-Computer Interface Lab)
- **Signals**: 17-channel EEG, 4-channel Forehead EEG, 36-dimensional EOG features
- **Features**: Differential Entropy (DE) and Power Spectral Density (PSD) in 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **Ground Truth**: Continuous PERCLOS (Percentage of Eye Closure) values (0–1)
- **Task**: Real-time adaptive alerting during simulated driving

> **Note**: The SEED-VIG dataset must be requested from the official [SJTU BCMI website](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html).

---

## 🛠️ Technologies & Dependencies

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Reinforcement Learning**: Stable-Baselines3
- **Environment**: Gymnasium
- **Data Processing**: NumPy, SciPy, scikit-learn
- **Visualization**: Matplotlib

See `requirements.txt` for the complete list.

---

## 📁 Project Structure

```bash
NeuroDRL-Driver-Alerts/
├── data/                    # SEED-VIG .mat files (not included)
├── env/
│   └── vigilance_env.py     # Custom Gymnasium environment
├── utils/
│   └── data_loader.py       # Data loading and preprocessing
├── models/
│   └── ppo_agent.py         # Model training utilities
├── config.py                # Hyperparameters and configurations
├── train.py                 # Training script
├── evaluate.py              # Subject-independent evaluation
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/NeuroDRL-Driver-Alerts.git
   cd NeuroDRL-Driver-Alerts
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SEED-VIG Dataset**
   - Request access from: https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html
   - Place `.mat` files in the `data/` directory

4. **Configure paths** (if needed) in `config.py`

---

## 📋 Usage

### Training the Agent
```bash
python train.py
```

### Evaluating the Model
```bash
python evaluate.py
```

### Custom Configuration
Edit `config.py` to modify:
- Training timesteps
- Learning rate
- Window size
- Test subjects for cross-validation

---

## 📈 Results & Evaluation

The project evaluates performance using the following key metrics:
- Average Cumulative Reward
- Mean PERCLOS
- Dangerous Time Ratio (PERCLOS > 0.70)
- Intervention Rate
- Strong Intervention Frequency

Results are automatically saved in the `evaluation_results/` directory, including:
- JSON summary report
- Per-subject vigilance trajectory plots

---

## 🔬 Research Contributions

- First DRL-based adaptive vigilance control framework using the SEED-VIG dataset.
- Novel formulation of driver alerting as a reinforcement learning problem.
- Balanced reward design addressing both safety and user experience.
- Subject-independent benchmarking with real-time deployment considerations.

---

## 📚 References

- Zheng, W.-L., & Lu, B.-L. (2017). "A multimodal approach to estimating vigilance using EEG and forehead EOG." *Journal of Neural Engineering*.
- Stable-Baselines3 Documentation
- SEED-VIG Dataset Official Page

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📧 Contact

**Your Name**  
Researcher in Brain-Computer Interfaces & Intelligent Transportation Systems  
GitHub: [@yourusername](https://github.com/yourusername)

---

**Built with ❤️ for safer roads through intelligent neurotechnology.**
