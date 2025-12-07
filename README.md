# Impossible Game RL Agent

Proiect Reinforcement Learning - Agent care învață să joace Geometry Dash

## 📋 Echipa
- Membru 1
- Membru 2  
- Membru 3

## 🎯 Descriere

Agent RL capabil să învețe să joace Geometry Dash prin:
- **3 algoritmi RL**: Q-Learning (tabular), DQN (deep), PPO (policy-based)
- **Environment Gymnasium custom** adaptat din pygame
- **Analiză comparativă** completă a performanței

## 🚀 Instalare

```bash
git clone <repository-url>
cd geometry-dash-rl
pip install -r requirements.txt
```

## 📦 Structură Proiect

```
geometry-dash-rl/
├── environment/          # Mediul Gymnasium
├── agents/              # Implementări algoritmi
│   ├── tabular/        # Q-Learning, SARSA
│   ├── deep/           # DQN
│   └── policy/         # PPO
├── training/           # Scripts antrenament
├── evaluation/         # Evaluare și comparație
├── analysis/           # Generare grafice
└── results/            # Modele salvate
```

## 🎮 Utilizare

### Antrenament

```bash
# Q-Learning (5000 episoade, ~2-3 ore)
python training/train_q_learning.py

# DQN (2000 episoade, ~4-6 ore cu GPU)
python training/train_dqn.py

# PPO (1M timesteps, ~6-8 ore)
python training/train_ppo.py
```

### Evaluare

```bash
# Evaluare individuală
python evaluation/evaluate.py

# Comparație între agenți
python evaluation/compare_agents.py

# Generare grafice
python analysis/plot_results.py
```

### Vizualizare Agent

```bash
# Q-Learning
python evaluation/visualize_agent.py --agent q_learning --model results/models/q_learning_agent.pkl

# DQN
python evaluation/visualize_agent.py --agent dqn --model results/models/dqn_agent.pth

# PPO
python evaluation/visualize_agent.py --agent ppo --model results/models/ppo_agent
```

## 📊 Rezultate
