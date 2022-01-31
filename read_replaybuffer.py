
import pickle
import stable_baselines3

with open('sac_replaybuffer.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data.size())