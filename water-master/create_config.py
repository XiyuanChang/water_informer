import json

with open("config_ablation.json") as f:
    config = json.load(f)

# for i in range(16):
    # config['exclude'] = i+1
    # json.dump(config, open(f"ablation_small_donet/config_ablation_{i+1}.json", "w"), indent=4)

for i in range(16):
    config['exclude'] = i+1
    json.dump(config, open(f"ablation_small_donet/config_ablation_{i+1}.json", "w"), indent=4)