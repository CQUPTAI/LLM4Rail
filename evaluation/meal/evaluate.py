import json, argparse, os
from tqdm import tqdm
from evaluation.meal.models.user_simulator import UserSimulator
from evaluation.meal.models.zeroshot import ZeroShot_CRS
from evaluation.meal.models.feature_augmented import FeatureAugmented_CRS

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, action='store')
parser.add_argument('--saveDir', '-s', required=True, action='store')
parser.add_argument('--model', '-m', required=True, choices=['qwen-3', 'gpt-4o'], action='store')
parser.add_argument('--rec_model', '-r', required=True, choices=['zero-shot', 'feature-augmented'], action='store')
parser.add_argument('--topk', '-t', required=True, type=int, action='store')
parser.add_argument('--GPTKey', '-g', required=True, action='store')
parser.add_argument('--QwenKey', '-q', required=True, action='store')
args = parser.parse_args()

topK = args.topk
model = args.model
rec_model = args.rec_model
config_path = args.config
result_path = f"{args.saveDir}/{rec_model}/results.json"
traj_path = f"{args.saveDir}/{rec_model}/trajectory.json"
log_path = f"{args.saveDir}/{rec_model}/log.json"
os.environ['GPTKey'] = args.GPTKey
os.environ['QwenKey'] = args.QwenKey

user_simulator = UserSimulator()
if rec_model == 'zero-shot':
    crs = ZeroShot_CRS(model, topK=topK)
else:
    crs = FeatureAugmented_CRS(model, topK=topK)

with open(config_path, encoding="utf-8") as f:
    conf = json.load(f)
start = conf['start']
end = conf['end']
offset = conf['offset']
max_trials = conf['max_trials']
max_num_turns = conf['max_num_turns']
with open(conf['data_path'], encoding='utf-8') as f:
    target_items = [line.strip() for line in f.readlines()]

pbar = tqdm(target_items[start+offset:end+1])
for sample in pbar:
    id, target_item = sample.split(',')
    pbar.set_description(target_item)
    print(f"----------目标物品：{target_item}----------\n")

    success = False
    user_simulator.set_target(target_item)
    result_dict = {"target_item": target_item, "success": False, "trial_count": 0,"final_response": None, "rec_list": None}

    for trial in range(max_trials):
        exception = False
        try:
            history = []
            seeker_response = user_simulator(history)
            history.append({"role": "seeker", "content": seeker_response})
            print(seeker_response)

            for round in range(max_num_turns):
                pbar.set_postfix({"#trial": trial + 1, "dialog round": round + 1})

                rec_response, rec_list = crs(history)
                history.append({"role": "recommender", "content": rec_response, "rec_list": rec_list})
                print(rec_response)

                seeker_response = user_simulator(history)
                history.append({"role": "seeker", "content": seeker_response})
                print(seeker_response)

                if "这正是我想要的，谢谢你！" in seeker_response:
                    result_dict['success'] = True
                    result_dict['final_response'] = seeker_response
                    result_dict['rec_list'] = rec_list
                    success = True
                    break

                if target_item in seeker_response:
                    raise Exception("Data Leakage Error")
        except Exception as err:
            exception = True
            log = {'target_item': target_item, "error": str(err), "trajectory": history}
            with open(log_path, encoding='utf-8', mode='a+') as f:
                f.write(str(log)+'\n')
        if not exception:
            break

    traj = {"target_item":target_item, "trajectory":history}
    with open(traj_path, encoding='utf-8', mode='a+') as f:
         f.write(str(traj)+'\n')

    result_dict['trial_count'] = trial+1
    with open(result_path, encoding='utf-8', mode='a+') as f:
         f.write(str(result_dict)+'\n')

    conf['offset'] = conf['offset'] + 1
    with open(config_path, encoding='utf-8', mode='w') as f:
        f.write(json.dumps(conf))