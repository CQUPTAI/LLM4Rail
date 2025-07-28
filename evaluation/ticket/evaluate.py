import json, os, argparse
import pandas as pd
from tqdm import tqdm
from service.agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument('--userFile', '-u', required=True, action='store')
parser.add_argument('--config', '-c', required=True, action='store')
parser.add_argument('--saveDir', '-s', required=True, action='store')
parser.add_argument('--model', '-m', required=True, choices=['qwen-3', 'gpt-4o'], action='store')
parser.add_argument('--GPTKey', '-g', required=True, action='store')
parser.add_argument('--QwenKey', '-q', required=True, action='store')
parser.add_argument('--amapKey', '-a', required=True, action='store')
args = parser.parse_args()

users = pd.read_csv(args.userFile)
config_path = args.config
result_path = f"{args.saveDir}/results.json"
log_path = f"{args.saveDir}/log.txt"
os.environ['GPTKey'] = args.GPTKey
os.environ['QwenKey'] = args.QwenKey
os.environ['amapKey'] = args.amapKey

with open(config_path, encoding="utf-8") as f:
    conf = json.load(f)
file = conf['file']
start = conf['start']
end = conf['end']
offset = conf['offset']

agent = Agent(args.model)

with open(f"{file}", encoding="utf-8") as f:
    lines = f.readlines()
lines = [line.split(',') for line in lines]

for query_type, label, query in tqdm(lines[start+offset:end+1]):
    query = query.strip()
    history = []

    user = users.sample(n=1).to_dict()
    idx = list(user['身份证号码'].keys())[0]
    user_info = {}
    for key in ['性别', '年龄', '出生地']:
        user_info[key] = user[key][idx]
    user_info['当前日期'] = "2025-05-11"
    user_info['所在地'] = user_info['出生地']

    trajectory = {'user_info': user_info,
                  'query_type': query_type,
                  'label': label,
                  'trajectory': []}

    max_trial = 3
    success = False
    for trial in range(1, max_trial+1):
        try:
            agent.query(user_info, history, query, trajectory)
            trajectory['trial_count'] = trial
            with open(result_path, encoding='utf-8', mode='a+') as f:
                f.write(str(trajectory) + '\n')
            success = True
            break
        except Exception as e:
            with open(log_path, encoding='utf-8', mode='a+') as f:
                log = {"error": str(e), "traj": trajectory}
                f.write(str(log)+"\n")
            trajectory['trajectory'] = []
    if not success:
        trajectory['trajectory'].append('Failed!')
        with open(result_path, encoding='utf-8', mode='a+') as f:
            f.write(str(trajectory) + '\n')

    conf['offset'] = conf['offset'] + 1
    with open(config_path, encoding='utf-8', mode='w') as f:
        f.write(json.dumps(conf))


