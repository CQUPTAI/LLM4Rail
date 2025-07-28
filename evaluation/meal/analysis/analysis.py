import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rec_model', '-r', required=True, action='store')
parser.add_argument('--saveDir', '-s', required=True, action='store')
args = parser.parse_args()
result_path = f'{args.saveDir}/{args.rec_model}/results.json'
traj_path = f"{args.saveDir}/{args.rec_model}/trajectory.json"

with open(result_path, encoding='utf-8') as f:
    samples = [eval(line.strip()) for line in f.readlines()]

with open(traj_path, encoding='utf-8') as f:
    trajs = [eval(line.strip()) for line in f.readlines()]

K = None
for i in range(len(samples)):
    if samples[i]['rec_list'] is not None:
        K = len(samples[i]['rec_list'])
        break
print(f'K={K}')

total_rec = 0
out_of_dataset_rec = 0
for traj in trajs:
    for turn in traj['trajectory']:
        if turn['role']=='recommender' and turn['rec_list'] is not None:
            total_rec += len(turn['rec_list'])
            for rec_item in turn['rec_list']:
                if rec_item is not None:
                    out_of_dataset_rec += 1
print(f'freq={(1-out_of_dataset_rec/total_rec)*100:.2f}')

hit, total = 0, 0
for sample in samples:
    total += 1
    if sample['success']:
        hit += 1
print(f'HR={hit/total*100:.2f}')