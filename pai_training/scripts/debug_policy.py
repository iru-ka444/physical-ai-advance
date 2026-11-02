import argparse
import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--repo-id', type=str, required=True)
    parser.add_argument('--policy-path', type=str, required=True)
    parser.add_argument('--episode-index', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    print(f'Loaded dataset: {len(dataset)} samples, {dataset.num_episodes} episodes')

    print('Loading weights from local directory')
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.eval()
    policy.to(args.device)

    from_idx = dataset.episode_data_index['from'][args.episode_index].item()
    to_idx = dataset.episode_data_index['to'][args.episode_index].item()

    gt_actions = []
    pred_actions = []

    for idx in range(from_idx, to_idx):
        sample = dataset[idx]
        
        # 画像テンソルの柔軟な取得
        img_tensor = None
        for k, v in sample.items():
            if 'image' in k:
                img = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
                img_tensor = img.unsqueeze(0).to(args.device)
                break
        
        # 状態テンソルの柔軟な取得
        state_raw = sample.get('observation.state', sample.get('observation.environment_state'))
        state_tensor = state_raw if isinstance(state_raw, torch.Tensor) else torch.tensor(state_raw, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0).to(args.device)

        # モデル設定に基づいたバッチ構築
        obs = {}
        for img_key in policy.config.image_features:
            obs[img_key] = img_tensor
        obs['observation.state'] = state_tensor
        obs['observation.environment_state'] = state_tensor

        gt_action = sample['action']
        if not isinstance(gt_action, torch.Tensor):
            gt_action = torch.tensor(gt_action, dtype=torch.float32)
        
        with torch.no_grad():
            pred_action = policy.select_action(obs).squeeze(0).cpu().numpy()
        
        gt_actions.append(gt_action.cpu().numpy())
        pred_actions.append(pred_action)

    gt_actions = np.array(gt_actions)
    pred_actions = np.array(pred_actions)

    print(f'=== Episode {args.episode_index} Evaluation Results ===')
    print(f'Total Steps: {len(gt_actions)}')
    
    # 各アクション次元の平均絶対誤差（MAE）を算出
    mae = np.mean(np.abs(gt_actions - pred_actions), axis=0)
    print(f'Mean Absolute Error (MAE) per dimension: {mae}')

if __name__ == '__main__':
    main()