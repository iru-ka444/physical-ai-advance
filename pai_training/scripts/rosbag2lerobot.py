#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class RosbagToLeRobot:
    def __init__(self, bag_dir, output_dir, config_path, target_freq=None):
        self.bag_path = Path(bag_dir)
        self.output_dir = Path(output_dir)
        self.target_freq = target_freq

        self.episode_dirs = sorted(p.parent for p in self.bag_path.rglob('*.db3'))
        if not self.episode_dirs:
            raise FileNotFoundError(f"No rosbag2 episode directories found in {bag_dir}")

        print(f"Found {len(self.episode_dirs)} episode directories:")
        for ep_dir in self.episode_dirs:
            print(f"  - {ep_dir.name}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        features = config['features']
        # shape を list から tuple へ変換 (LeRobot の型不一致防止)
        for ft in features.values():
            if "shape" in ft and ft["shape"] is not None:
                ft["shape"] = tuple(ft["shape"])

        robot_type = config.get('robot_type', 'drone')
        repo_id = config.get('repo_id', 'custom_drone_flight')
        task_name = config.get('task_name', 'teleop')

        self.task_name = task_name

        if self.output_dir.exists():
            print(f"[Warning] Directory {self.output_dir} already exists. Removing and recreating...")
            shutil.rmtree(self.output_dir)

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=target_freq or 30,
            root=self.output_dir,
            robot_type=robot_type,
            features=features,
            use_videos=True,
        )

        if "teleop" not in self.dataset.meta.task_to_task_index:
            self.dataset.meta.add_task("teleop")

    def _get_messages(self, episode_dir):
        messages = {
            '/front_camera/image_raw': [],
            '/drone/odom': [],
            '/drone/cmd_vel': [],
        }
        with Reader(episode_dir) as reader:
            connections = [x for x in reader.connections if x.topic in messages]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                messages[connection.topic].append((msg, timestamp))
        return messages

    def _sample_and_hold(self, times, values, target_times):
        values = np.asarray(values)
        idx = np.searchsorted(times, target_times, side='right') - 1
        idx = np.clip(idx, 0, len(times) - 1)
        return values[idx]

    def _msg_to_image(self, msg):
        h, w = msg.height, msg.width
        encoding = msg.encoding.lower()
        data = msg.data
        img = np.frombuffer(data, dtype=np.uint8)
        
        if 'bgr' in encoding:
            img = img.reshape((h, w, -1))[:, :, ::-1]
        elif 'rgb' in encoding:
            img = img.reshape((h, w, -1))
        elif 'mono' in encoding or '8uc1' in encoding:
            img = img.reshape((h, w))
            img = np.stack([img] * 3, axis=-1)
        else:
            img = img.reshape((h, w, -1))
            if img.shape[-1] == 4:
                img = img[:, :, :3]
        return Image.fromarray(img)

    def convert(self):
        print("\nFirst pass: calculating episode lengths...")
        episode_lengths = []
        max_frames = 0

        for episode_dir in self.episode_dirs:
            try:
                messages = self._get_messages(episode_dir)
                img_msgs = messages['/front_camera/image_raw']
                odom_msgs = messages['/drone/odom']
                cmd_msgs = messages['/drone/cmd_vel']

                if not img_msgs or not odom_msgs or not cmd_msgs:
                    print(f"Warning: Missing messages in {episode_dir.name}")
                    episode_lengths.append(0)
                    continue

                img_times = np.array([ts for _, ts in img_msgs])
                odom_times = np.array([ts for _, ts in odom_msgs])
                cmd_times = np.array([ts for _, ts in cmd_msgs])

                start_time = max(img_times[0], odom_times[0], cmd_times[0])
                end_time = min(img_times[-1], odom_times[-1], cmd_times[-1])

                target_freq = self.target_freq or 30
                num_frames = int((end_time - start_time) * target_freq / 1e9)
                num_frames = max(1, num_frames)
                max_frames = max(max_frames, num_frames)
                episode_lengths.append(num_frames)
                print(f"Episode {episode_dir.name}: {num_frames} frames")
            except Exception as e:
                print(f"Error processing episode {episode_dir.name}: {e}")
                episode_lengths.append(0)
                continue

        print(f"\nLongest episode: {max_frames} frames")

        for idx, episode_dir in enumerate(self.episode_dirs):
            num_frames = episode_lengths[idx]
            if num_frames == 0:
                print(f"Skipping empty/invalid episode {episode_dir.name}")
                continue

            print(f"\nProcessing episode {idx} from {episode_dir.name}...")
            try:
                messages = self._get_messages(episode_dir)
                img_msgs = messages['/front_camera/image_raw']
                odom_msgs = messages['/drone/odom']
                cmd_msgs = messages['/drone/cmd_vel']

                img_times = np.array([ts for _, ts in img_msgs])
                odom_times = np.array([ts for _, ts in odom_msgs])
                cmd_times = np.array([ts for _, ts in cmd_msgs])

                odom_data = np.array([
                    [
                        m.pose.pose.position.x,
                        m.pose.pose.position.y,
                        m.pose.pose.position.z,
                        m.twist.twist.linear.x,
                        m.twist.twist.linear.y,
                        m.twist.twist.linear.z,
                    ]
                    for m, _ in odom_msgs
                ])

                cmd_data = np.array([
                    [
                        m.linear.x,
                        m.linear.y,
                        m.linear.z,
                        m.angular.z,
                    ]
                    for m, _ in cmd_msgs
                ])

                start_time = max(img_times[0], odom_times[0], cmd_times[0])
                end_time = min(img_times[-1], odom_times[-1], cmd_times[-1])
                target_times = np.linspace(start_time, end_time, num_frames)

                state_sampled = self._sample_and_hold(odom_times, odom_data, target_times)
                action_sampled = self._sample_and_hold(cmd_times, cmd_data, target_times)

                img_indices = np.searchsorted(img_times, target_times, side='right') - 1
                img_indices = np.clip(img_indices, 0, len(img_msgs) - 1)

                for f_idx in range(num_frames):
                    msg_img, _ = img_msgs[img_indices[f_idx]]
                    pil_img = self._msg_to_image(msg_img)

                    frame_data = {
                        "observation.images.front": pil_img,
                        "observation.state": torch.tensor(state_sampled[f_idx], dtype=torch.float32),
                        "action": torch.tensor(action_sampled[f_idx], dtype=torch.float32),
                        "task": self.task_name,
                    }
                    self.dataset.add_frame(frame_data)

                self.dataset.save_episode()
                print(f"Episode {idx} ({episode_dir.name}) saved successfully.")

            except Exception as e:
                print(f"Error saving episode {episode_dir.name}: {e}")

        print(f"\nDataset saved successfully to: {self.output_dir}")


def main():
    config_path = "configs/conversion_config.yaml"
    with open(config_path, 'r') as f:
        conv_cfg = yaml.safe_load(f)

    bag_dir = conv_cfg["bag_dir"]
    output_dir = conv_cfg["output_dir"]
    target_freq = conv_cfg.get("fps", 30)

    converter = RosbagToLeRobot(bag_dir, output_dir, config_path, target_freq)
    converter.convert()


if __name__ == "__main__":
    main()