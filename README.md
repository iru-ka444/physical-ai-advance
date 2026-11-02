# physical-ai-advance

# 始めに

Dockerのイメージをpullする
```
docker pull airobotbook/ros2-desktop-ai-robot-book-humble:latest
```

コンテナを作って、
```
docker run -d --name pai_ros2 -p 6080:80 -p 9090:9090 -p 9876:9876 --shm-size=512m airobotbook/ros2-desktop-ai-robot-book-humble:latest
```

localhost:6080で接続する。

このgitには一部のファイルを書き換えたものが置かれている。


# エピソードの集め方
エピソードは事前に集めたものが
```
/home/ubuntu/dataset/rosbags/drone_flight
```
に10個入っている

一から集めるには
1つ目のターミナルで
```
source ~/pai_ws/install/setup.bash
ros2 launch custom_drone_description spawn_drone.launch.py
```
2つ目のターミナルで
```
source ~/pai_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/drone/cmd_vel
```
3つ目のターミナルで
```
ros2 bag record -o ~/dataset/rosbags/drone_flight/ep0 \
  /front_camera/image_raw \
  /drone/odom \
  /drone/cmd_vel
  ```
  を行う

  データ変換は
  ```
  cd ~/pai_training
  pixi run convert
  ```

  次は学習を行う
  学習ステップは以下のコマンドで変更できる。（例は2000回）
  ```
  import yaml

config_path = '/home/ubuntu/pai_training/configs/train_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['training_steps'] = 2000

with open(config_path, 'w') as f:
    yaml.dump(config, f)

print('Updated training_steps to 2000!')
"
```

学習は以下のコマンド
```
cd ~/pai_training
pixi run train
```

推論は
1つ目のターミナルで
```
cd ~/pai_training
ros2 launch custom_drone_description spawn_drone.launch.py
```

２つ目のターミナルで
```
cd ~/pai_training
pixi run infer -- --model-path /home/ubuntu/pai_training/outputs/train/drone_flight
```

また、以下のコマンドで軌跡の平均絶対誤差を求めることができる。
```
cd ~/pai_training
pixi run debug-policy \
  --dataset-root /home/ubuntu/dataset/lerobot_dataset/drone_flight \
  --repo-id custom_drone_flight \
  --policy-path /home/ubuntu/pai_training/outputs/train/drone_flight \
  --episode-index 1
  ```