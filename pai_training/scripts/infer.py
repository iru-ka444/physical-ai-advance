import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import torch
import numpy as np
import argparse
from lerobot.common.policies.act.modeling_act import ACTPolicy

class InferenceNode(Node):
    def __init__(self, model_path):
        super().__init__('drone_act_infer')
        self.cmd_pub = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/front_camera/image_raw', self.image_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/drone/odom', self.odom_callback, 10)
        
        self.get_logger().info(f'Loading weights from {model_path}...')
        self.policy = ACTPolicy.from_pretrained(model_path)
        self.policy.eval()
        
        self.current_image = None
        self.current_state = None
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Drone Inference Node Started!')

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            np_arr = np_arr[:, :, ::-1]
        
        img_tensor = torch.from_numpy(np_arr.copy()).permute(2, 0, 1).float() / 255.0
        self.current_image = img_tensor.unsqueeze(0)

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        lin_vel = msg.twist.twist.linear
        # 学習データ（LeRobot）が期待する6次元の状態ベクトル（位置3 + 線速度3）
        state = [pos.x, pos.y, pos.z, lin_vel.x, lin_vel.y, lin_vel.z]
        self.current_state = torch.tensor([state], dtype=torch.float32)

    def control_loop(self):
        if self.current_image is None or self.current_state is None:
            return
        
        batch = {}
        for key in self.policy.config.image_features:
            batch[key] = self.current_image
        
        batch['observation.state'] = self.current_state
        batch['observation.environment_state'] = self.current_state
        
        with torch.no_grad():
            action = self.policy.select_action(batch)
            act_vec = action.squeeze().cpu().numpy()
            
            cmd = Twist()
            if act_vec.ndim == 0:
                cmd.linear.x = float(act_vec)
            else:
                cmd.linear.x = float(act_vec[0]) if len(act_vec) > 0 else 0.0
                cmd.linear.y = float(act_vec[1]) if len(act_vec) > 1 else 0.0
                cmd.linear.z = float(act_vec[2]) if len(act_vec) > 2 else 0.0
                cmd.angular.z = float(act_vec[3]) if len(act_vec) > 3 else 0.0
            
            self.cmd_pub.publish(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()
    
    rclpy.init()
    node = InferenceNode(args.model_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()