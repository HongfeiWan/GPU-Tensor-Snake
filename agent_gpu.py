import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import threading
import time

# 设置环境变量以避免CUDA错误和GIL问题
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 强制使用GPU并设置CUDA选项
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# 确保CUDA初始化在主线程中进行
if torch.cuda.is_available():
    device = torch.device("cuda")
    # 预热GPU以避免首次使用时的延迟
    torch.cuda.empty_cache()
    print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("✓ 使用CPU (GPU不可用)")

# 创建线程锁来保护CUDA操作
cuda_lock = threading.Lock()

# 全局设备变量，确保所有模块使用相同的设备
import model
model.device = device

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        global device, cuda_lock
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        
        # 使用线程锁保护CUDA操作
        with cuda_lock:
            # 确保在主线程中移动模型到GPU
            if device.type == 'cuda':
                # 预热GPU
                torch.cuda.empty_cache()
                # 创建一个小张量来确保CUDA上下文已初始化
                dummy = torch.zeros(1, device=device)
                del dummy
                
            self.model.to(device)
            print(f"✓ 模型成功移动到 {device}")
                
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            try:
                # 使用线程锁保护CUDA操作
                with cuda_lock:
                    # 确保状态张量在正确的设备上
                    if isinstance(state, np.ndarray):
                        # 确保数据类型正确
                        state_array = state.astype(np.float32)
                        state0 = torch.tensor(state_array, dtype=torch.float, device=device).unsqueeze(0)
                    elif isinstance(state, list):
                        # 如果是列表，先转换为numpy数组
                        state_array = np.array(state, dtype=np.float32)
                        state0 = torch.tensor(state_array, dtype=torch.float, device=device).unsqueeze(0)
                    else:
                        state0 = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                    
                    # 使用torch.no_grad()来避免梯度计算，提高推理速度
                    with torch.no_grad():
                        prediction = self.model(state0)
                        move = torch.argmax(prediction).item()
                        final_move[move] = 1
                        
                    # 清理GPU内存
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                print(f"⚠️ 推理失败: {e}")
                # 回退到随机动作
                move = random.randint(0, 2)
                final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    print("🚀 开始训练...")
    
    while True:
        try:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                
                # 使用线程锁保护训练操作
                with cuda_lock:
                    agent.train_long_memory()

                if score > record:
                    record = score
                    with cuda_lock:
                        agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                
                # 定期清理GPU内存
                if device.type == 'cuda':
                    with cuda_lock:
                        torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            print("🔄 重新开始游戏...")
            # 清理GPU内存
            if device.type == 'cuda':
                with cuda_lock:
                    torch.cuda.empty_cache()
            game.reset()
            continue


if __name__ == '__main__':
    train() 