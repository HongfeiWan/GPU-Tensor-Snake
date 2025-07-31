import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import threading

# 设置线程数以避免GIL问题
torch.set_num_threads(1)

# 创建线程锁来保护CUDA操作
cuda_lock = threading.Lock()

# 检测GPU可用性并设置设备
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
except Exception as e:
    device = torch.device("cpu")

# 如果从外部设置了设备，使用外部设置
if 'device' in globals():
    pass  # 使用外部设置的设备

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name, map_location=device))
            self.eval()
            return True
        return False


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # 确保模型在正确的设备上
        self.model = self.model.to(device)

    def train_step(self, state, action, reward, next_state, done):
        # 使用线程锁保护CUDA操作
        with cuda_lock:
            # 确保所有张量都在正确的设备上
            if not isinstance(state, torch.Tensor):
                # 如果是列表，先转换为numpy数组
                if isinstance(state, list):
                    # 检查列表中的元素类型
                    if len(state) > 0 and isinstance(state[0], np.ndarray):
                        # 如果是numpy数组的列表，先堆叠
                        try:
                            state = np.vstack(state).astype(np.float32)
                        except ValueError:
                            # 如果vstack失败，尝试其他方法
                            state = np.array([s.flatten() if isinstance(s, np.ndarray) else s for s in state], dtype=np.float32)
                    else:
                        state = np.array(state, dtype=np.float32)
                elif isinstance(state, np.ndarray):
                    state = state.astype(np.float32)
                state = torch.tensor(state, dtype=torch.float, device=device)
            else:
                state = state.to(device)
            
        if not isinstance(next_state, torch.Tensor):
            # 如果是列表，先转换为numpy数组
            if isinstance(next_state, list):
                # 检查列表中的元素类型
                if len(next_state) > 0 and isinstance(next_state[0], np.ndarray):
                    # 如果是numpy数组的列表，先堆叠
                    try:
                        next_state = np.vstack(next_state).astype(np.float32)
                    except ValueError:
                        # 如果vstack失败，尝试其他方法
                        next_state = np.array([s.flatten() if isinstance(s, np.ndarray) else s for s in next_state], dtype=np.float32)
                else:
                    next_state = np.array(next_state, dtype=np.float32)
            elif isinstance(next_state, np.ndarray):
                next_state = next_state.astype(np.float32)
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        else:
            next_state = next_state.to(device)
            
        if not isinstance(action, torch.Tensor):
            # 如果是列表，先转换为numpy数组
            if isinstance(action, list):
                # 检查列表中的元素类型
                if len(action) > 0 and isinstance(action[0], np.ndarray):
                    # 如果是numpy数组的列表，先堆叠
                    try:
                        action = np.vstack(action).astype(np.int64)
                    except ValueError:
                        # 如果vstack失败，尝试其他方法
                        action = np.array([s.flatten() if isinstance(s, np.ndarray) else s for s in action], dtype=np.int64)
                else:
                    action = np.array(action, dtype=np.int64)
            elif isinstance(action, np.ndarray):
                action = action.astype(np.int64)
            action = torch.tensor(action, dtype=torch.long, device=device)
        else:
            action = action.to(device)
            
        if not isinstance(reward, torch.Tensor):
            # 如果是列表，先转换为numpy数组
            if isinstance(reward, list):
                # 检查列表中的元素类型
                if len(reward) > 0 and isinstance(reward[0], np.ndarray):
                    # 如果是numpy数组的列表，先堆叠
                    try:
                        reward = np.vstack(reward).astype(np.float32)
                    except ValueError:
                        # 如果vstack失败，尝试其他方法
                        reward = np.array([s.flatten() if isinstance(s, np.ndarray) else s for s in reward], dtype=np.float32)
                else:
                    reward = np.array(reward, dtype=np.float32)
            elif isinstance(reward, np.ndarray):
                reward = reward.astype(np.float32)
            reward = torch.tensor(reward, dtype=torch.float, device=device)
        else:
            reward = reward.to(device)
        
        # (n, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



