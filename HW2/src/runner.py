import sys
import random
sys.path.append("./src/")

from tqdm import tqdm
from itertools import count

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

import utils
import DQN

class Runner:
    def __init__(self, 
                BATCH_SIZE: int = 128,
                GAMMA: float = 0.99,
                EPS_START: float = 0.9,
                EPS_END: float = 0.05,
                EPS_DECAY: int = 1000,
                TAU: float = 0.005,
                LR: float = 1e-4,
                FULL_MEMORY_LENGTH: float = 10000
            ):
        """
        Parameters
        ----------
        BATCH_SIZE : int
            КОЛИЧЕСТВО ЭПИЗОДОВ, ОТОБРАННЫХ ИЗ БУФЕРА ВОСПРОИЗВЕДЕНИЯ
            
        GAMMA : float
            КОЭФФИЦИЕНТ ДИСКОНТИРОВАНИЯ
            
        EPS_START : float
            НАЧАЛЬНОЕ ЗНАЧЕНИЕ ЭПСИЛОН
            
        EPS_END : float
            КОНЕЧНОЕ ЗНАЧЕНИЕ ЭПСИЛОН
            
        EPS_DECAY : int
            СКОРОСТЬ ЭКСПОНЕНЦИАЛЬНОГО СПАДА ЭПСИЛОН, ЧЕМ БОЛЬШЕ - ТЕМ МЕДЛЕННЕЕ ПАДЕНИЕ
            
        TAU : float
            СКОРОСТЬ ОБНОВЛЕНИЯ ЦЕЛЕВОЙ СЕТИ
            
        LR : float
            СКОРОСТЬ ОБУЧЕНИЯ ОПТИМИЗАТОРА
            
        FULL_MEMORY_LENGTH : float
           ОБЪЕМ REPLAY MEMORY BUFFER
        """
        self.Transition = utils.Transition
        
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.LR = LR
        self.FULL_MEMORY_LENGTH = FULL_MEMORY_LENGTH
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Среда
        self.env = gym.make("LunarLander-v3")
        
        # Получить число действий
        self.n_actions = self.env.action_space.n

        # Получить число степеней свободы состояний
        state, _ = self.env.reset()
        self.n_observations = len(state)        
        
        # Инициилизировать сети: целевую и политики
        # dim s_t = 8, элементы (первые 5) континуальны
        # a_t могут принимать 4 значения (бинарны)
        self.policy_net = DQN.DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN.DQN(self.n_observations, self.n_actions).to(self.device)
        
        # Подгрузить в целевую сеть коэффициенты из сети политики
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Задать оптимайзер
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        
        # Инициализировать Replay Memory buffer
        self.memory = utils.ReplayMemory(self.FULL_MEMORY_LENGTH, device=self.device)
        
        # Счетчик шагов
        self.steps_done = 0
        
        self.episode_durations = []
        self.total_reward = []
        
    def select_action(self, state: torch.Tensor):
        # случайное значение для определения какой шаг будем 
        # делать жадный или случайный
        sample = random.random()
        
        # установка порога принятия решения - уровня epsilon
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp( -1. * self.steps_done / self.EPS_DECAY)
            
        self.steps_done = self.steps_done + 1
        
        
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) вернет наибольшее значение столбца в каждой строке.
                # Второй столбец в результате max - это индекс того места, 
                # где был найден максимальный элемент, 
                # поэтому мы выбираем действие с наибольшим ожидаемым вознаграждением.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Иначы выбираем случайное дайствие
            return torch.tensor([[self.env.action_space.sample()]], 
                                device=self.device, dtype=torch.long)
        
        
    def optimize_model(self, batch_size: int):
        if not batch_size: batch_size = self.BATCH_SIZE
        
        if len(self.memory) < batch_size: 
            return
        
        # Получить из памяти батч
        transitions = self.memory.sample(batch_size)
        # Преобразовать его в namedtuple
        batch = self.Transition(*zip(*transitions))
        
        """
        Для наглядности будем действовать немного по-другому, 
        нежели в обучающем примере, а именно по алгоритму
        из https://arxiv.org/abs/2201.09746 (стр 89-90)
        Нам нужно определить таргет, лосс, сделать шаг градиентного спуска
        В Transition добавил поле done
        
        Такой путь несколько ускоряет обучение.
        """
        
        # Собираем батчи для состояний, действий и наград
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)
        
        # Вычислить Q(s_t, a) - модель вычисляет Q(s_t), 
        # Эти из policy_net, ту, что обучаем
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Определяем таргет с помощью таргет-сети
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state = torch.nan_to_num(next_state)
            next_state_values = self.target_net(next_state).max(1)[0]
            
        
        ones = torch.ones_like(dones).to(self.device)
        dones = ones.sub(dones).reshape(-1)
        
        # Вычисляем ожидаемые Q значения
        expected_state_action_values =  torch.mul(dones, next_state_values) * self.GAMMA + reward_batch
        
        # Объединяем все в общий лосс
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Готовим градиент
        self.optimizer.zero_grad()
        loss.backward()
        # Обрезаем значения градиента - проблемма исчезающего/взрывающего градиента
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        
        self.optimizer.step()


    def run(self):
        if torch.cuda.is_available():
            num_episodes = 1500
        else:
            num_episodes = 500
        
        for i_episode in tqdm(range(num_episodes)):
            episode_reward = 0
            # Для каждого эпизода инициализируем начальное состояние
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # выполняем действия пока не получим флаг done
            # t - считает сколько шагов успели сделать пока шест не упал
            for t in count():
                # выбираем действие [0, 1]
                action = self.select_action(state)
                # Делаем шаг

                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward
        
                # Преобразуем в тензор
                reward = torch.tensor([reward], device=self.device)
                
                # Объединяем done по двум конечным состояниям
                done = terminated or truncated
                done_ = torch.Tensor([[int(done)]]).to(self.device)
                # присваиваем следующее состояние
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, 
                                              device=self.device).unsqueeze(0)
        
                # отправляем в память
                self.memory.push(state, action, next_state, reward, done_)
        
                # переходим на следующее состояние
                state = next_state
        
                # запускаем обучение сети
                self.optimize_model(batch_size=128)
        
                # делаем "мягкое" обновление весов
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + \
                        target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                
                # Если получили terminated or truncated завершаем эпизод обучения
                if done:
                    # добавляем в массив продолжительность эпизода
                    self.episode_durations.append(t + 1)
                    self.total_reward.append(episode_reward)
                    break
        
        print()
        print('Complete')

    