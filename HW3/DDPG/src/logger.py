import time
import numpy as np


class Logger:
    def __init__(self, total_steps: int, num_checkpoints: int):
        self.total_steps = total_steps
        self.num_checkpoints = num_checkpoints

        self.start_time = time.time()

        self.episode_returns = []
        self.episode_lengths = []

        self.current_step = 0
        self.current_return = 0
        self.current_length = 0

        self.current_episode = 1

        self.log_interval = 100
        self.window = 5

        self.header_printed = False
        self.checkpoint_interval = max(1, self.total_steps // self.num_checkpoints)

    def log(self, reward: float, termination: bool, truncation: bool):
        self.current_step += 1
        self.current_return += reward
        self.current_length += 1

        if termination or truncation:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            self.current_episode += 1
            self.current_return = 0.0
            self.current_length = 0

    def print_logs(self):
        if self.current_episode > 1:
            if (
                self.current_step % self.log_interval == 0
                and len(self.episode_returns) > 0
            ):
                elapsed_time = time.time() - self.start_time

                progress = 100 * self.current_step / self.total_steps
                mean_reward = (
                    np.mean(self.episode_returns[-self.window :])
                    if len(self.episode_returns) >= self.window
                    else np.mean(self.episode_returns)
                )
                mean_ep_length = (
                    np.mean(self.episode_lengths[-self.window :])
                    if len(self.episode_lengths) >= self.window
                    else np.mean(self.episode_lengths)
                )

                # Format elapsed time into hh:mm:ss
                hours, remainder = divmod(int(elapsed_time), 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

                if not self.header_printed:
                    log_header = (
                        f"{'Progress':>8}  |  "
                        f"{'Step':>8}  |  "
                        f"{'Episode':>8}  |  "
                        f"{'Mean Rew':>8}  |  "
                        f"{'Mean Len':<7}  |  "
                        f"{'Time':>8}"
                    )
                    print(log_header)
                    self.header_printed = True

                log_string = (
                    f"{progress:>7.1f}%  |  "
                    f"{self.current_step:>8,}  |  "
                    f"{self.current_episode:>8,}  |  "
                    f"{mean_reward:>8.2f}  |  "
                    f"{mean_ep_length:>8.1f}  |  "
                    f"{formatted_time:>8}"
                )

                print(f"\r{log_string}", end="")

        # Check if a checkpoint is reached
        if self.current_step % self.checkpoint_interval == 0:
            print()
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_step = self.current_step
