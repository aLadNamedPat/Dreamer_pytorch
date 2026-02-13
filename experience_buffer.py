import random
import torch

class ExperienceBuffer:
    def __init__(self, max_size=1000000):
        """Store episodes for training"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.episode_lengths = []
        self.max_size = max_size

    def add_episode(self, obs_seq, action_seq, reward_seq):
        """Add a complete episode to the buffer"""
        self.observations.extend(obs_seq)
        self.actions.extend(action_seq)
        self.rewards.extend(reward_seq)
        self.episode_lengths.append(len(obs_seq))

        # Remove oldest episodes if buffer is too large
        while len(self.observations) > self.max_size:
            oldest_episode_len = self.episode_lengths.pop(0)
            self.observations = self.observations[oldest_episode_len:]
            self.actions = self.actions[oldest_episode_len:]
            self.rewards = self.rewards[oldest_episode_len:]

    def get_random_sequences(self, batch_size, sequence_length):
        """Sample sequences that don't cross episode boundaries"""
        if len(self.observations) < sequence_length:
            return None, None, None        
        valid_starts = []
        current_idx = 0
        
        for ep_len in self.episode_lengths:
            if ep_len >= sequence_length:
                # Can sample any start position where we have seq_len steps remaining
                for start in range(current_idx, current_idx + ep_len - sequence_length + 1):
                    valid_starts.append(start)
            current_idx += ep_len
        
        if len(valid_starts) < batch_size:
            # Not enough valid starting points - sample with replacement
            if len(valid_starts) == 0:
                return None, None, None
            chosen_starts = random.choices(valid_starts, k=batch_size)
        else:
            chosen_starts = random.sample(valid_starts, batch_size)
        
        obs_batch = []
        action_batch = []
        reward_batch = []
        
        for start_idx in chosen_starts:
            obs_seq = self.observations[start_idx:start_idx + sequence_length]
            action_seq = self.actions[start_idx:start_idx + sequence_length]
            reward_seq = self.rewards[start_idx:start_idx + sequence_length]
            
            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
        
        return (
            torch.stack([torch.stack(seq) for seq in obs_batch]),
            torch.stack([torch.stack(seq) for seq in action_batch]),
            torch.stack([torch.stack(seq) for seq in reward_batch])
        )

    def merge_buffer(self, other_buffer):
        """Merge another buffer into this one"""
        self.observations.extend(other_buffer.observations)
        self.actions.extend(other_buffer.actions)
        self.rewards.extend(other_buffer.rewards)
        self.episode_lengths.extend(other_buffer.episode_lengths)

        while len(self.observations) > self.max_size:
            oldest_episode_len = self.episode_lengths.pop(0)
            self.observations = self.observations[oldest_episode_len:]
            self.actions = self.actions[oldest_episode_len:]
            self.rewards = self.rewards[oldest_episode_len:]
