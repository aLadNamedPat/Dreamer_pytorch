import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from setup import *

from models.action_value import Action, Value
from models.RSSM import RSSM
from torch.distributions import Normal, kl_divergence
import wandb
from experience_buffer import ExperienceBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collect_random_episodes(env, num_episodes, max_steps_per_episode=1000):
    """Collect S random seed episodes for initial dataset"""
    buffer = ExperienceBuffer()

    print(f"\n{'='*60}")
    print(f"STARTING RANDOM EPISODE COLLECTION")
    print(f"Target episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"{'='*60}")

    total_steps = 0
    total_reward = 0.0

    for episode in range(num_episodes):
        print(f"\n Episode {episode + 1}/{num_episodes} - Starting...")

        obs_sequence = []
        action_sequence = []
        reward_sequence = []
        episode_reward = 0.0
        episode_steps = 0

        # Reset environment
        print(f"  Resetting environment...")
        obs, info = env.reset()
        # Convert image from (H, W, C) to (C, H, W) and normalize to [0, 1]
        # Use .copy() to ensure contiguous array for PyTorch
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
        obs_sequence.append(obs_tensor)

        # Episode rollout
        for step in range(max_steps_per_episode):
            # Random action sampling
            action = env.action_space.sample()
            action_tensor = torch.tensor(action, dtype=torch.float32)
            action_sequence.append(action_tensor)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            # Convert image from (H, W, C) to (C, H, W) and normalize to [0, 1]
            # Use .copy() to ensure contiguous array for PyTorch
            obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_sequence.append(obs_tensor)
            reward_sequence.append(torch.tensor(reward, dtype=torch.float32))

            episode_reward += reward
            episode_steps += 1

            # Progress indicator every 100 steps
            if (step + 1) % 100 == 0:
                print(f"    Step {step + 1}/{max_steps_per_episode} | Reward: {episode_reward:.2f}")

            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"  Episode ended at step {episode_steps} ({reason})")
                break

        # Add episode to buffer (excluding last observation for action alignment)
        buffer.add_episode(obs_sequence[:-1], action_sequence, reward_sequence)

        total_steps += episode_steps
        total_reward += episode_reward
        avg_reward = total_reward / (episode + 1)
        avg_steps = total_steps / (episode + 1)

        print(f"     Episode {episode + 1} complete!")
        print(f"     Steps: {episode_steps} | Reward: {episode_reward:.3f}")
        print(f"     Running averages - Steps: {avg_steps:.1f} | Reward: {avg_reward:.3f}")

        # Milestone updates
        if (episode + 1) % 5 == 0:
            print(f"\n PROGRESS UPDATE:")
            print(f"   Episodes completed: {episode + 1}/{num_episodes}")
            print(f"   Total timesteps collected: {len(buffer.observations)}")
            print(f"   Average episode length: {avg_steps:.1f} steps")
            print(f"   Average episode reward: {avg_reward:.3f}")

    print(f"\n{'='*60}")
    print(f"RANDOM EPISODE COLLECTION COMPLETE!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total timesteps: {len(buffer.observations)}")
    print(f"Average episode length: {total_steps / num_episodes:.1f} steps")
    print(f"Average episode reward: {total_reward / num_episodes:.3f}")
    print(f"{'='*60}")

    return buffer

def compute_losses(rssm_output, reconstructed_obs, target_obs, predicted_rewards, target_rewards, 
                   free_nats=3.0, debug=False):
    """
    Compute RSSM training losses with summed pixel log-probs.
    """
    prior_states, posterior_states, hiddens, prior_mus, prior_stds, \
        posterior_mus, posterior_stds, rewards = rssm_output
    mse_per_pixel = (reconstructed_obs - target_obs) ** 2
    reconstruction_loss = mse_per_pixel.sum(dim=(2, 3, 4)).mean()

    reward_dist = Normal(predicted_rewards, 1.0)
    if target_rewards.dim() == 2:
        target_rewards = target_rewards.unsqueeze(-1)

    reward_loss = -reward_dist.log_prob(target_rewards).sum(dim=-1).mean()
    prior_dist = Normal(prior_mus, prior_stds)
    posterior_dist = Normal(posterior_mus, posterior_stds)
    
    kl_per_timestep = kl_divergence(posterior_dist, prior_dist).sum(dim=-1)
    raw_kl = kl_per_timestep.mean()

    free_nats_tensor = torch.tensor(free_nats, device=kl_per_timestep.device)
    kl_loss = torch.max(kl_per_timestep, free_nats_tensor).mean()
    
    if debug:
        print(f"    DEBUG - Recon Loss (Summed): {reconstruction_loss.item():.2f}")
        print(f"    DEBUG - Reward Loss: {reward_loss.item():.2f}")
        print(f"    DEBUG - Raw KL: {raw_kl.item():.2f}")

    return reconstruction_loss, reward_loss, kl_loss, raw_kl


def evaluate_model(rssm, action_model, env, action_dim, state_dim = 30, hidden_dim = 200, num_episodes=5, max_steps=1000):
    """
    Evaluate the trained models

    Args:
        rssm: Trained RSSM model
        env: Environment to test in
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        avg_return: Average return over episodes
    """
    episode_returns = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        # Convert from [H, W, C] to [C, H, W] for PyTorch CNN and normalize
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1).to(device) / 255.0
        
        episode_return = 0.0

        state = torch.zeros(1, state_dim, device=device)
        hidden = torch.zeros(1, hidden_dim, device=device)
        action = torch.zeros(1, action_dim, device=device)

        for _ in range(max_steps):
            with torch.no_grad():
                state, hidden = rssm.encode_one_step(obs_tensor, state, hidden, action)
                action = action_model(torch.cat((state, hidden), dim = -1))
                action_np = action.detach().cpu().numpy()
            action = np.clip(action_np, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            action_tensor = torch.tensor(action, dtype=torch.float32).to(device)
            action = action_tensor  # so next iteration's encode_one_step gets a tensor
            obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1).to(device) / 255.0


            episode_return += reward
            if terminated or truncated:
                break

        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}: Return = {episode_return:.2f}")

    avg_return = np.mean(episode_returns)

    return avg_return, episode_returns

def compute_action_value_loss(value_model, states, hiddens, state_values):
    # going to receive state values, states and hiddens of size [B, H, T], [B, H, T, D], and [B, H, T, L] respectively
    actor_loss = -torch.mean(torch.sum(state_values, dim = 1))
    value_preds = value_model(torch.cat((states[:, :-1], hiddens[:, :-1]), dim=-1)).squeeze(-1)
    value_loss = F.mse_loss(value_preds, state_values.detach())
    return actor_loss, value_loss

def imagine_trajectories(rssm : RSSM, action_model : Action, value_model: Value, prev_state, prev_hidden, lmbda, discount, horizon = 15):
    """
    Need to compute the state value by using the lambda construction

    This means that we have (1 - lambda)\sum_{n=1}^{H-1}\lambda^{n-1}V_{N}^{n}(s_{\tau})+\lambda^{H-1}V_{N}^{H}(s_{\tau})

    where we have that V_N^{k} = E(\sum_{n=\tau}^{h-1}discount^{n-\tau}r_{n}+discount^{h-\tau}v_{\gamma}(s_{h}))

    need to return the state value estimate here

    The process to do this --- first use the past state and the previous hidden

    The action model should return some action that it'd take given the latent state and the hidden
    """
    actions = []
    rewards = []
    # prev state is of size [B, T, D]
    B1, T1, D = prev_state.size()
    B2, T2, L = prev_hidden.size()

    prev_state = prev_state.detach().reshape(B1 * T1, D)
    prev_hidden = prev_hidden.detach().reshape(B2 * T2, L)

    states = [prev_state]
    hiddens = [prev_hidden]
    state_values = []

    for _ in range(horizon):
        action = action_model((torch.cat((prev_state, prev_hidden), dim = -1)))
        prev_state, prev_hidden, reward = rssm.imagine_one_step(prev_state, prev_hidden, action)
        states.append(prev_state)
        hiddens.append(prev_hidden)
        rewards.append(reward)
        actions.append(action)
    states = torch.stack(states, dim=1)
    hiddens = torch.stack(hiddens, dim=1)
    rewards = torch.stack(rewards, dim=1)
    actions = torch.stack(actions, dim=1)

    # states is a list composed of next generated states of sequences that are of size B, T, D. Total shape is 
    # [15, B, T , D]
    for i in range(horizon):
        # need to adjust the horizon based on the current value of tau
        state_value = calculate_state_value(value_model, rewards, lmbda, discount, states, hiddens, i, horizon)
        state_values.append(state_value)

    state_values = torch.stack(state_values, dim=1)

    return state_values, rewards, states, hiddens, actions
    
def calculate_state_value(value_model : Value, rewards, lmbda, discount, states, hiddens, tau, horizon):
    '''
    This function will run after a trajectory of a model has been run.

    Therefore, we need to pass in the full trajectory of rewards, states, and hiddens.

    We will be able to return the estimated state at each 

    We will make the assumption that we are calculaing from different starting points of tau

    Initially, we have that tau = t, but then with each new state, we make the assumption that this is no longer true.

    We should have that tau increases from tau = t to tau = t  + H

    Therefore, k should decrease from H to 1
    '''
    B, H, T = rewards.size()
    total_sum = torch.zeros(B, device=rewards[0].device)

    one_minus_lambda = 1 - lmbda
    for i in range(1, horizon - tau):
        # find the lambda exponentiated
        lambda_exp = lmbda ** (i - 1)

        # should use the last state and last hidden based on h
        est_state_val = calculate_beyond_k_state(value_model, rewards, discount, states, hiddens, i, tau, horizon)
        total_sum += lambda_exp * est_state_val.squeeze(-1)

    total_sum *= (1 - lmbda)
    total_sum += (lmbda ** (horizon - tau - 1)) * calculate_beyond_k_state(value_model, rewards, discount, states, hiddens, horizon - tau, tau, horizon)
    return total_sum

def calculate_beyond_k_state(value_model : Value, rewards, discount, states, hiddens, k, tau, horizon):
    # k is defined as how many states in the future we'd like to search
    h = min(tau + k, horizon)
    last_state, last_hidden = states[:, h], hiddens[:, h]
    value = value_model(torch.cat((last_state, last_hidden), dim = -1)).squeeze(-1)
    discount_t = discount ** (h - tau)
    summed_value = value * discount_t
    for i in range(tau, h):
        summed_value += discount ** (i - tau) * rewards[:, i].squeeze(-1)
    return summed_value

def collect_action_episodes(rssm, action_model, env, encoded_dim = 30, hidden_dim = 200, num_episodes=5, max_steps=1000, action_repeat=2, imagination_horizon = 15, exploration_noise=0.3):
    """
    Args:
        rssm: Trained RSSM model
        env: Environment
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        action_repeat: Action repeat parameter (R in paper)

    Returns:
        ExperienceBuffer with action generated data
    """
    action_dim = env.action_space.shape[0]
    buffer = ExperienceBuffer()

    print(f"\n{'='*60}")
    print(f"Imagination horizon: {imagination_horizon}")
    print(f"{'='*60}")

    total_steps = 0
    total_reward = 0.0

    for episode in range(num_episodes):
        obs_sequence = []
        action_sequence = []
        reward_sequence = []
        episode_reward = 0.0
        episode_steps = 0

        # Reset environment
        obs, info = env.reset()
        obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1).to(device) / 255.0
        obs_sequence.append(obs_tensor)

        state = torch.zeros(1, encoded_dim, device=device)
        hidden = torch.zeros(1, hidden_dim, device=device)
        action = torch.zeros(1, action_dim, device=device)

        for step in range(max_steps // action_repeat):
            with torch.no_grad():
                state, hidden = rssm.encode_one_step(obs_tensor, state, hidden, action)
                action = action_model(torch.cat((state, hidden), dim = -1))
                action_np = action.detach().cpu().numpy()
            noise = np.random.normal(0, exploration_noise, size=action_np.shape)
            action = action_np + noise
            action_clipped = np.clip(action, -1.0, 1.0)

            action_tensor = torch.tensor(action_clipped, dtype=torch.float32).to(device)
            for _ in range(action_repeat):
                action_sequence.append(action_tensor)
                obs, reward, terminated, truncated, info = env.step(action_clipped)
                obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).permute(2, 0, 1).to(device) / 255.0
                obs_sequence.append(obs_tensor)
                reward_sequence.append(torch.tensor(reward, dtype=torch.float32))

                episode_reward += reward
                episode_steps += 1

                # Progress indicator every 100 steps
                if (step + 1) % 100 == 0:
                    print(f"    Step {step + 1}/{max_steps} | Reward: {episode_reward:.2f}")

                if terminated or truncated:
                    reason = "terminated" if terminated else "truncated"
                    print(f"  Episode ended at step {episode_steps} ({reason})")
                    break
            action = action_tensor
        # Add episode to buffer (excluding last observation for action alignment)
        buffer.add_episode(obs_sequence[:-1], action_sequence, reward_sequence)

        total_steps += episode_steps
        total_reward += episode_reward
        avg_reward = total_reward / (episode + 1)
        avg_steps = total_steps / (episode + 1)

        print(f"     Episode {episode + 1} complete!")
        print(f"     Steps: {episode_steps} | Reward: {episode_reward:.3f}")
        print(f"     Running averages - Steps: {avg_steps:.1f} | Reward: {avg_reward:.3f}")

        # Progress update
        if (episode + 1) % 2 == 0:  # More frequent updates for CEM (every 2 episodes)
            print(f"\n PROGRESS UPDATE:")
            print(f"   Episodes completed: {episode + 1}/{num_episodes}")
            print(f"   Total timesteps collected: {len(buffer.observations)}")
            print(f"   Average episode length: {avg_steps:.1f} steps")
            print(f"   Average episode reward: {avg_reward:.3f}")

    print(f"\n{'='*60}")
    print(f"EPISODE COLLECTION COMPLETE!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total timesteps: {len(buffer.observations)}")
    print(f"Average episode length: {total_steps / num_episodes:.1f} steps")
    print(f"Average episode reward: {total_reward / num_episodes:.3f}")
    print(f"{'='*60}")

    return buffer

def train_rssm(S=5, B=32, L=50, num_epochs=100,
               evaluate_every=50, evaluation_episodes=3,
               plan_every=25, planning_episodes=3, action_repeat=2):
    """
    Main training loop for RSSM
    S: Number of random seed episodes
    B: Batch size
    L: Sequence length for training chunks
    """

    print("Initializing DMC walker environment...")
    dmc_env = create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0)
    env = DMCWrapper(dmc_env)
    environment_name = "DMC-walker-walk"

    # Environment specifications
    obs_shape = env.observation_space.shape 
    action_dim = env.action_space.shape[0]

    # RSSM and action / value model hyperparameters
    encoded_size = 1024
    latent_size = 30
    hidden_size = 200

    wandb.init(
        project="planet-rssm-dmc-walker",
        config={
            "S": S,
            "B": B,
            "L": L,
            "num_epochs": num_epochs,
            "encoded_size": encoded_size,
            "latent_size": latent_size,
            "hidden_size": hidden_size,
            "obs_shape": obs_shape,
            "action_dim": action_dim,
            "plan_every": plan_every,
            "planning_episodes": planning_episodes,
            "action_repeat": action_repeat,
            "environment": environment_name
        }
    )

    # Setup device (GPU if available, CPU otherwise)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")

    # Initialize RSSM
    rssm = RSSM(
        action_size=action_dim,
        latent_size=latent_size,
        encoded_size=encoded_size,
        hidden_size=hidden_size,
        min_std_dev=0.1,
        device=device
    ).to(device)

    # Initialize the Action and Value models
    action_model = Action(hidden_size + latent_size, action_dim).to(device)
    value_model = Value(hidden_size + latent_size).to(device)

    world_optimizer = optim.Adam(rssm.parameters(), lr=6e-4, eps=1e-4)
    value_optimizer = optim.Adam(value_model.parameters(), lr = 8e-5, eps=1e-4)
    action_optimizer = optim.Adam(action_model.parameters(), lr = 8e-5, eps=1e-4)

    # Collect initial dataset
    dataset = collect_random_episodes(env, S)

    # Log dataset statistics
    wandb.log({
        "dataset_size": len(dataset.observations),
        "num_episodes": S,
        "avg_episode_length": len(dataset.observations) / S if S > 0 else 0
    })

    # Training loop
    best_eval_return = float('-inf')
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print(f"\n Starting epoch {epoch}/{num_epochs}...")
            print(f"   Current dataset size: {len(dataset.observations)} timesteps")
        # Sample batch of sequences
        obs_batch, action_batch, reward_batch = dataset.get_random_sequences(B, L)

        if obs_batch is None:
            continue

        # Move data to device
        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)

        # Debug tensor shapes
        if epoch % 10 == 0:
            print(f"   Batch shapes - Obs: {obs_batch.shape}, Actions: {action_batch.shape}, Rewards: {reward_batch.shape}")

        world_optimizer.zero_grad()
        value_optimizer.zero_grad()
        action_optimizer.zero_grad()

        # Encode observations
        # obs_batch shape: [batch_size, seq_len, channels, height, width]
        batch_size, seq_len = obs_batch.shape[:2]
        obs_channels, obs_height, obs_width = obs_batch.shape[2:]

        # Only print batch info on milestone epochs to avoid spam
        if epoch % 10 == 0:
            print(f"   Processing batch: {batch_size} sequences x {seq_len} timesteps")
            print(f"   Observation shape: {obs_channels}x{obs_height}x{obs_width}")

        # Flatten for encoding: [batch_size * seq_len, channels, height, width]
        flat_obs = obs_batch.view(-1, obs_channels, obs_height, obs_width)
        encoded_obs = rssm.encode(flat_obs)

        # Reshape back: [batch_size, seq_len, encoded_dim]
        encoded_dim = encoded_obs.shape[-1]
        encoded_obs = encoded_obs.view(batch_size, seq_len, encoded_dim)

        encoded_obs_for_posterior = encoded_obs[:, 1:, :]
        obs_targets = obs_batch[:, 1:, :, :, :]
        action_batch_aligned = action_batch[:, :-1, :]
        reward_batch_aligned = reward_batch[:, :-1]

        effective_seq_len = seq_len - 1

        # Initialize states
        prev_state = torch.zeros(batch_size, latent_size, device=device)
        prev_hidden = torch.zeros(batch_size, hidden_size, device=device)

        rssm_output = rssm.pass_through(
            prev_state, 
            prev_hidden, 
            encoded_obs_for_posterior,
            action_batch_aligned
        )

        prior_states, posterior_states, hiddens, _, _, _, _, predicted_rewards = rssm_output
        reconstructed = []

        for t in range(effective_seq_len):
            decoded = rssm.decode(hiddens[:, t, :], posterior_states[:, t, :])
            reconstructed.append(decoded)

        reconstructed_obs = torch.stack(reconstructed, dim=1)

        reconstruction_loss, reward_loss, kl_loss, raw_kl = compute_losses(
            rssm_output, 
            reconstructed_obs, 
            obs_targets,
            predicted_rewards, 
            reward_batch_aligned,
            free_nats=3.0,
            debug=(epoch % 50 == 0)
        )

        world_model_loss = reconstruction_loss + 10 * reward_loss + kl_loss

        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(rssm.parameters(), max_norm=1000.0)  # Paper uses 1000
        world_optimizer.step()

        # Need to imagine trajectories here using the imagine_trajectories function. Will use the posterior states that were generated
        # posterior states are of size [B, T, D] where there are B batches, a sequence of length T, and posterior states possess dimensionality D
        state_values, rewards, states, imagined_hiddens, actions = imagine_trajectories(rssm, action_model, value_model, posterior_states.detach(), hiddens.detach(), lmbda = 0.95, discount = 0.99, horizon = 15)

        actor_loss, value_loss = compute_action_value_loss(
            value_model, states, imagined_hiddens, state_values
        )
        action_optimizer.zero_grad()
        value_optimizer.zero_grad()

        actor_loss.backward(retain_graph=True)
        value_loss.backward()

        action_optimizer.step()
        value_optimizer.step()

        wandb.log({
            "epoch": epoch,
            "world model loss": world_model_loss.item(),
            "actor model loss" : actor_loss.item(),
            "value model loss" : value_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item(),
            "raw_kl": raw_kl.item(),  # Add this line
        })

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: World Model Loss: {world_model_loss.item():.4f}, Actor Model Loss: {actor_loss.item():.4f}, Value Model Loss : {value_loss.item():.4f}"
                  f"Reconstruction: {reconstruction_loss.item():.4f}, "
                  f"Reward: {reward_loss.item():.4f}, "
                  f"KL: {kl_loss.item():.4f}")

        # Quick progress indicator for non-milestone epochs
        elif epoch % 5 == 0:
            print(f"   Epoch {epoch}: World Loss={world_model_loss.item():.4f}, Actor Loss={actor_loss.item():.4f}, Value Loss={value_loss.item():.4f}")

        if epoch % evaluate_every == 0 and epoch > 0:
            print(f"\n=== Evaluating Controller at Epoch {epoch} ===")
            rssm.eval()  # Set to evaluation mode

            avg_return, episode_returns = evaluate_model(
                rssm, action_model, env, action_dim, num_episodes=evaluation_episodes, max_steps=1000
            )

            # Log evaluation results to wandb
            wandb.log({
                "eval_avg_return": avg_return,
                "eval_std_return": np.std(episode_returns),
                "eval_epoch": epoch
            })

            # Save checkpoint if this is the best performance so far
            if avg_return > best_eval_return:
                best_eval_return = avg_return
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': rssm.state_dict(),
                    'best_eval_return': best_eval_return,
                    'avg_return': avg_return,
                    'episode_returns': episode_returns
                }, best_checkpoint_path)
                print(f"New best model saved! Return: {avg_return:.2f}")

            # Save periodic checkpoint
            if epoch % (evaluate_every * 2) == 0:  # Every 2nd evaluation
                periodic_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': rssm.state_dict(),
                    'best_eval_return': best_eval_return,
                    'avg_return': avg_return,
                    'episode_returns': episode_returns
                }, periodic_checkpoint_path)
                print(f"Periodic checkpoint saved: epoch_{epoch}.pth")

            rssm.train()  # Back to training mode
            print("=== Evaluation Complete ===\n")

        # CEM Planning and dataset augmentation
        if epoch % plan_every == 0 and epoch > 0:
            print(f"\n=== Planning and Data Collection at Epoch {epoch} ===")
            rssm.eval()  # Set to evaluation mode for planning

            # Collect new data using
            action_buffer = collect_action_episodes(
                rssm, action_model, env, encoded_dim = latent_size, hidden_dim = hidden_size, num_episodes=planning_episodes,
                max_steps=1000, action_repeat=action_repeat
            )

            dataset.merge_buffer(action_buffer)

            # Log dataset statistics
            wandb.log({
                "dataset_size_after_planning": len(dataset.observations),
                "cem_episodes_added": planning_episodes,
                "planning_epoch": epoch
            })

            rssm.train()  # Back to training mode
            print(f"=== Planning Complete. Dataset now has {len(dataset.observations)} timesteps ===\n")

    # Save model and log as wandb artifact
    model_path = 'trained_rssm_dmc_walker.pth'
    torch.save(rssm.state_dict(), model_path)

    # Create wandb artifact for model
    artifact = wandb.Artifact("rssm_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    env.close()
    wandb.finish()
    return rssm

if __name__ == "__main__":
    # Train the model with CEM evaluation and planning
    trained_rssm = train_rssm(
        S=5, B=50, L=50,
        num_epochs= 100000,
        evaluate_every=500,
        evaluation_episodes=1,
        plan_every=100,  # CEM planning every 25 epochs
        planning_episodes=1,
        action_repeat=2  # Action repeat parameter R
    )

    print("Training complete! Model saved as 'trained_rssm_dmc_walker.pth'")

    # Final evaluation using DMC walker
    print("\n=== Final Evaluation ===")
    final_dmc_env = create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0)
    final_eval_env = DMCWrapper(final_dmc_env)

    final_avg_return, _ = evaluate_controller(trained_rssm, final_eval_env, num_episodes=10)
    final_eval_env.close()
    print(f"Final Average Return: {final_avg_return:.2f}")