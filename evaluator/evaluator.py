from typing import Type
import numpy as np
from loguru import logger
import timeout_decorator

from l2r.envs.env import RacingEnv

from config import SubmissionConfig, EnvConfig, SimulatorConfig


class Learn2RaceEvaluator:
    """Evaluator class which consists of a 1-hour pre-evaluation phase followed by an evaluation phase."""

    def __init__(
        self,
        submission_config: Type[SubmissionConfig],
        env_config: Type[EnvConfig],
        sim_config: Type[SimulatorConfig],
    ):
        logger.info("Starting learn to race evaluator")
        self.submission_config = submission_config
        self.env_config = env_config
        self.sim_config = sim_config

        self.agent = None
        self.env = None

        self.metrics = dict()
        # Metrics that are summed internally by the env tracker
        self.metrics_to_replace = [
            "laps_completed",  # Global counter
            "num_infractions",  # Global counter
            "success_rate",  # Resets after a lap
            "pct_complete",  # Resets after a lap
        ]
        self.infractions_till_last_lap = 0

        # Metrics that are reset after episode but relevant for the lap
        self.metrics_to_add = ["total_time", "total_distance"]

        # Metrics that are averaged over episode
        self.metrics_to_append = [
            "average_speed_kph",
            "average_displacement_error",
            "trajectory_efficiency",
            "trajectory_admissibility",
            "movement_smoothness",
            "timestep/sec",
            "reward",
        ]

        self.laps_completed = 0

    def init_agent(self):
        """ """
        self.agent = self.submission_config.agent()

    def load_agent_model(self, path):
        self.agent.load_model(path)

    def save_agent_model(self, path):
        self.agent.save_model(path)

    @timeout_decorator.timeout(1 * 60 * 60)
    def train(self):
        logger.info("Starting one-hour 'practice' phase")
        self.agent.training(self.env)

    def evaluate(self):
        """Evaluate the episodes."""
        logger.info("Starting evaluation")
        episode_count = 0

        while self.laps_completed < self.env_config.n_eval_laps:
            state, _ = self.env.reset()
            self.agent.register_reset(state)

            done = False
            info = {}
            cumulative_reward = 0

            while not done:
                action = self.agent.select_action(state)
                state, reward, done, info = self.env.step(action)
                cumulative_reward += reward

            info["metrics"]["reward"] = cumulative_reward
            self.register_metrics(metrics=info["metrics"])
            self.display_metrics()

            episode_count += 1
            logger.info(
                f"Completed episode: {episode_count} with metrics: {self.metrics}"
            )

        return self.metrics

    def register_metrics(self, metrics):
        lap_completed = self.laps_completed != metrics["laps_completed"]
        idx = self.laps_completed

        # Check if current lap is part of metrics
        if idx not in self.metrics:
            self.metrics[idx] = {}

        # If the lap is completed, record lap stats
        if lap_completed:
            self.laps_completed = metrics["laps_completed"]
            # Record the infractions for the completed lap
            self.metrics[idx]["num_infractions"] = (
                metrics["num_infractions"] - self.infractions_till_last_lap
            )
            self.metrics[idx]["pct_complete"] = 100

            # Start recording metrics for the next lap
            idx = self.laps_completed
            self.metrics[idx] = {}
            self.infractions_till_last_lap = metrics["num_infractions"]

        for metric_name in self.metrics_to_add:
            if metric_name in self.metrics[idx]:
                self.metrics[idx][metric_name] += metrics[metric_name]
            else:
                self.metrics[idx][metric_name] = metrics[metric_name]

        for metric in self.metrics_to_append:
            if metric in self.metrics[idx]:
                self.metrics[idx][metric].append(metrics[metric])
            else:
                self.metrics[idx][metric] = [metrics[metric]]

        for metric in self.metrics_to_replace:
            self.metrics[idx][metric] = metrics[metric]

        # Subtract infractions from previous laps
        self.metrics[idx]["num_infractions"] -= self.infractions_till_last_lap

    def display_metrics(self):
        for lap, metrics in self.metrics.items():
            print("=" * 10)
            print(f"LAP #{lap+1}")
            print("=" * 10)

            for key in self.metrics_to_replace:
                print(key, metrics[key])

            for key in self.metrics_to_add:
                print(key, metrics[key])

            for key in self.metrics_to_append:
                print(key, round(np.mean(metrics[key]), 3))

    def create_env(self):
        """Your configuration yaml file must contain the keys below."""
        self.env = RacingEnv(self.env_config.__dict__, self.sim_config.__dict__)
        self.env.make()
