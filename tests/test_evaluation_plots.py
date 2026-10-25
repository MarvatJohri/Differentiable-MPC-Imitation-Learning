"""Run from mj_work: python -m unittest discover -s tests -v."""
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from configs import SpacecraftEnvConfig
import simulation_env_jax as original
from utils import generate_trajs as rollouts
from utils import plotting_and_evaluate as plots


class DummyNetwork(eqx.Module):
    decomposition_type: str = eqx.field(static=True, default="diagonal")

    def __call__(self, obs):
        return jnp.zeros(9, dtype=obs.dtype)


class DummyController(eqx.Module):
    network: DummyNetwork
    nx: int = eqx.field(static=True, default=7)
    nu: int = eqx.field(static=True, default=3)
    qr_output_horizon: int = eqx.field(static=True, default=1)

    def __call__(self, obs, goal, nominal_states, nominal_controls):
        action = jnp.array([5e-6, -2e-6, 1e-6], dtype=obs.dtype)
        return action, nominal_states, jnp.broadcast_to(action, nominal_controls.shape)


class FixedController(eqx.Module):
    Q: jax.Array
    R: jax.Array

    def __call__(self, obs, goal, nominal_states, nominal_controls):
        return jnp.zeros(3), nominal_states, nominal_controls


def policy(obs):
    return jnp.array([0.1, -0.04, 0.02], dtype=obs.dtype)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = SpacecraftEnvConfig()
        cls.env = original.SpacecraftEnvJax(
            dynamics_params=cfg.spacecraft_dynamics_parameters,
            max_ep_steps=4,
        )
        cls.controller = DummyController(DummyNetwork())

    def tearDown(self):
        plt.close("all")

    def make_data(self, *, horizon=False):
        obs = np.zeros((2, 5, 7))
        obs[..., 0] = np.cos(np.pi / 4)
        obs[..., 1] = np.sin(np.pi / 4)
        obs[..., 4] = np.deg2rad(2)
        actions = np.zeros((2, 4, 3))
        actions[..., 0] = 2  # Clipped to one by the environment.
        q = np.broadcast_to(np.diag(np.arange(1, 7)), (2, 5, 6, 6))
        r = np.broadcast_to(np.eye(3) * 6.4 / (5e-5)**2, (2, 5, 3, 3))
        if horizon:
            q = np.repeat(q[:, :, None], 3, axis=2)
            r = np.repeat(r[:, :, None], 3, axis=2)
        trajectory = rollouts.Trajectory(
            obs, np.zeros((2, 7)), actions, np.ones((2, 4)), q, r,
        )
        return trajectory, plots.extract_trajectory_data(
            trajectory, action_units="normalized", dt=0.1, max_torque=5e-5,
        )

    def test_extraction_units_alignment_and_control_metrics(self):
        trajectory, data = self.make_data()
        np.testing.assert_allclose(data["angle_error"], 90)
        np.testing.assert_allclose(data["omega_error"], 2)
        np.testing.assert_allclose(data["requested_torque"][..., 0], 1e-4)
        np.testing.assert_allclose(data["applied_torque"][..., 0], 5e-5)
        np.testing.assert_allclose(data["reward_time"], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(data["returns"], 4)
        metrics = plots.compute_control_metrics({"PPO": data})
        np.testing.assert_allclose(metrics["Integrated squared torque"], 4 * 0.1 * (5e-5)**2)
        np.testing.assert_allclose(metrics["Clipping fraction"], 1)
        single = rollouts.Trajectory(*(field[0] for field in trajectory))
        single_data = plots.extract_trajectory_data(single, action_units="normalized")
        self.assertEqual(single_data["angle_error"].shape, (1, 5))
        with self.assertRaises(ValueError):
            plots.extract_trajectory_data(trajectory._replace(rewards=np.ones((2, 5))), action_units="torque")

    def test_plots_and_saved_figures(self):
        _, data = self.make_data()
        runs = {"PPO": data, "MPC": data}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "errors.png"
            fig = plots.plot_error_trajectories(runs, save_path=path)
            self.assertTrue(path.is_file())
            self.assertEqual(len(fig.axes[0].lines[0].get_xdata()), 5)
            fig = plots.plot_control_trajectories(runs, episode=0)
            self.assertEqual(len(fig.axes[0].lines[0].get_xdata()), 4)
            plots.plot_control_trajectories(runs, episode=None, requested=True)
            fig = plots.plot_rewards(runs)
            np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [1, 2, 3, 4])
            plots.plot_rewards(runs, episode=0)
            plots.plot_episode_returns(runs)
            fig = plots.plot_cost_weights(runs)
            np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), 6.4)
            plots.plot_cost_weights(runs, view="prediction_horizon")
            _, horizon_data = self.make_data(horizon=True)
            fig = plots.plot_cost_weights({"MPC": horizon_data}, horizon_index=2)
            self.assertEqual(len(fig.axes[0].lines[0].get_xdata()), 5)
            fig = plots.plot_cost_weights({"MPC": horizon_data}, view="prediction_horizon", timestep=3)
            self.assertEqual(len(fig.axes[0].lines[0].get_xdata()), 3)
        with self.assertRaises(ValueError):
            plots.plot_cost_weights({"PPO": {"returns": np.ones(2)}})

    def test_training_metric_steps_and_missing_episode_returns(self):
        history = {"episode_return": np.array([np.nan, 2, 4]), "approx_kl": np.array([0.1, 0.2, 0.3])}
        fig = plots.plot_training_metrics(
            {"ACMPC": history}, steps_per_update={"ACMPC": 4500},
            start_steps=9000, smoothing_window=2,
        )
        line = fig.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(), [13500, 18000, 22500])
        np.testing.assert_allclose(line.get_ydata(), [np.nan, 2, 3], equal_nan=True)

    def test_controller_rollout_matches_original_and_qr_flag(self):
        key = jax.random.PRNGKey(42)
        for frequency in (1, 2):
            with self.subTest(replan_freq=frequency):
                old, old_key = original.n_rollouts_controller(
                    self.env, self.controller, key, horizon=2, max_ep_steps=4,
                    n_rollouts=2, replan_freq=frequency,
                )
                new, new_key = rollouts.n_rollouts_controller(
                    self.env, self.controller, key, horizon=2, max_ep_steps=4,
                    n_rollouts=2, replan_freq=frequency, record_qr=True,
                )
                np.testing.assert_allclose(new.observations, old[0], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(new.actions, old[2][:, :-1], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(new.Q, old[3])
                np.testing.assert_allclose(new.R, old[4])
                np.testing.assert_array_equal(new_key, old_key)
                self.assertEqual(new.rewards.shape, (2, 4))
        fixed = FixedController(jnp.eye(6), jnp.eye(3))
        for flag in (False, True):
            recorded, _ = rollouts.n_rollouts_controller(
                self.env, fixed, key, horizon=2, max_ep_steps=4, n_rollouts=2,
                record_qr=flag,
            )
            self.assertEqual(recorded.actions.shape, (2, 4, 3))
            self.assertEqual(recorded.Q is None, not flag)
            if flag:
                np.testing.assert_allclose(recorded.Q, np.broadcast_to(np.eye(6), (2, 5, 6, 6)))

    def test_policy_and_dagger_match_original(self):
        key = jax.random.PRNGKey(17)
        old, old_key = original.n_rollouts_policy(self.env, policy, key, 4, 2)
        new, new_key = rollouts.n_rollouts_policy(self.env, policy, key, 4, 2)
        np.testing.assert_allclose(new.observations, old[0])
        np.testing.assert_allclose(new.actions, old[2][:, :-1])
        np.testing.assert_array_equal(new_key, old_key)
        self.assertEqual(new.rewards.shape, (2, 4))
        self.assertIsNone(new.Q)
        old, old_key = original.generate_n_trajectories(self.env, self.controller, policy, key, 0.5, 4, 2)
        new, new_key = rollouts.generate_n_dagger_trajectories(self.env, self.controller, policy, key, 0.5, 4, 2)
        for a, b in zip(old, new):
            np.testing.assert_allclose(a, b)
        np.testing.assert_array_equal(new_key, old_key)


if __name__ == "__main__":
    unittest.main()
