#!/usr/bin/env python3


from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context

import rospy
from openai_ros.task_envs.deepleng import deepleng_docking


class RlpytPpo():

    def __init__(self):
        self.env_id = "DeeplengDocking-v1"
        self.run_ID = 0
        self.cuda_idx = None

    def __call__(self, *args, **kwargs):

        sampler = SerialSampler(
                EnvCls=gym_make,
                env_kwargs=dict(id=self.env_id),
                CollectorCls=CpuResetCollector,
                batch_T=2048,
                batch_B=1,
                max_decorrelation_steps=0,

            )
        algo = PPO(discount=0.99,
                   learning_rate=3e-4,
                   clip_grad_norm=1e6,
                   entropy_loss_coeff=0.0,
                   gae_lambda=0.95,
                   minibatches=32,
                   epochs=10,
                   ratio_clip=0.2,
                   normalize_advantage=True,
                   linear_lr_schedule=True,
                   )  # Run with defaults.

        agent = MujocoFfAgent()
        # agent = MujocoFfAgent(model_kwargs=dict())

        runner = MinibatchRl(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=1e6,
            log_interval_steps=50,
            affinity=dict(cuda_idx=self.cuda_idx),
        )

        config = dict(env_id=self.env_id)
        name = "ppo_" + self.env_id
        log_dir = "/home/dfki.uni-bremen.de/mpatil/Documents/rlpyt_logs"
        with logger_context(log_dir, run_ID, name, config, use_summary_writer=True, snapshot_mode="last"):
            runner.train()

def main():
    rospy.init_node('RlpytPpo_docker', anonymous=True)
    train = RlpytPpo()
    train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main()
