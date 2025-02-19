import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import argparse
import os


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite.
    #                                       Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    task_start_id: int = 0                          # Start task ID
    task_end_id: int = 0                            # End task ID
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    root_data_dir = "/home/daiyp/openvla-mini/runs"
    
    if cfg.pretrained_checkpoint.endswith(".pt"):
        model_name = cfg.pretrained_checkpoint.split("/")[-3]
    else:
        model_name = cfg.pretrained_checkpoint.split("/")[-1]
    
    save_dir = os.path.join(root_data_dir, cfg.task_suite_name, model_name)
    os.makedirs(save_dir, exist_ok=True)
    

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    print(f"Task suite: {cfg.task_suite_name} has total task number: {num_tasks_in_suite}, run on task from {cfg.task_start_id} to {cfg.task_end_id}")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    for task_id in tqdm.tqdm(range(cfg.task_start_id, cfg.task_end_id)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=resize_size)
        
        results = {"task_id": task_id, "task_description":task_description, "data": []} 
        
        for episode_idx in range(cfg.num_trials_per_task):
            
            # Reset environment
            env.reset()
            
            success = False

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"\nTask: {task_description}, Episode: {episode_idx+1}, Task ID: {task_id}")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size, flip_twice=cfg.model_family == "openvla")
                    
                    

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # use_wrist_image
                    if cfg.use_wrist_image:
                        wrist_img = get_libero_image(obs, resize_size, key="robot0_eye_in_hand_image", flip_twice=cfg.model_family == "openvla")
                        replay_wrist_images.append(wrist_img)

                    # buffering #obs_history images, optionally
                    image_history = replay_images[-cfg.obs_history :]
                    if len(image_history) < cfg.obs_history:
                        image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                    # same but for optional wrist images
                    if cfg.use_wrist_image:
                        wrist_image_history = replay_wrist_images[-cfg.obs_history :]
                        if len(wrist_image_history) < cfg.obs_history:
                            wrist_image_history.extend(
                                [replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_image_history))
                            )
                        # interleaved images [... image_t, wrist_t ...]
                        image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    
                    observation = {
                        "full_image": image_history,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )
                    

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family in ["openvla", "prismatic"]:
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        success = True
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    break
            
            if episode_idx<2:
                save_rollout_video(
                    replay_images, episode_idx, success=success, task_description=task_description, rollout_dir=os.path.join(save_dir,f"seed{cfg.seed}")
                )
            print(f"\nTask: {task_description}, Episode: {episode_idx+1}, Success: {success}")
            
            results["data"].append({"episode": episode_idx, "success": success})
        
        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
        json_name = f"task{task_id}-seed{cfg.seed}-{processed_task_description}.json"
        with open(os.path.join(save_dir, json_name), "w") as f:
            json.dump(results, f, indent=2)

    env.close()

if __name__ == "__main__":
    eval_libero()
    
    # python experiments/robot/libero/run_libero_eval_batch.py --pretrained_checkpoint "/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/VLA/minivla-wrist-vq-libero90-prismatic/checkpoints/step-110000-epoch-24-loss=0.3550.pt" --model_family prismatic --task_suite_name libero_90 --use_wrist_image True --task_start_id 0 --task_end_id 10
    
    # python experiments/robot/libero/run_libero_eval_batch.py --pretrained_checkpoint "/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/VLA/openvla-7b-finetuned-libero-spatial" --model_family openvla --task_suite_name libero_spatial --task_start_id 0 --task_end_id 1