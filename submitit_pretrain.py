# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import slurm_main_pretrain as trainer
import submitit


def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")

    parser.add_argument('--exclude', default=None, type=str, help='nodes we dont want')
    parser.add_argument('--nodelist', default=None, type=str, help='nodes to use')

    parser.add_argument("--account")
    parser.add_argument("--job_name")
    parser.add_argument("--output")
    parser.add_argument("--error")

    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/burg/zgroup/users/gzg2104/checkpoint/").is_dir():
        p = Path(f"/burg/zgroup/users/gzg2104/checkpoint/{user}/experiments")
        print(p)
        os.makedirs(str(p), exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import slurm_main_pretrain as trainer

        self._setup_gpu_args()
        trainer.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=48*num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=None,
        slurm_signal_delay_s=120,
        slurm_account=args.account,
        slurm_job_name=args.job_name,
        slurm_exclude=args.exclude,
        slurm_nodelist=args.nodelist,
        **kwargs
    )

    executor.update_parameters(name="mae")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    args.resume = find_most_recent_checkpoint(args.resume)

    trainer = Trainer(args)
    job = executor.submit(trainer)

    # print("Submitted job_id:", job.job_id)
    print(job.job_id)

def get_epoch_num(checkpoint_filename):
    assert checkpoint_filename.startswith('checkpoint-'), f"{checkpoint_filename}"
    assert checkpoint_filename.endswith('.pth'), f"{checkpoint_filename}"
    curr_epoch_num = int(checkpoint_filename.split('-')[1][:-4])

    return curr_epoch_num
    
def find_most_recent_checkpoint(curr_checkpoint):
    print(f"Starting checkpoint: {curr_checkpoint}")
    checkpoint_dir = os.path.dirname(curr_checkpoint)

    highest_epoch_num = get_epoch_num(os.path.basename(curr_checkpoint))
    print(f"Curr epoch num: {highest_epoch_num}")
    highest_checkpoint = curr_checkpoint

    for item in os.listdir(checkpoint_dir):
        if 'checkpoint-' in item and '.pth' in item:
            curr_epoch_num = get_epoch_num(item)
            if curr_epoch_num > highest_epoch_num:
                highest_epoch_num = curr_epoch_num
                highest_checkpoint = os.path.join(checkpoint_dir, item)
    
    print(f"New checkpoint: {highest_checkpoint}")

    return highest_checkpoint

if __name__ == "__main__":
    main()
