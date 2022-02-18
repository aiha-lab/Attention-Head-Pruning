# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/multiproc.py
import sys
import subprocess
import signal
import os
import time
from argparse import ArgumentParser, REMAINDER


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--nnodes",
        type=int, default=1, help="The number of nodes to use for distributed training.",
    )
    parser.add_argument(
        "--node_rank",
        type=int, default=0, help="The rank of the node for multi-node distributed training.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int, default=1, help="The number of processes to launch on each node, for GPU training, "
                                  "this is recommended to be set to the number of GPUs in your system "
                                  "so that each process can be bound to a single GPU.",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1", type=str, help="Master node (rank 0)'s address, should be either "
                                            "the IP address or the hostname of node 0, for "
                                            "single node multi-proc training, the --master_addr can be 127.0.0.1",
    )
    parser.add_argument(
        "--master_port",
        default=29500, type=int, help="Master node (rank 0)'s free port that needs to "
                                      "be used for communication during distributed training.",
    )

    parser.add_argument(
        "--stdout_all", action="store_true", help="Whether to print all outputs in console."
    )

    # positional
    parser.add_argument(
        "training_script", type=str, help="The full path to the single GPU training "
                                          "program/script to be launched in parallel, "
                                          "followed by all the arguments for the training script.",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------------------- #
    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    # -------------------------------------------------------------------------------- #
    # override --gpus flag in args.training_script_args
    if "--gpus" in args.training_script_args:
        gpus_index = args.training_script_args.index("--gpus")
        args.training_script_args[gpus_index + 1] = str(args.nproc_per_node)
    else:
        args.training_script_args = ["--gpus", str(args.nproc_per_node)] + args.training_script_args

    # -------------------------------------------------------------------------------- #
    processes = []
    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)  # maybe unused?
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [sys.executable, "-u", args.training_script] + args.training_script_args

        print("-" * 72)
        print(f"Initializing distributed process: (LOCAL RANK: {local_rank} / WORLD SIZE: {dist_world_size})")
        print(cmd)

        if not args.stdout_all:
            stdout = (None if local_rank == 0 else open("GPU_" + str(local_rank) + ".log", "w"))  # when to be closed?
        else:
            stdout = None
        # stdout = open("GPU_" + str(local_rank) + ".log", "w")

        process = subprocess.Popen(cmd, env=current_env, stdout=stdout)
        processes.append(process)

    # -------------------------------------------------------------------------------- #
    # TODO match recently introduced PyTorch/distributed/run
    print("-" * 72)
    try:
        up = True
        error = False
        while up and not error:
            up = False
            for p in processes:
                ret = p.poll()
                if ret is None:
                    up = True
                elif ret != 0:
                    error = True
            time.sleep(1)

        if error:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            exit(1)

    except KeyboardInterrupt:
        print("-" * 72)
        # try:
        #     for p_i, p in enumerate(processes):
        #         print(f"KeyboardInterrupt detected: ({p_i} / WORLD SIZE: {dist_world_size})")
        #         p.send_signal(sig=signal.CTRL_C_EVENT)
        #         # p.terminate()
        # except (KeyboardInterrupt, SystemExit):
        #     for p_i, p in enumerate(processes):
        #         p.terminate()
        # raise
        for p_i, p in enumerate(processes):
            print(f"KeyboardInterrupt detected: ({p_i} / WORLD SIZE: {dist_world_size})")
            p.terminate()
        raise
    except SystemExit:
        for p_i, p in enumerate(processes):
            print(f"SystemExit detected: ({p_i} / WORLD SIZE: {dist_world_size})")
            p.terminate()
        raise
    except:
        for p_i, p in enumerate(processes):
            print(f"Other interrupt detected: ({p_i} / WORLD SIZE: {dist_world_size})")
            p.terminate()
        raise


if __name__ == "__main__":
    main()
