# SSH machines

A machine you can log in to over SSH runs jobs like a Kaggle account: the same commands, responses, run folders and failover. Its account ID is `ssh:NAME`, where you choose the name:

```bash
compute-runner account add ssh lab --host 10.0.0.5 --login alice --key ~/.ssh/id_ed25519 --trust-new-host
compute-runner account add ssh lab-cpu --host lab.example.org --port 2222 --login alice --enter-password
compute-runner account add ssh lab --gpu-limit 2        # the machine has two GPUs jobs may use
compute-runner doctor                                   # Python version, GPUs and runs on each machine
```

The key or password goes to the [credentials file](accounts.md#credentials); without either, the login uses ssh-agent and the default keys in `~/.ssh`. The machine's host key must be known: `--trust-new-host` reads it without logging in, accepts a host that is not in `known_hosts` yet and prints its fingerprint, and a host key that changes later is refused until you remove the old one. Compare the printed fingerprint with the one the machine shows (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`). `~/.ssh/config` is not read, so give the host, port and login on the account.

Each job runs as your login, in its own folder under the work directory on the machine (`~/.compute-runner`, or `--workdir`: a folder below your home folder, without `..`), started by a supervisor that keeps running when the connection or the worker stops. The supervisor enforces `timeout_seconds` and records how the run ended. A timeout or `cancel` stops every process the run started, including a notebook's kernel, with SIGTERM and then SIGKILL 20 seconds later; so does a workload that exits and leaves processes behind. A GPU is handed to the next job only once they are gone. Source and input bundles are uploaded once, verified, and kept read-only for later runs. A `requirements` file installs into a virtual environment of that run, which can still use packages installed on the machine. Notebooks run with `nbconvert` from the machine's Python, and the executed notebook is downloaded with the outputs.

What differs from Kaggle:

| | SSH machine |
| --- | --- |
| Network | Cannot be blocked, so SSH jobs must set `internet: true`; jobs without it are refused and never fail over to an SSH machine |
| GPUs | `gpu: true` only, no accelerator IDs. GPU slots are the account's `--gpu-limit` (0 unless set), each GPU job gets its own device through `CUDA_VISIBLE_DEVICES`, and CPU jobs see no GPU. There is no GPU time quota |
| Data on the machine | `inputs: {data: "ssh:/data/imagenet"}` attaches a folder already on the machine where it is, without uploading it. A single file arrives as a folder holding it, as a copy does. Submission records the machine, as `ssh:lab:/data/imagenet`, so the reference keeps meaning that machine: on another account, including another SSH machine with the same path, the folder is used only as a copy (see [Datasets across accounts](accounts.md#datasets-across-accounts)), which is made again when its files change. A job on a Kaggle account names the machine itself: `ssh:lab:/data/imagenet` |
| Kaggle datasets | Copied through a connected Kaggle account when copies are allowed |
| Capacity | `--cpu-limit` and `--gpu-limit` count only this runner's jobs, not other work on the machine |
| Cleanup | The runner writes only inside its work directory and never deletes run folders, bundles or your data. To reclaim the space when no jobs use it, remove the work directory yourself; bundles are read-only, so `chmod -R u+w ~/.compute-runner && rm -rf ~/.compute-runner` |
