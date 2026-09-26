# Accounts and failover

An account is one set of credentials on one provider. Its ID is `PROVIDER:USER`, such as `kaggle:alice`. Accounts are kept in order of preference, and the first is the default for new jobs:

```bash
compute-runner account add kaggle alice                   # Kaggle's standard credentials (~/.kaggle)
compute-runner account add kaggle bob --enter-key         # type bob's API key or access token
compute-runner account add kaggle carol --credentials ~/Downloads/kaggle.json --gpu-limit 1
compute-runner account add kaggle bob --default          # prefer bob from now on
compute-runner account list
compute-runner account remove kaggle:bob
```

Every account checks that its credentials authenticate as its user, so a global token cannot act for another account. An account cannot be removed while unfinished jobs, or pending downloads, use it. Failed downloads of its runs are not retried while it is removed and resume if it is added again.

SSH machines are accounts too; see [SSH machines](ssh.md).

The worker reads accounts and settings when it starts. After `account add`, `account remove`, or `init`, apply the change with `compute-runner service restart`; these commands print that reminder while a worker is running. Until then, a job submitted to a newly added account waits with a reason saying the running worker does not know its account.

Each job records its account, and each attempt records the account, remote reference, and URL it ran on. Choose an account with `--account` on `submit` and `retry`, or move a job that has not been submitted yet:

```bash
compute-runner submit train.py --gpu --account kaggle:bob
compute-runner move JOB_ID --account kaggle:alice
```

When a waiting job cannot start on its account, because its slots are busy, its GPU quota is exhausted, or the account cannot be checked, the failover policy decides what happens:

| Policy | Behavior |
| --- | --- |
| `off` | The job waits on its account |
| `ask` (default) | The job waits and shows `suggested_account`: the first other account, in preference order, that can start it now. An agent asks the user before moving it |
| `auto` | The worker moves the job to that account. Its `account` changes, and its event history records `Moved from ACCOUNT: reason` |

```bash
compute-runner init --failover auto
compute-runner service restart
```

Failover counts jobs already preparing on the other account, so a burst moves only as many jobs as that account can start. A job whose inputs are already uploading stays on its account, and jobs queued behind it wait for it rather than failing over. An account that rejected one of a job's launches is not chosen for that job again, so a job cannot bounce between two full accounts. Moving a job uploads its inputs again, because private datasets belong to one account. Failover checks that the other account can read the job's provider datasets; see [Datasets across accounts](#datasets-across-accounts). Runs that have started never move; continue a stopped resumable run on another account with `compute-runner continue JOB_ID --account ID`, as described in [Optional resumable training](workloads.md#optional-resumable-training).

## Credentials

Every account's secrets live in one file, `~/.config/compute-runner/credentials.json`, beside the configuration. `account add` writes it: `--enter-key` and `--enter-password` ask for the secret without showing it, `--credentials` and `--password-file` read it from a file you downloaded, and `--key` records the path of an SSH private key. You can also edit the file yourself:

```json
{
  "kaggle:alice": {"username": "alice", "key": "0123abcd..."},
  "kaggle:bob": {"token": "KGAT_..."},
  "ssh:lab": {"key": "~/.ssh/id_ed25519", "passphrase": "only if the key has one"},
  "ssh:lab-cpu": {"password": "..."}
}
```

The file is created readable by you alone and never leaves this machine: it is not uploaded, stored with jobs, logged, or shown in any response, and `account remove` deletes the account's entry. An account without an entry uses its provider's defaults: Kaggle's `~/.kaggle` and `KAGGLE_*` variables, or ssh-agent and the keys in `~/.ssh`. Configurations from earlier versions that name a credentials, key or password file keep working until `account add` saves new secrets for the account. `compute-runner doctor --offline` shows where each account's login comes from, without the secrets.

## Datasets across accounts

A job can attach existing provider datasets, such as another account's private dataset, and the account that runs it may not be able to read them. Before attaching a dataset, the worker asks the job's account whether it can read it, then:

| Situation | Behavior |
| --- | --- |
| The job's account can read it | Attach it directly, pinned to its current version |
| Another connected account can read it, and copies are allowed | Download it once through that account, then upload the copy to the job's account like a local input |
| Another connected account can read it, and copies are not allowed | Block the job; the error names the account that can read it |
| No connected account can read it, or that version does not exist | Block the job before it runs; the error asks to check the reference or the accounts' access |

Copies are off by default because they place the data in another account and take its storage. Allow them for one job with `compute-runner agent move JOB_ID --account ACCOUNT --transfer` (the job's own account also works), or for every job with `compute-runner init --transfer`. Only aliased inputs (`inputs: {data: "kaggle:owner/slug"}`) can be copied, because the workload finds a copy through `KGR_INPUT_DATA`; the unaliased `datasets` list cannot. A copied dataset version is cached in the state directory and reused. Check the dataset's license before copying it.

Failover prefers an account that can read every dataset. Under `ask`, a suggested account that would need a copy is shown with `suggested_transfer: true`; under `auto`, the worker moves a job there only when copies are allowed.
