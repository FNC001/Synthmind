# SynPred AutoDL 24h Autorun

These scripts prepare and run the SynPred 24-hour optimization/reporting cycle on an AutoDL/SeetaCloud host.

Do not put passwords in these files, command history, logs, reports, or Markdown. Use the SSH password prompt, an SSH key, or a transient local mechanism that does not write secrets to disk.

## Usage

```bash
export AUTODL_HOST=<host>
export AUTODL_PORT=<port>
export AUTODL_USER=root

bash scripts/09_remote_autodl/remote_setup.sh
bash scripts/09_remote_autodl/sync_to_autodl.sh

ssh -p "$AUTODL_PORT" "$AUTODL_USER@$AUTODL_HOST"
cd /root/SynPred_autorun_20260613/SynPred
bash scripts/09_remote_autodl/run_24h_cycle.sh
```

Collect after the run:

```bash
bash scripts/09_remote_autodl/collect_results.sh
```

Remote output root:

`/root/SynPred_autorun_20260613/SynPred/outputs/autorun/24h_optimization_20260613`

