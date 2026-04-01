---
name: env_command_pattern
description: User requires specific bash pattern for running commands in llamafactory conda environment
type: feedback
---

Must use this exact pattern to run commands in the llamafactory conda env:

```bash
PYTHONPATH=. bash -c 'eval "$(conda shell.bash hook)" && conda activate llamafactory && CUDA_VISIBLE_DEVICES=1 python -u <script>' 2>&1 | tee /home/liuxinyao/output/ctagent/eval/<logname>.log
```

**Why:** Direct `conda run` or hardcoded python paths don't work in this environment. The conda hook + activate pattern is the only reliable way.

**How to apply:** Always use this pattern when running Python scripts. Reference WORKSPACE_GUIDE.md section 4 for command templates.
