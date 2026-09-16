# RoboGuardian Evidence Demo

This directory contains a read-only Streamlit dashboard for the RoboGuardian submission. It replays checked-in JSON evidence and event traces from the repository; it does not run a physical robot and it does not claim safety certification.

## Run locally

From the repository root:

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## What the demo shows

The dashboard presents the G6 A/B/C reliability result, a selectable replay of the recorded baseline or governed-recovery trace, the G3 controlled failure, the H1 robustness matrix, H2 simulation-provider latency, and G7 perception metrics. Each section identifies its source artifact.

## Hosting for a submission URL

Use a public GitHub repository as the source and select `demo/app.py` as the Streamlit entrypoint on the hosting platform. If the host asks for a dependency file, use `demo/requirements.txt`. The application URL should point to the hosted dashboard, not to the repository alone.

## Claim boundary

This is an evidence replay for a controlled MuJoCo simulation. It is not a live physical SO-101 demo, it is not an end-to-end camera-to-actuator latency benchmark, and it does not provide safety certification.
