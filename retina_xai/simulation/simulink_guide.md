# RetiNetra SimEvents Discrete-Event Triage Architecture

## Topology Overview
1. **Entity Generator:** Simulates fundus scans arriving from Primary Health Centers (PHCs).
2. **Network Queue & Server:** Models bandwidth latency and upload bottlenecks across regional clinics.
3. **GPU Inference Server:** Simulates EfficientNet-B0 inference execution (15 fps).
4. **Output Switch (Routing):** Directs scans based on calculated severity:
   - Grades 0–1 -> Auto-Archived.
   - Grades 2–4 -> Priority Specialist Review Queue.
5. **Specialist Reviewers:** Priority-based server queue serving highest severity Grade 4 cases first.

## Model Configuration Instructions
1. Open MATLAB and load your Simulink Discrete-Event model file (`retinetra_triage_sim.slx`).
2. Run simulations directly in Simulink to analyze queue wait times and GPU server utilization.