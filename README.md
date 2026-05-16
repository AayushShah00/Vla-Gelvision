# Vla-Gelvision
Multimodal Tactile-Aware Manipulation via Octo VLA and Moonshine ASR
Overview
This repository contains the implementation of a multimodal robotic control system leveraging a Vision-Language-Action (VLA) architecture based on the Octo model, seamlessly integrated with Moonshine ASR for real-time vocal command processing. The system is designed to execute tasks requiring both extreme delicacy (e.g., sorting strawberries from blueberries based on material compliance) and high precision (e.g., industrial bolt-threading).

System Architecture
[Insert System Architecture Diagram Here]

Our architecture employs a Late Fusion approach to combine distinct sensory modalities before final action generation.

Visual Processing: A Vision Transformer (ViT) processes the primary camera input, extracting spatial features and object affordances.

Tactile Processing: A dedicated Tactile Encoder processes input from a GelSight-style optical sensor. By tracking the deformation of gels and internal markers, the model leverages specific inductive biases to interpret surface interactions that visual data alone cannot capture.

Fusion & Action: The encoded features are fused late in the network, allowing the Octo VLA to generate highly accurate manipulation trajectories using Action Chunking for smooth, non-myopic motor execution.

Voice Interface: Moonshine ASR
The system features real-time, on-device command processing via Moonshine ASR.
Unlike traditional pipelines where ASR strictly feeds text strings to a downstream planner, our implementation utilizes a bi-directional token exchange. Moonshine's acoustic and linguistic tokens are mapped directly into the Octo model's embedding space. This allows the VLA to dynamically adjust its attention weights based on continuous vocal feedback (e.g., a human operator saying "stop, a bit softer") while managing audio context through Sliding Window Memory Management.

Tactile Sensing Logic
To handle complex physical interactions, the tactile sensing stream is quantized into specific embeddings:

Tokenization: Deformation data from the optical sensor is discretized into distinct 'Hardness' and 'Texture' tokens. This allows the VLA to logically reason about the physical properties of the grasped object (e.g., identifying the high material compliance of a strawberry versus a rigid bolt).

Bridging the Sim-to-Real Gap: To ensure robust deployment from simulation to the physical hardware, we introduced targeted domain randomization during training. This included injecting synthetic sensor noise and enforcing a strict 200ms hysteresis delay in the simulation environment, perfectly mirroring the latency profile of the physical GelSight-style sensor.

Safety & Fail-safes
Safety in human-robot interaction is paramount. The system implements a strict Contradiction Resolution protocol at the control loop level:

If the VLA's immediate perception (visual bounding boxes or tactile hardness tokens) fundamentally conflicts with the Moonshine command (e.g., the user commands "squeeze firmly" but the tactile sensor registers a delicate berry), the contradiction triggers a hardware interrupt.

Safe State: The robotic arm immediately halts trajectory execution, enters a 'Safe State' (initiating a gentle drop/release maneuver), and illuminates a dedicated physical red LED on the GPIO array to alert the operator.

Technical Specifications & Performance
Framework: PyTorch

Deployment Optimization: FP16 Quantization via TensorRT for edge deployment.

Hardware: The control node runs on a Raspberry Pi utilizing a Real-Time Operating System (RTOS) kernel to guarantee deterministic control loop frequencies.

Task Performance: The system achieves a high success rate in binary sorting tasks (strawberries vs. blueberries) relying solely on tactile compliance feedback, and maintains the sub-millimeter accuracy required for precision bolt-threading.
