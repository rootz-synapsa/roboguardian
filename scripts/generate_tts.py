import asyncio
import edge_tts

TEXT = """
In robotics, an open-loop plan is only valid if the world stands perfectly still. But the world never stands still. Here is our baseline arm. The plan was calculated for the plate's original position. But we moved the plate 10 centimeters. The arm executes the plan blindly, grasps thin air, and fails. In our tests, this open-loop baseline fails 10 out of 10 times.

This is RoboGuardian. Our thesis is simple: Governed execution is more resilient than open-loop execution when the world changes. We wrap the control loop in a governance layer. We observe the world, evaluate the drift, and if the error crosses a threshold, we pause, re-plan from fresh observations, and execute.

But how do you guarantee safety during a replan? The biggest risk is executing a stale action. In RoboGuardian, we enforce a structural guarantee we call Section 18: The stale plan is never executed. Look at the code. The stale target is passed in only to be logged for evidence. There is no code path that routes it to the actuators. The only values sent to the motors come from a fresh IK solve. It's enforced by construction, not by runtime checks.

Let's look at the head-to-head reliability test. 10 deterministic trials per arm. Arm A, the happy path, succeeds 10 out of 10. Arm B, the open-loop baseline under disturbance, fails 10 out of 10. Arm C, RoboGuardian, detects the disturbance, recovers, and succeeds 10 out of 10. Zero stale actions executed.

For perception, we trained a tiny CNN that achieves 0.7 centimeter accuracy. We compiled it with OpenVINO. Now, here is where we have to be honest. We expected OpenVINO to be faster. The data shows it's actually 0.44x the speed of ONNX runtime for a model this small. We don't claim a speedup that isn't in the data. But at under 2 milliseconds, it's well within our control budget.

This is controlled simulation evidence, backed by deterministic JSON artifacts you can reproduce yourself. Governed execution isn't just about being smarter; it's about being resilient. Check out the repo, run the gates, and verify the evidence for yourself. Thank you.
"""

async def main():
    voice = "en-US-ChristopherNeural"
    output_file = "narration.mp3"
    
    print(f"Generating AI Voiceover using {voice}...")
    communicate = edge_tts.Communicate(TEXT.strip(), voice, rate="-2%")
    await communicate.save(output_file)
    print(f"Done! Audio saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
