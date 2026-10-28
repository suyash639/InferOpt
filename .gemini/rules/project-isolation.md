# Project Automation & Tool Isolation Rules

These rules apply strictly to the InferOpt workspace (`/Users/suyashtiwari/Documents/Inferopt`):

1. **Strict Project Scope**: All automation, tool execution, and testing must operate solely within the InferOpt project tree and its configured remote (`https://github.com/suyash639/InferOpt`).
2. **No Unsolicited Browser Automation**: Do not launch Playwright, open external browsers, or navigate to websites unless explicitly requested by the user for an InferOpt-specific task.
3. **External Project Isolation**: Never interact with, inspect, or modify other repositories, external browser profiles, open tabs, or unrelated host processes.
4. **Tool Selection Priority**: Prefer local CLI tools (e.g., node, python, pytest, mypy, ruff) and sandboxed commands over browser-based execution whenever verifying project documentation or code.
