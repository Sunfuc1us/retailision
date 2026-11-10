# RetailVision — Design Principles

## Required Workflow — Stop That Shit

All subsequent development in this project must follow the `stop-that-shit` skill.
Before starting work, read `/home/gzh/.agents/skills/stop-that-shit/SKILL.md` and apply it throughout the task.

- Complete the requested scope and its necessary consequences; do not add speculative features, abstractions, dependencies, or defensive mechanisms.
- Prefer existing project code and direct, maintainable solutions.
- Expand work only to close a concrete requirement or observed gap.
- Preserve required validation, accessibility, data integrity, offline inference, and CUDA guarantees.
- Run the checks required for the affected behavior, reuse still-valid evidence, and finish once acceptance criteria are met. Avoid redundant verification loops.
- Resolve routine choices autonomously within the user's authorization. Read-only review tasks remain read-only.

## Core Philosophy

Less, but better.

Simple Goals. Simple Architecture. Simple Design. Simple Implementation.

Design inspiration: Linear and Apple Notes.

Do not imitate their appearance. Adopt their restraint, clarity, hierarchy, and interaction philosophy.

## Non-negotiable Rules

1. Every page must have one clear primary purpose.
2. Remove slogans, decorative copy, redundant labels, unnecessary badges, and meaningless UI elements.
3. Prefer layout and visual hierarchy over explanatory text.
4. If a page requires excessive instructions to be understandable, redesign the layout.
5. Use whitespace, typography, alignment, and subtle separators to establish hierarchy.
6. Avoid unnecessary cards, borders, gradients, shadows, animations, and decorative icons.
7. Every component must serve a real functional purpose.
8. Do not add abstractions, dependencies, or configuration options without a concrete need.
9. Preserve essential accessibility labels, semantic structure, units, and error messages.
10. Never sacrifice usability for visual minimalism.

## UX / DX / AX

Every design and engineering decision must be evaluated against:

- UX: Is it easier and more intuitive for users?
- DX: Is it simpler to implement, understand, debug, and maintain?
- AX: Is it easier for AI agents to understand, operate, and verify?

Prefer solutions that improve all three dimensions.

## Engineering Rules

- Keep components small and responsibilities explicit.
- Maintain predictable state transitions.
- Use clear, typed API contracts.
- Avoid duplicated business logic.
- Keep frontend and inference backend decoupled.
- Do not implement speculative features.
- Build only what the current milestone requires.

## Visual Acceptance

Before considering a UI task complete:

1. Run the actual application.
2. Inspect real screenshots.
3. Check spacing, hierarchy, typography, alignment, and interaction.
4. Remove unnecessary elements.
5. Verify responsive behavior and accessibility.
6. Confirm that the final design is simpler than the initial implementation.

When in doubt, remove rather than add.

Keep all inference offline. Never report fixture results as retail accuracy. Preserve historical model and dataset versions. CUDA is required for model work.

Development root: /home/gzh/rocco/retailvision. The former DemoRoom copy is historical only; never use it for new work.
Historical logs and delivery reports may contain source paths; never use those paths for runtime imports or model loading. Preserve model checkpoints and dataset versions during relocation.
