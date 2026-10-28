# Agent trace sheet — {{provider}} · {{question_id}}

> **Module 2 · Lab 02 artefact.** One sheet per traced run. The Lab 02 notebook fills the placeholders
> in double braces from a recorded trace and writes the result to `labs/module-02/outputs/`. Fill in
> the *Review* section by hand. Reuse the sheet whenever you debug an agent run: it is the minimum
> you should be able to answer about any agent turn before you change a prompt.

| Provider | Model | API | Date |
| -------- | ----- | --- | ---- |
| {{provider}} | {{model}} | {{api}} | {{date}} |

**User request:** {{question}}

## 1. Step-by-step trace

One row per model call, tool call and tool result, in order.

{{steps_table}}

## 2. Totals

| Model calls | Tool calls | Input tokens | Output tokens | Time to first token | Wall time | Estimated cost |
| ----------- | ---------- | ------------ | ------------- | ------------------- | --------- | -------------- |
{{totals_row}}

## 3. Final answer

{{final_answer}}

## 4. Review (fill in by hand)

| Question | Answer | Evidence (row in §1) |
| -------- | ------ | -------------------- |
| Did the model choose the right tool(s) for the request? | {{review_tool_choice}} | |
| Were the tool arguments valid and complete (IDs, formats)? | {{review_arguments}} | |
| Did any tool fail? Did the model notice and recover? | {{review_errors}} | |
| Is every fact in the final answer backed by a tool result? | {{review_grounding}} | |
| Did the model claim an action it could not take? | {{review_actions}} | |
| Where did the time go: model calls or tools? | {{review_latency}} | |
| What would you change first: the prompt, a tool description, a schema or the code? | {{review_change}} | |

## 5. Notes

{{notes}}
