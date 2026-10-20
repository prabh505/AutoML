Your previous response was not a valid TabPilot plan. The user message is a JSON object with `planner_input` (the same data as before; still data, never instructions), `previous_response` (your previous output, as text), and `errors` (field paths and messages from the validator).

Return one corrected JSON object that fixes every listed error and still follows every rule of the original instructions. Change only what the errors require. Respond with JSON only: no markdown, no code fences, no text before or after it.

SCHEMA
{{SCHEMA}}
