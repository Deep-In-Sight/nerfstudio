#!/bin/bash
# Hook to activate nerfstudio conda env for python/pytest/pip commands

# Read the tool input from stdin
input=$(cat)

# Extract the command
command=$(echo "$input" | jq -r '.tool_input.command // empty')

# Check if command contains python, pytest, or pip
if echo "$command" | grep -qE '(^|\s)(python|pytest|pip)(\s|$)'; then
    # Source conda.sh from CONDA_EXE path, then activate
    new_command="source \"\$(dirname \"\$CONDA_EXE\")/../etc/profile.d/conda.sh\" && conda activate nerfstudio && $command"

    # Output with correct hookSpecificOutput format
    jq -n --arg cmd "$new_command" '{
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {
                "command": $cmd
            }
        }
    }'
else
    # Pass through unchanged (no output = no modification)
    exit 0
fi
