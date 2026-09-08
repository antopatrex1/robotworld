# World Labs connection

This workspace uses the World Labs API through `scripts/worldlabs.py` (Python 3, no additional dependencies). This is a local API integration, not a globally installed Codex plugin.

The key is stored in `.env.worldlabs`, excluded by `.gitignore`, with owner-only file permissions. `WORLDLABS_API_KEY` in the environment overrides that file. Do not print, commit, or put the key in frontend code.

Read-only commands:

```sh
python3 scripts/worldlabs.py credits
python3 scripts/worldlabs.py list
python3 scripts/worldlabs.py get WORLD_ID
python3 scripts/worldlabs.py operation OPERATION_ID
```

To create a world after the user requests generation, write a JSON request file and run `python3 scripts/worldlabs.py generate --request request.json`. This uses API credits. Requests follow https://docs.worldlabs.ai/api; a text request contains `model`, `display_name`, and `world_prompt` with `type: "text"` and `text_prompt`. Image, multi-image, and video requests use the same command with their documented request bodies.

Generation returns an operation ID. Check it with the `operation` command until `done` is true; inspect `error` before treating the result as successful. Generation requests are not automatically retried to avoid duplicate paid jobs.

API billing is separate from the Marble web app: https://docs.worldlabs.ai/api/faq
