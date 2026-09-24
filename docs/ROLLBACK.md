# Rollback

Everything Switchboard changed outside its own directory, and how to undo it.
Nothing here is required for the stack to work: with Switchboard stopped, the
machine behaves exactly as it did before.

## What was changed

| File | Change | Backup |
| --- | --- | --- |
| `~/.cptr/config.toml` | Added a "Switchboard" chat connection and a "switchboard" MCP tool server. Existing entries untouched. | `backups/cptr-config.toml.20260921-151843.bak`, moved to the Recycle Bin on 2026-09-23 because it held cptr's plaintext Browser Use key; `integrate-cptr.ps1 -Remove` undoes the edit without it |
| `C:\AI\start.ps1` | Added a marker-bracketed block that starts Switchboard before the final health check. | `backups/start.ps1.20260921-101859.bak` |
| `C:\AI\Config\models.json` | `activeModel` changed from `medgemma-27b-it-Q4_K_M.gguf` to `Qwen3.5-9B-UD-Q6_K_XL.gguf`, by the dashboard, when a model was loaded. On 2026-09-23 the `Qwen3.5-9B-Q4_K_M.gguf` profile, whose file does not exist, was removed. | `backups/models.json.20260921-150256.premanual.bak` (before any change), `backups/models.json.<stamp>.bak` (one per Switchboard process, before its first switch), `backups/models.json.20260923-082421.pre-stale-cleanup.bak` |
| `C:\AI\dashboard.html` | Added a **Settings** tab that frames Switchboard's settings panel (API keys included). Restart the dashboard server to pick up a change to this file. | `backups/dashboard.html.20260923-093849.bak`; also in the `C:\AI` repo's history |

Created, not changed: `C:\AI\Switchboard\` (the project), and
`C:\AI\Switchboard\data\switchboard.sqlite3` (its database).

**Not touched at all:** the GGUF files, `Tools\llama-native\`,
`dashboard-server.ps1`, `stop.ps1`, `Build-LlamaArgs.ps1`, the existing cptr
connection to llama-server, the Personal Health MCP server, and any other
project under `C:\AI`.

## Undo, one piece at a time

### Stop using Switchboard (no changes needed)

```powershell
C:\AI\Switchboard\scripts\stop.ps1
```

llama-server, the dashboard and cptr keep running. cptr's "Llama.cpp"
connection still talks to llama-server directly; only the "Switchboard"
connection stops working.

### Remove it from cptr

```powershell
cd C:\AI\Switchboard
.\scripts\integrate-cptr.ps1 -Remove      # stop cptr first
```

Or, if you restore the backup from the Recycle Bin first, put it back wholesale:

```powershell
Copy-Item C:\AI\Switchboard\backups\cptr-config.toml.<stamp>.bak $HOME\.cptr\config.toml -Force
```

cptr reads `config.toml` at startup, so it must be stopped for either.

### Remove it from start.ps1

```powershell
cd C:\AI\Switchboard
.\scripts\integrate-start.ps1 -Remove
```

The block is bracketed by
`# --- Switchboard (added by ...) ---` and `# --- end Switchboard ---`, so
deleting those lines by hand is equally valid. Or restore the backup:

```powershell
Copy-Item C:\AI\Switchboard\backups\start.ps1.<stamp>.bak C:\AI\start.ps1 -Force
```

### Put the active model back

Switchboard routes through the dashboard, which records the loaded model as the
stack's `activeModel`. Check the Models tab for the model currently loaded;
it changed during the 2026-09-23 review as the stack was used.
To go back to MedGemma, use the dashboard's Models tab, or:

```powershell
C:\AI\model-manage.ps1 load medgemma-27b-it-Q4_K_M
```

Restoring the backup file instead only changes what `start.ps1` loads *next*
time; it does not unload the running model.

```powershell
Copy-Item C:\AI\Switchboard\backups\models.json.<stamp>.premanual.bak C:\AI\Config\models.json -Force
```

### Remove Switchboard entirely

```powershell
C:\AI\Switchboard\scripts\stop.ps1
cd C:\AI\Switchboard
.\scripts\integrate-cptr.ps1 -Remove       # cptr stopped
.\scripts\integrate-start.ps1 -Remove
Remove-Item C:\AI\Switchboard -Recurse -Force
```

That deletes the project, its virtual environment and its database. Nothing
else on the machine depends on it.

## Making it inert without removing it

Safer than uninstalling, and reversible with one word:

```toml
# config/switchboard.toml
[routing]
mode = "shadow"     # never switches models; records what it would have done
```

or `mode = "off"` for a pure pass-through proxy that routes nothing.

## If something goes wrong

```powershell
C:\AI\Switchboard\scripts\doctor.ps1
```

`doctor` works with everything stopped and names the file to look at for each
failure. The usual causes:

| Symptom | Cause | Fix |
| --- | --- | --- |
| `llama-server FAIL` | the stack is not running | `C:\AI\start.ps1` |
| `dashboard FAIL` | dashboard not running; switches cannot happen | start it, or set `routing.mode = "shadow"` |
| Port 8002 in use | something else took it | change `server.port` in `config/switchboard.toml` |
| cptr shows no Switchboard models | cptr started before the config edit | restart cptr |
| Switchboard will not start | see `logs/switchboard.err.log` | |
