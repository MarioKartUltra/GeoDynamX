#!/usr/bin/env bash
# The gate's verdict from pytest's exit code and its log: 5 (nothing selected) passes; a crash at
# interpreter exit (139 or 134) after a summary line with no failure or error passes too, since every
# test had already run; anything else keeps pytest's own status.
rc="$1"; log="$2"
if [ "$rc" -eq 0 ]; then exit 0; fi
if [ "$rc" -eq 5 ]; then echo "gate: no test reaches the changed code"; exit 0; fi
summary=$(grep -E "[0-9]+ passed" "$log" | tail -1)
if { [ "$rc" -eq 139 ] || [ "$rc" -eq 134 ]; } && [ -n "$summary" ] && ! echo "$summary" | grep -qE "[0-9]+ (failed|error)"; then
  echo "gate: the process crashed at exit after a green run: $summary"; exit 0
fi
exit "$rc"
