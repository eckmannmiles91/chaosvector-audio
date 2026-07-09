#!/bin/bash
# Force the echo-cancelled PipeWire source (ec_source) as the default input.
#
# The XVF3800 hardware AEC is dead (measured 2026-07-09: self-echo RMS ~15000 on
# the raw array vs ~450 through ec_source). Without this the daemon captures the
# raw array, hears its own playback, and false-barges/cuts off its own answers.
# A USB replug re-enumerates the array and can steal the default back, so this
# runs as ExecStartPre on every daemon start to re-assert it.
for i in $(seq 1 20); do
  if wpctl status 2>/dev/null | grep -q "Echo Cancelled Microphone"; then
    pw-metadata -n default 0 default.configured.audio.source '{"name":"ec_source"}' >/dev/null 2>&1
    pw-metadata -n default 0 default.audio.source '{"name":"ec_source"}' >/dev/null 2>&1
    sleep 1
    got=$(wpctl inspect @DEFAULT_AUDIO_SOURCE@ 2>/dev/null | grep -oE 'node.name = "[^"]*"')
    echo "default source set: $got"
    exit 0
  fi
  sleep 1
done
echo "WARN: ec_source not found after 20s; daemon will use raw array (self-echo likely)"
exit 0
