#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  namenode)
    # Format only a brand-new named volume, never a populated namespace.
    if [ ! -f /hadoop/name/current/VERSION ]; then
      hdfs namenode -format -nonInteractive
    fi
    exec hdfs namenode
    ;;
  datanode) exec hdfs datanode ;;
  *) exec "$@" ;;
esac
