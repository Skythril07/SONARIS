#!/bin/sh
# Download datasets described in configs/datasets.yaml when a source URL is supplied.
# Entries without a URL intentionally remain manual; do not guess a source or licence.
set -eu

config_path="configs/datasets.yaml"
[ -f "$config_path" ] || { echo "dataset config not found: $config_path" >&2; exit 1; }

awk '
function emit() { if (name != "") print name "|" path "|" licence "|" url }
/^  [A-Za-z0-9_-]+:$/ { emit(); name=$0; sub(/^  /,"",name); sub(/:$/,"",name); path=licence=url=""; next }
/^    path:/ { path=$0; sub(/^    path:[[:space:]]*/,"",path); next }
/^    licence:/ { licence=$0; sub(/^    licence:[[:space:]]*/,"",licence); next }
/^    url:/ { url=$0; sub(/^    url:[[:space:]]*/,"",url); next }
END { emit() }
' "$config_path" | while IFS='|' read -r name path licence url; do
    echo "dataset: $name"
    echo "license: ${licence:-not listed}"
    if [ -z "$url" ] || [ "$url" = "TODO" ]; then
        echo "source URL: not listed"
        echo "manual download required: see $config_path for $name"
        continue
    fi
    echo "source URL: $url"
    if [ -e "$path" ]; then
        echo "already present: $path"
        continue
    fi
    mkdir -p "$path"
    archive="$path/${name}.download"
    echo "downloading: $archive"
    curl -fL --retry 3 "$url" -o "$archive"
done
