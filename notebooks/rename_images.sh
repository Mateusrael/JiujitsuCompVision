#!/usr/bin/env bash
# Renomeia as imagens de uma pasta para image_0001.<ext>, image_0002.<ext>, ...
# A extensão original de cada arquivo é mantida (só o nome muda).
# Apenas para organizar imagens ANTES da criação das anotações.
#
# Uso:
#   bash rename_images.sh <pasta> [--dry-run]
#
#   --dry-run  mostra o que seria feito, sem renomear nada.

set -euo pipefail

DIR="${1:-}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ -z "$DIR" || ! -d "$DIR" || $# -gt 2 || ( $# -eq 2 && "$2" != "--dry-run" ) ]]; then
  echo "Uso: bash $0 <pasta> [--dry-run]" >&2
  exit 1
fi

DIR=$(cd -- "$DIR" && pwd -P)

# Os IDs das anotações dependem dos caminhos relativos. Procura também nos
# pais: o JSON do dataset costuma ficar ao lado da pasta de imagens.
if (( ! DRY_RUN )); then
  candidate="$DIR"
  while :; do
    for annotation in "$candidate"/annotations.json \
                      "$candidate"/*_annotations.json \
                      "$candidate"/annotations_*.json; do
      if [[ -f "$annotation" ]]; then
        echo "Recusado: anotações existentes em '$annotation'." >&2
        echo "Este utilitário só pode renomear imagens antes de criar as anotações; os IDs existentes devem ser preservados." >&2
        exit 1
      fi
    done
    [[ "$candidate" == / ]] && break
    parent=$(dirname -- "$candidate")
    [[ "$parent" == "$candidate" ]] && break
    candidate="$parent"
  done
fi

# Extensões consideradas imagem (comparação sem diferenciar maiúsculas)
EXTS="jpg|jpeg|png|bmp|gif|tif|tiff|webp"

# Lista as imagens em ordem alfabética "natural" (img2 antes de img10).
# Usa separador NUL para lidar com nomes com espaços.
mapfile -d '' FILES < <(
  find "$DIR" -maxdepth 1 -type f -regextype posix-extended \
       -iregex ".*\.($EXTS)$" -print0 | sort -z -V
)

TOTAL=${#FILES[@]}
if (( TOTAL == 0 )); then
  echo "Nenhuma imagem encontrada em '$DIR'."
  exit 0
fi

# Número de dígitos: no mínimo 4, mais se houver mais de 9999 imagens
WIDTH=$(( ${#TOTAL} > 4 ? ${#TOTAL} : 4 ))

# Planeja todos os nomes antes de mover qualquer arquivo.
TMP_PREFIX=".tmp_rename_$$_"
declare -a TMPS TARGETS
i=0
for f in "${FILES[@]}"; do
  base=$(basename "$f")
  ext="${base##*.}"                       # extensão original, intacta
  i=$((i + 1))
  new=$(printf "image_%0${WIDTH}d.%s" "$i" "$ext")
  echo "$base -> $new"
  if (( ! DRY_RUN )); then
    tmp="$DIR/${TMP_PREFIX}${i}.${ext}"
    if [[ -e "$tmp" || -L "$tmp" || -d "$DIR/$new" || -L "$DIR/$new" ]]; then
      echo "Recusado: nome temporário ou destino em uso: '$tmp' / '$DIR/$new'." >&2
      exit 1
    fi
    TMPS+=("$tmp")
    TARGETS+=("$DIR/$new")
  fi
done

# Dois passos evitam colisões entre nomes antigos e novos.
if (( ! DRY_RUN )); then
  for (( i=0; i<TOTAL; i++ )); do
    mv -- "${FILES[i]}" "${TMPS[i]}"
  done
  for (( i=0; i<TOTAL; i++ )); do
    mv -- "${TMPS[i]}" "${TARGETS[i]}"
  done
  echo "Pronto: $TOTAL imagens renomeadas em '$DIR'."
else
  echo "(dry-run: nada foi alterado)"
fi
