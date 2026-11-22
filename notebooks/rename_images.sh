#!/usr/bin/env bash
# Renomeia as imagens de uma pasta para image_0001.<ext>, image_0002.<ext>, ...
# A extensão original de cada arquivo é mantida (só o nome muda).
#
# Uso:
#   bash rename_images.sh <pasta> [--dry-run]
#
#   --dry-run  mostra o que seria feito, sem renomear nada.

set -euo pipefail

DIR="${1:-}"
DRY_RUN=0
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ -z "$DIR" || ! -d "$DIR" ]]; then
  echo "Uso: bash $0 <pasta> [--dry-run]" >&2
  exit 1
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

# Passo 1: renomeia tudo para um nome temporário.
# Isso evita colisões (ex.: um arquivo que já se chama image_0002.jpg).
TMP_PREFIX=".tmp_rename_$$_"
declare -a TMPS
i=0
for f in "${FILES[@]}"; do
  base=$(basename "$f")
  ext="${base##*.}"                       # extensão original, intacta
  i=$((i + 1))
  new=$(printf "image_%0${WIDTH}d.%s" "$i" "$ext")
  echo "$base -> $new"
  if (( ! DRY_RUN )); then
    tmp="$DIR/${TMP_PREFIX}${i}.${ext}"
    mv -- "$f" "$tmp"
    TMPS+=("$tmp|$DIR/$new")
  fi
done

# Passo 2: nomes temporários -> nomes finais
if (( ! DRY_RUN )); then
  for pair in "${TMPS[@]}"; do
    mv -- "${pair%%|*}" "${pair#*|}"
  done
  echo "Pronto: $TOTAL imagens renomeadas em '$DIR'."
else
  echo "(dry-run: nada foi alterado)"
fi
