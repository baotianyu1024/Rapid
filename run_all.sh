#!/bin/bash
# RAPID full pipeline: CPS + ACP + Verification
# Usage: ./run_all.sh <input_bin> <num_devices> <output_dir> [options]
#
# Example (Criteo Terabyte, 8 GPUs):
#   ./run_all.sh /path/to/original.bin 8 /path/to/output \
#       --table_lengths 3 2 1 2 6 1 1 1 1 7 3 8 1 6 9 5 1 1 1 12 100 27 10 3 1 1 \
#       --dense_count 13 --topk 40000 --batch_size 8192
#
# Example (Taobao, 32 GPUs):
#   ./run_all.sh /path/to/taobao_4tables.bin 32 /path/to/output \
#       --table_lengths 1 1 1 1 --dense_count 1 --topk 40000 --batch_size 8192

set -e
cd "$(dirname "$0")"

# Parse required args
INPUT=$1
ND=$2
OUTPUT=$3
shift 3

# Parse optional args with defaults
TABLE_LENGTHS=""
DENSE_COUNT=13
TOPK=40000
BATCH_SIZE=8192
NUM_BATCHES=""
HOT_BATCHES=80

while [[ $# -gt 0 ]]; do
    case $1 in
        --table_lengths)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                TABLE_LENGTHS="$TABLE_LENGTHS $1"
                shift
            done
            ;;
        --dense_count) DENSE_COUNT=$2; shift 2 ;;
        --topk) TOPK=$2; shift 2 ;;
        --batch_size) BATCH_SIZE=$2; shift 2 ;;
        --num_batches) NUM_BATCHES=$2; shift 2 ;;
        --hot_batches) HOT_BATCHES=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TABLE_LENGTHS=$(echo $TABLE_LENGTHS | xargs)

if [[ -z "$INPUT" || -z "$ND" || -z "$OUTPUT" || -z "$TABLE_LENGTHS" ]]; then
    echo "Usage: $0 <input_bin> <num_devices> <output_dir> --table_lengths ... [--dense_count N] [--topk K] [--batch_size B]"
    exit 1
fi

# Compute total sparse columns and record size
TOTAL_SPARSE=0
for tl in $TABLE_LENGTHS; do TOTAL_SPARSE=$((TOTAL_SPARSE + tl)); done
RECORD_SIZE=$((4 + DENSE_COUNT * 4 + TOTAL_SPARSE * 4))
FILE_SIZE=$(stat -c%s "$INPUT")
TOTAL_SAMPLES=$((FILE_SIZE / RECORD_SIZE))
GLOBAL_BATCH=$((ND * BATCH_SIZE))

if [[ -z "$NUM_BATCHES" ]]; then
    NUM_BATCHES=$((TOTAL_SAMPLES / GLOBAL_BATCH))
fi

NB_HOT=$((NUM_BATCHES < HOT_BATCHES ? NUM_BATCHES : HOT_BATCHES))
HOT_ROWS=$((NB_HOT * GLOBAL_BATCH))

echo "============================================================"
echo "RAPID Pipeline"
echo "============================================================"
echo "  Input:        $INPUT ($TOTAL_SAMPLES samples, $((FILE_SIZE/1024/1024)) MB)"
echo "  Devices:      $ND"
echo "  Batch size:   $BATCH_SIZE (global: $GLOBAL_BATCH)"
echo "  Batches:      $NUM_BATCHES"
echo "  Table lengths: $TABLE_LENGTHS"
echo "  Dense count:  $DENSE_COUNT"
echo "  Top-K:        $TOPK"
echo "  Output:       $OUTPUT"
echo ""

mkdir -p "$OUTPUT"/{hot_ids,alloc_output,dataset_output}

# Step 0: Hot ID Extraction
echo "=== Step 0: Hot ID Extraction ==="
python3 cps/hot_id_extraction.py \
    --input "$INPUT" \
    --output "$OUTPUT/hot_ids" \
    --topk $TOPK \
    --dense_count $DENSE_COUNT \
    --table_lengths $TABLE_LENGTHS \
    --max_rows $HOT_ROWS
echo ""

# Step 1-4: ID Allocation (CPS embedding co-location)
echo "=== Step 1-4: ID Allocation ==="
OMP_NUM_THREADS=16 cps/arrange_ids_to_devices \
    --input_bin "$INPUT" \
    --hot_ids_dir "$OUTPUT/hot_ids" \
    --output_dir "$OUTPUT/alloc_output" \
    --batch_size $BATCH_SIZE \
    --num_devices $ND \
    --dense_count $DENSE_COUNT \
    --n_batches $NUM_BATCHES \
    --table_lengths $TABLE_LENGTHS
echo ""

# Step 5: Sample Scheduling + Dataset Generation
echo "=== Step 5: Sample Scheduling ==="
OMP_NUM_THREADS=8 cps/generate_local_dataset \
    --input_file "$INPUT" \
    --table_files_dir "$OUTPUT/alloc_output" \
    --output_dir "$OUTPUT/dataset_output" \
    --num_devices $ND \
    --train_batch_size $BATCH_SIZE \
    --steps_per_chunk 1 \
    --num_chunks $NUM_BATCHES \
    --num_threads 8 \
    --dense_count $DENSE_COUNT \
    --gamma 0.2 \
    --table_lengths $TABLE_LENGTHS
echo ""

# Verification: Hit Rate
echo "=== Verification: Hit Rate ==="
OMP_NUM_THREADS=8 analysis/verify_hit_rate \
    --dataset_dir "$OUTPUT/dataset_output" \
    --num_devices $ND \
    --batch_size $BATCH_SIZE \
    --num_batches $NUM_BATCHES \
    --dense_count $DENSE_COUNT \
    --table_lengths $TABLE_LENGTHS
echo ""

# ACP: Strategy Selection
echo "=== ACP: Strategy Selection ==="
python3 acp/compute_acp.py \
    --input "$INPUT" \
    --num_gpus $ND \
    --batch_size $BATCH_SIZE \
    --table_lengths $TABLE_LENGTHS \
    --dense_count $DENSE_COUNT \
    --num_batches 50 \
    --cps_dataset_dirs "$OUTPUT/dataset_output" \
    --output "$OUTPUT/acp_results.json"
echo ""

echo "============================================================"
echo "Pipeline complete. Results in: $OUTPUT"
echo "============================================================"
