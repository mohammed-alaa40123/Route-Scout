%%writefile predict_with_candidate_routes_scheduling.sh
#!/bin/bash -e
# if you choose fermi change frameworks line and uncomment GRAPH_PATH line
PYTHON=/scratch/gilbreth/mohame43/Routenet/routenet/bin/python
SAMPLE_INDEX=1
FRAMEWORKS=("erlang" "fermi") #fermi
dataset="Scheduling"
METRICS=("delay" "jitter")
TRAFFIC_MODES=("wfq")
SOURCE=0
TARGET=5
K=1

# Define possible graphs for each tt
declare -A GRAPHS
GRAPHS[train]="geant2 nsfnet"
GRAPHS[test]="rediris"

# Loop over train/test
for tt in train test; do
    for graph in ${GRAPHS[$tt]}; do
        # Set graph path
        GRAPH_PATH="/scratch/gilbreth/mohame43/Routenet/RouteNet-Fermi/data/scheduling/${tt}/graphs/graph-${graph}-wfq-6.txt"

        # GRAPH_PATH="/content/RouteNet-Fermi/data/traffic_models/all_multiplexed/${tt}/${graph}-multiplexed/graphs/graph_attr.txt"

        # Generate candidate routes for this combination
        ROUTES_FILE="/scratch/gilbreth/mohame43/Routenet/candidate_routes_scheduling_${SOURCE}_${TARGET}_${tt}_${graph}.txt"
        echo "Generating routes for tt=$tt, graph=$graph ..."
        $PYTHON k_shortest_routes.py \
            --src $SOURCE \
            --target $TARGET \
            --k $K \
            --graph_attr_path $GRAPH_PATH \
            --output_path $ROUTES_FILE

      for framework in "${FRAMEWORKS[@]}"; do
          for metric in "${METRICS[@]}"; do
              for traffic_mode in "${TRAFFIC_MODES[@]}"; do
                  OUTPUT_DIR="/scratch/gilbreth/mohame43/Routenet/Results/MSE/${framework}_S${SAMPLE_INDEX}"
                  mkdir -p "$OUTPUT_DIR"
                  OUTPUT_FILE="$OUTPUT_DIR/${dataset}_results_${SOURCE}_${TARGET}_${traffic_mode}_${metric}_${dataset}_${graph}_${framework}.txt"

                  echo "Running sample_index=$SAMPLE_INDEX, metric=$metric, traffic_mode=$traffic_mode ..."
                  $PYTHON "/scratch/gilbreth/mohame43/Routenet/predict_unified_with_candidate_routes_scheduling copy.py" \
                    --dataset_split $tt \
                    --framework $framework \
                    --metric $metric \
                    --sample_index $SAMPLE_INDEX \
                    --routes_file $ROUTES_FILE \
                    --graph $graph \
                    > $OUTPUT_FILE 2>&1
                  echo "Saved output to $OUTPUT_FILE"
              done
          done

        done
    done
done
