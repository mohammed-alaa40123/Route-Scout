#!/bin/bash -e
# if you choose fermi change frameworks line and uncomment GRAPH_PATH line
PYTHON=/scratch/gilbreth/mohame43/Routenet/routenet/bin/python
SAMPLE_INDEX=0
FRAMEWORKS=("erlang") #fermi
dataset="TrafficModels"
METRICS=("delay" "jitter")
TRAFFIC_MODES=("all_multiplexed" "autocorrelated" "constant_bitrate" "modulated" "onoff")
SOURCE=0
TARGET=5
K=1

# Define possible graphs for each tt
declare -A GRAPHS
GRAPHS[train]="geant2 nsfnet"
GRAPHS[test]="gbn"

# Loop over train/test
for tt in train test; do
    for graph in ${GRAPHS[$tt]}; do
        # Set graph path
        GRAPH_PATH="/scratch/gilbreth/mohame43/Routenet/RouteNet-Fermi/data/traffic_models/autocorrelated/${tt}/${graph}-autocorrelated/graphs/graph_attr.txt"

        # GRAPH_PATH="/content/RouteNet-Fermi/data/traffic_models/all_multiplexed/${tt}/${graph}-multiplexed/graphs/graph_attr.txt"

        # Generate candidate routes for this combination
        ROUTES_FILE="/scratch/gilbreth/mohame43/Routenet/candidate_routes_${SOURCE}_${TARGET}_${tt}_${graph}.txt"
        echo "Generating routes for tt=$tt, graph=$graph ..."
        $PYTHON k_shortest_routes.py \
            --src $SOURCE \
            --target $TARGET \
            --k $K \
            --graph_attr_path $GRAPH_PATH \
            --output_path $ROUTES_FILE

      for framework in "${FRAMEWORKS}"; do
          for metric in "${METRICS[@]}"; do
              for traffic_mode in "${TRAFFIC_MODES[@]}"; do
                  OUTPUT_FILE="/scratch/gilbreth/mohame43/Routenet/Results/MSE/${framework}_TM/${dataset}_results_${SOURCE}_${TARGET}_${traffic_mode}_${metric}_${dataset}_${graph}_${framework}.txt"

                  # Fix graphp naming
                  if [ "$traffic_mode" == "all_multiplexed" ]; then
                      graphp="${graph}-multiplexed"
                  elif [ "$traffic_mode" == "constant_bitrate" ]; then
                      graphp="${graph}-constant"
                  else
                      graphp="${graph}-${traffic_mode}"
                  fi

                #   echo "Running sample_index=$SAMPLE_INDEX, metric=$metric, traffic_mode=$traffic_mode ..."
                #   $PYTHON predict_with_candidate_routes.py \
                #       --sample_index $SAMPLE_INDEX \
                #       --metric $metric \
                #       --traffic_mode $traffic_mode \
                #       --routes_file $ROUTES_FILE \
                #       --tt $tt \
                #       --graph $graphp \
                #       > $OUTPUT_FILE 2>&1

                  $PYTHON "/scratch/gilbreth/mohame43/Routenet/predict_unified_with_candidate_routes.py" \
                    --framework $framework \
                    --dataset_split $tt \
                    --metric $metric \
                    --traffic_mode $traffic_mode \
                    --routes_file $ROUTES_FILE \
                    --topology $graph \
                    > $OUTPUT_FILE 2>&1
                  echo "Saved output to $OUTPUT_FILE"
              done
          done

        done
    done
done
