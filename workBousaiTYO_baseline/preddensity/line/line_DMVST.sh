#!/bin/bash
base_path="../../../bousai_tokyo_jiang/DMVST_density/"  # read data(graph_embed.txt) from this path; generate data(graph_embed_1and2.txt) to this path

graph_path=$base_path"graph_embed.txt"
embed1_path=$base_path"graph_embed_1.line"
embed2_path=$base_path"graph_embed_2.line"
norm1_path=$base_path"graph_embed_1_norm.line"
norm2_path=$base_path"graph_embed_2_norm.line"
topo_path=$base_path"graph_embed_1and2.txt"

chmod 777 line concatenate normalize
./line -train $graph_path -output $embed1_path -size 16 -order 1 -binary 1
./line -train $graph_path -output $embed2_path -size 16 -order 2 -binary 1
./normalize -input $embed1_path -output $norm1_path -binary 1
./normalize -input $embed2_path -output $norm2_path -binary 1
./concatenate -input1  $norm1_path -input2 $norm2_path -output $topo_path
