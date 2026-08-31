#!/usr/bin/env -S gnuplot -c
#
# Invoke as e.g.:
#
#   $ scripts/plot.gnuplot gmm

EVAL=ARG1
print "EVAL=".EVAL
DATA_PRIMAL = sprintf('%s-primal.data', EVAL)
DATA_DIFF = sprintf('%s-diff.data', EVAL)
DATA_RATIO = sprintf('%s-ratio.data', EVAL)
PLOT_PRIMAL = sprintf('%s-primal.svg', EVAL)
PLOT_DIFF = sprintf('%s-diff.svg', EVAL)
PLOT_RATIO = sprintf('%s-ratio.svg', EVAL)
TITLE_PRIMAL = sprintf('%s - primal', EVAL)
TITLE_DIFF = sprintf('%s - diff', EVAL)
TITLE_RATIO = sprintf('%s - primal ÷ diff', EVAL)

set key autotitle columnheader
set datafile missing "?"
set xlabel "Workload"
set ylabel "Seconds"
set logscale y 10
set xtics right rotate by 45
set key outside
set grid ytics lt -1
set style data linespoints
set pointsize 0.5

stats DATA_PRIMAL u (0)

set term svg font "Monospace,8"

set title TITLE_PRIMAL
set output PLOT_PRIMAL
plot for [i=2:STATS_columns] DATA_PRIMAL u (column(0)):i:xtic(1)

set title TITLE_DIFF
set output PLOT_DIFF
plot for [i=2:STATS_columns] DATA_DIFF u (column(0)):i:xtic(1)

set title TITLE_RATIO
set output PLOT_RATIO
set ylabel "Overhead"
plot for [i=2:STATS_columns] DATA_RATIO u (column(0)):i:xtic(1)
