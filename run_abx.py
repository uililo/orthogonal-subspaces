import ABXpy.task
import ABXpy.distances.distances as distances
import ABXpy.distances.metrics.cosine as cosine
import ABXpy.distances.metrics.dtw as dtw
import ABXpy.distances.metrics.kullback_leibler as kl
import ABXpy.score as score
import ABXpy.misc.items as items
import ABXpy.analyze as analyze
import os

def dtw_cosine_distance(x, y, normalized):
    return dtw.dtw(x, y, cosine.cosine_distance, normalized)

def dtw_kl_distance(x, y, normalized):
    return dtw.dtw(x, y, kl.kl_divergence, normalized)

def run_abx(feature_file, item_file, task_file, distance_file, score_file, analyze_file, distance):
    distance_funcs = {'KL': dtw_kl_distance, 'cosine': dtw_cosine_distance}
    distances.compute_distances(feature_file, '/features/', task_file, distance_file, distance_funcs[distance], normalized=True, n_cpu=1)
    score.score(task_file, distance_file, score_file)
    analyze.analyze(task_file, score_file, analyze_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_file', help = "h5features to be evaluated")
    parser.add_argument('item_file', help = "ABX item file")
    parser.add_argument('task_file', help = "ABX task file")
    parser.add_argument('distance_file', help = "file to save ABX distances")
    parser.add_argument('score_file', help = "file to save ABX scores")
    parser.add_argument('analyze_file', help = "output CSV file")
    parser.add_argument('distance', help = "distance function")
    args = parser.parse_args()
    run_abx(args.feature_file, args.item_file, args.task_file, args.distance_file, args.score_file, args.analyze_file, args.distance)