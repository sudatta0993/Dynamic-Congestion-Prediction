import numpy as np
import pandas as pd
from generate_io_curves import get_links_output_curve

MINS_PER_DAY = 1440
MIN_INTERVALS = 5
NUM_BINS = MINS_PER_DAY / MIN_INTERVALS

def find_best_route_tts(best_route_input_curve, best_route_output_curve):
    tts = np.array(
            [(np.argmin(np.abs(best_route_output_curve.copy() - best_route_input_curve.iloc[i])) - i * MIN_INTERVALS)
             for i in range(NUM_BINS)])
    return tts

def find_route_switch_indices(best_route_tts, alternate_route_fftts, best_route_input_curve,
                              best_route_bottleneck_capacity):
    index_of_first_switch = next(x[0] for x in enumerate(best_route_tts) if x[1] > sum(alternate_route_fftts))
    index_of_last_switch = NUM_BINS - 1
    for i in range(index_of_first_switch, NUM_BINS - 1):
        if (best_route_input_curve.iloc[i + 1] - best_route_input_curve.iloc[i]) / float(
                MIN_INTERVALS) < best_route_bottleneck_capacity:
            index_of_last_switch = i
            break
    return index_of_first_switch, index_of_last_switch

def calculate_cumulative_diverted_curve(index_of_first_switch, index_of_last_switch, best_route_input_curve,
                                        best_route_bottleneck_capacity):
    cum_diverted_number = np.zeros(NUM_BINS)
    for i in range(index_of_first_switch, index_of_last_switch - 1):
        cum_diverted_number[i] = best_route_input_curve.iloc[i] - (
            best_route_input_curve.iloc[i - 1] + best_route_bottleneck_capacity * MIN_INTERVALS)
        best_route_input_curve.iloc[i] = best_route_input_curve.iloc[i] - cum_diverted_number[i]
    for i in range(index_of_last_switch - 1, NUM_BINS):
        cum_diverted_number[i] = cum_diverted_number[i - 1]
        best_route_input_curve.iloc[i] = best_route_input_curve.iloc[i] - cum_diverted_number[i]
    return cum_diverted_number, best_route_input_curve

def update_best_route_io_curves(best_route_input_curve, best_route_output_curve, best_route_bottleneck_capacity,
                                best_route_fftt):
    new_best_route_io_curves = pd.DataFrame(index=np.arange(0, MINS_PER_DAY, MIN_INTERVALS))
    new_best_route_io_curves[0] = best_route_input_curve
    new_best_route_output_curve = \
    get_links_output_curve(new_best_route_io_curves, [best_route_bottleneck_capacity], [best_route_fftt])[0]
    for i in range(NUM_BINS):
        best_route_output_curve.iloc[i] = new_best_route_output_curve.iloc[i]
    return best_route_input_curve, best_route_output_curve

def update_alternate_route_io_curves(alternate_route, alternate_route_fftts, cum_diverted_number):
    for (link_number, (link_input_curve, link_output_curve)) in enumerate(alternate_route):
        for i in range(NUM_BINS):
            if i == 0:
                link_input_curve.iloc[i] = link_input_curve.iloc[i] + cum_diverted_number[i]
                if i + int(alternate_route_fftts[link_number] / MIN_INTERVALS) <= NUM_BINS - 1:
                    link_output_curve.iloc[i + int(alternate_route_fftts[link_number] / MIN_INTERVALS)] = \
                        link_output_curve.iloc[i + int(alternate_route_fftts[link_number] / MIN_INTERVALS)] + \
                    cum_diverted_number[i]
            else:
                if i + int(alternate_route_fftts[link_number - 1] / MIN_INTERVALS) <= NUM_BINS - 1:
                    link_input_curve.iloc[i + int(alternate_route_fftts[link_number - 1] / MIN_INTERVALS)] = \
                        link_input_curve.iloc[i + int(alternate_route_fftts[link_number - 1] / MIN_INTERVALS)] + \
                    cum_diverted_number[i]
                if i + int(alternate_route_fftts[link_number] / MIN_INTERVALS) + int(
                                alternate_route_fftts[link_number - 1] / MIN_INTERVALS) <= NUM_BINS - 1:
                    link_output_curve.iloc[i + int(alternate_route_fftts[link_number] / MIN_INTERVALS) + int(
                        alternate_route_fftts[link_number - 1] / MIN_INTERVALS)] = \
                        link_output_curve.iloc[i + int(alternate_route_fftts[link_number] / MIN_INTERVALS) + int(
                            alternate_route_fftts[link_number - 1] / MIN_INTERVALS)] + \
                        cum_diverted_number[i]
    return alternate_route


def check_route_choice(best_route_input_curve, best_route_output_curve, alternate_route, best_route_fftt,
                       best_route_bottleneck_capacity, alternate_route_fftts):
    best_route_tts = find_best_route_tts(best_route_input_curve, best_route_output_curve)
    best_route_max_tt = max(best_route_tts)
    if best_route_max_tt < sum(alternate_route_fftts):
        print "Alternate route not chosen"
    else:
        print "Alternative route chosen"
        index_of_first_switch, index_of_last_switch = find_route_switch_indices(best_route_tts, alternate_route_fftts,
                                                      best_route_input_curve, best_route_bottleneck_capacity)
        cum_diverted_number, best_route_input_curve = calculate_cumulative_diverted_curve(index_of_first_switch,
                                                                                          index_of_last_switch,
                                                                                          best_route_input_curve,
                                                                                          best_route_bottleneck_capacity)
        best_route_input_curve, best_route_output_curve = update_best_route_io_curves(best_route_input_curve,
                                                                                      best_route_output_curve,
                                                                                      best_route_bottleneck_capacity,
                                                                                      best_route_fftt)
        alternate_route = update_alternate_route_io_curves(alternate_route, alternate_route_fftts, cum_diverted_number)
    return best_route_input_curve, best_route_output_curve, alternate_route