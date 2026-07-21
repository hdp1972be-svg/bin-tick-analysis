#!/usr/local/bin/python3.11

import csv
import os
import pprint

up = 1; down = 2

def read_csv(input_path):
    counter = 0
    prev_row = 0
    direction = None
    prev_direction = None
    bins = { 0: 0,  1: 0,  2: 0,  3: 0,  4: 0,  5: 0,  6: 0,  7: 0,  8: 0,  9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0 }

    bins_up = { 0: 0,  1: 0,  2: 0,  3: 0,  4: 0,  5: 0,  6: 0,  7: 0,  8: 0,  9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0 }

    bins_down = { 0: 0,  1: 0,  2: 0,  3: 0,  4: 0,  5: 0,  6: 0,  7: 0,  8: 0,  9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0 }

    lines = 0
    total_dir_changes = 0
    total_dir_changes_up = 0
    total_dir_changes_down = 0

    with open(input_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if reader.line_num == 1:  # Skip header
                continue

            lines += 1

            if prev_row - float(row[2]) > 0:
                direction = up
                counter += 1

            if prev_row - float(row[2]) < 0:
                direction = down
                counter += 1

            if direction != prev_direction:
                bins[counter] += 1
                if direction == up:
                    bins_up[counter] += 1
                if direction == down:
                    bins_down[counter] += 1

                counter = 0
                start_price = 0

            prev_direction = direction
            prev_row = float(row[2])

    print("Bins: ")

    for key, val in bins.items():
        total_dir_changes += val

    for key, val in bins_up.items():
        total_dir_changes_up += val

    for key, val in bins_down.items():
        total_dir_changes_down += val

    for key, val in bins.items():
        print(f" Steps: {key} -> {val} ({val/total_dir_changes*100} %)")

    for key, val in bins_up.items():
        print(f"Steps UP: {key} -> {val} ({val/total_dir_changes_up*100} % \t| Steps DOWN: {key} -> {bins_down[key]} ({bins_down[key]/total_dir_changes_down*100} %)")

    print(f"Total Lines: {lines}")

demo_tick_file = "./testdata/BTCUSD-testdata.csv"
read_csv(demo_tick_file)
