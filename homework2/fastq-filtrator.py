def main(input_fastq, output_file_prefix, gc_bounds=100, length_bounds=2**32, quality_threshold=0, save_filtered=False):
    if save_filtered:
        with open(input_fastq, 'r') as init_file, open(output_file_prefix +'_passed.fastq', 'w') as passed_file, \
                open(output_file_prefix + '_failed.fastq', 'w') as failed_file:
            open_and_check(gc_bounds, length_bounds, quality_threshold, init_file, passed_file, failed_file,
                           save_filtered)

    else:
        with open(input_fastq, 'r') as init_file, open(output_file_prefix + '_passed.fastq', 'w') as passed_file:
            open_and_check(gc_bounds, length_bounds, quality_threshold, init_file, passed_file)


def open_and_check(gc_bounds, length_bounds, quality_threshold, init_file, passed_file, failed_file=None,
                   save_filtered=False):
    while True:
        first_line = init_file.readline()
        if first_line == '':
            break
        seq_line = init_file.readline()
        third_line = init_file.readline()
        qual_line = init_file.readline()

        if (check_gc(seq_line.strip(), gc_bounds) and check_quality(qual_line.strip(), quality_threshold)
                and check_length(seq_line.strip(), length_bounds)):
            write_file(passed_file, first_line, seq_line, third_line, qual_line)
        else:
            if save_filtered:
                write_file(failed_file, first_line, seq_line, third_line, qual_line)


def write_file(file, first_l, second_l, third_l, fourth_l):
    file.write(first_l)
    file.write(second_l)
    file.write(third_l)
    file.write(fourth_l)


def check_gc(seq_line, gc_bounds):
    seq_len = len(seq_line)
    n_c = seq_line.count('C')
    n_g = seq_line.count('G')
    count_gc = (n_c+n_g) / seq_len * 100
    if isinstance(gc_bounds, int) or isinstance(gc_bounds, float):
        higher_bound = gc_bounds
        lower_bound = 0
    else:
        higher_bound = gc_bounds[1]
        lower_bound = gc_bounds[0]
    return lower_bound <= count_gc <= higher_bound


def check_quality(qual_line, quality_threshold):
    quality = sum(ord(i) - 33 for i in qual_line) / len(qual_line)
    return quality >= quality_threshold


def check_length(seq_line, length_bounds):
    if isinstance(length_bounds, int) or isinstance(length_bounds, float):
        higher_bound = length_bounds
        lower_bound = 0
    else:
        higher_bound = length_bounds[1]
        lower_bound = length_bounds[0]
    return lower_bound <= len(seq_line) <= higher_bound





