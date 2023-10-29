# FASTQ filtrator

This script works with fastq files.

It can:
1. Filter reads by GC content;
2. Filter reads by their quality;
3. Filter reads by their length;
4. Save filtered and failed reads to different files.

To run the script type:

```
main(input_fastq='YOUR FILE PATH', output_file_prefix='PREFIX FOR FILES', gc_bounds=(number/tuple of two numbers), length_bounds=(number/tuple of two numbers), quality_threshold=number, save_filtered=False/True)
```

If you pass one number to 'gc bounds' or 'length bounds' it will be the upper threshold and the lower threshold will be zero. File with reads that failed filtration will be created only when save_filtered == True.

Good luck!

