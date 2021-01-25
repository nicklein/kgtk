## Overview

The cat command combines (concatenates) one or more KGTK files, optionally
decompressing input files and compressing the output file, while managing the
KGTK column headers appropriately.  The input file(s) are read in the order
specified and edges are copied to the output file without deduplication.

### Merging Column Headers

Each column in the input file(s) becomes a column in the output file.  Input
columns with the same name in different files are merged into a single column.
Column names are case sensitive.

Input columns with one of KGTK's required column names will
also be merged into a single column even if their names do not
match exactly, so long as their names are matching KGTK aliases.
The first name or alias seen takes priority.  For example, if the first
input file has a "node1" column and the second input file has a "from" column, the two
columns will be combined as the "node1" column in the output file.

| Canonical Name | Alias Names |
| -------------- | ----------- |
| id             | ID          |
| label          | predicate relation relationship |
| node1          | from subject |
| node2          | to object |

### KGTK File Modes

Normally, the files being combined must be either all KGTK edge files or all
KGTK node files.  `kgtk cat` will complain if an input file is not a KGTK
edge or node file, or if `kgtk cat` is given a mixture of KGTK edge and node files.
These constraints can be overridden with the expert option `--mode=NONE`.

### Input File Format

Although KGTK commands use the KGTK File Format as their primary file format,
input files  can be  read in  another supported file  format using  the expert
option  `--input-format  INPUT_FORMAT`, where  `INPUT_FORMAT`  is  one of  the
format names shown in the table below:

| Format | Extension | Description |
| ------ | --------- | ----------- |
| kgtk   | .kgtk or .tsv | KGTK tab separated values file format. |
| csv    | .csv      | A simple comma separated value file with doubled quoting and column headers. |

When the `--input-format` option has not been specified, the default is to use `kgtk` format
for input files.

!!! note
    The expert option `--input-format INPUT_FORMAT` applies to all input files in
    the `kgtk cat` command, so it not possible at present to use `kgtk cat` to
    combine a file in KGTK format with a file in CSV format.  It is necessary to
    convert all input files to a common input format before using `kgtk cat` to combine them
    (although their compression format may vary, as described below).

!!! note
    At the present time, filename extensions are not used to select an input
    file format in the absence of the `--input-format` option.  This behavior
    may change in the future.

### Input File Decompression

Input files may be decompressed using an algorithm selected
by the filename extension.  The following compression algorithms
are supported:

| Extension | Algorithm |
| --------- | ----------- |
| .bz2      | [bzip2][1]  |
| .gz       | [gzip][2]   |
| .lz4      | [LZ4][3]    |
| .xz       | [XZ Utils][4], based on LZMA |

[1]: https://en.wikipedia.org/wiki/Bzip2
[2]: https://en.wikipedia.org/wiki/Gzip
[3]: https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)
[4]: https://en.wikipedia.org/wiki/XZ_Utils

When used, compression filename extensions must appear after any other
filename extensions, e.g. `.kgtk.gz`, `.csv.gz`.

Decompression may also be selected using the `--compression-type COMPRESSION_TYPE` option.
This is an expert option which does not appear in the normal
usage message (shown below).  The `COMPRESSION_TYPE` value is one of the extension values shown in the
table above, with or without the leading period.
This option may be used to specify decompression of standard input (`-`).

If `--compression-type` is not specified and the the filename extension is not
a recognized compression fielname extension, the input file will not be
decompressed.

!!! Note
    When the `--compression type` expert option is specified, all input
    files will be decompressed using the specified compression type, ignoring their file extensions.

!!! note
    At the present time, decompression is not supported for file descriptor input
    files (filenames that begin with `<`, followed by a file descriptor number).

### Output File Format

Although KGTK commands use the KGTK File Format as their primary file format,
the output file can be written in a selection of formats other
than KGTK format by using the `--output-format FORMAT` option, where `FORMAT`
is one of the values in the table shown below.

| Format | Extension | Description |
| ------ | --------- | ----------- |
| kgtk   | .kgtk or .tsv | KGTK tab separated values file format. |
| csv    | .csv      | A simple comma separated value file with doubled quoting and column headers. |
| md	 | .md       | GitHub markdown tables. |
| json   | .json     | JSON list of lists of strings with column header line. |
| json-map | (none)  | JSON list of maps from column names to string values. |
| json-map-compact | (none)  | JSON list of maps from column names to string values with empty values suppressed. |
| jsonl  | .jsonl    | JSON lines of lists of strings  with column header line. |
| jsonl-map | (none)  | JSON lines of maps from column names to string values. |
| jsonl-map-compact | (none)  | JSON lines of maps from column names to string values with empty values suppressed. |
| tsv | (none) | Tab separated values.  Dates have their sigils removed, and strings have the backslash escape removed before pipes. |
| tsv-csvlike | (none) | Tab separated values.  Dates have their sigils removed, and strings are transformed into CSV-like double quoted strings, losing the language code if present. |
| tsv-unquoted | (none) | Tab separated values.  Dates have their sigils removed, and strings have their content exposed without quotes and without escapes before pipes. |
| tsv-unquoted-ep | (none) | Tab separated values.  Dates have their sigils removed, and strings have their content exposed without quotes ; pipes retain their preceeding escapes. |

Output formats may also be selected by the filename extension on the output file if `--output-format`
has not been specified.  For example,
writing an output file with the extension `.csv` will automatically generate an output file
in CSV format.  Any unrecognized extensions default to kgtk format unless
overridden by the `--output-format` option.

!!! note
    The csv and json* formats use very primitive conversions at the present time,
    which do not provide proper treatment for different data types: booleans,
    numbers, strings.

### Output File Compression

Output files may be compressed using an algorithm selected
by the file extension.  The following compression algorithms
are supported:

| Extension | Algorithm |
| --------- | ----------- |
| .bz2      | [bzip2][1]  |
| .gz       | [gzip][2]   |
| .lz4      | [LZ4][3]    |
| .xz       | [XZ Utils][4], based on LZMA |

When specified, compression format extensions must appear after output format
selection extensions, e.g. `.kgtk.gz`, `.csv.gz`, `.json.bz2`.

!!! note
    At the present time, the `--compression-type COMPRESSION_TYPE` option does not
    affect output files.  Standard output (`-`) and file descriptor output
    files (filesnames that begin with `>`, followed by a file descriptor number)
    will not be compressed.  This behavior may change at a later date.

## Usage

```bash
usage: kgtk cat [-h] [-i INPUT_FILE [INPUT_FILE ...]] [-o OUTPUT_FILE]
                [--output-format {csv,json,json-map,json-map-compact,jsonl,jsonl-map,jsonl-map-compact,kgtk,md,tsv,tsv-csvlike,tsv-unquoted,tsv-unquoted-ep}]
                [-v [optional True|False]]

Concatenate two or more KGTK files, merging the columns appropriately. All files must be KGTK edge files or all files must be KGTK node files (unless overridden with --mode=NONE). 

Additional options are shown in expert help.
kgtk --expert cat --help

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE [INPUT_FILE ...], --input-files INPUT_FILE [INPUT_FILE ...]
                        KGTK input files (May be omitted or '-' for stdin.)
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        The KGTK output file. (May be omitted or '-' for
                        stdout.)
  --output-format {csv,json,json-map,json-map-compact,jsonl,jsonl-map,jsonl-map-compact,kgtk,md,tsv,tsv-csvlike,tsv-unquoted,tsv-unquoted-ep}
                        The file format (default=kgtk)

  -v [optional True|False], --verbose [optional True|False]
                        Print additional progress messages (default=False).
```

## Examples

### Sample Data

Suppose that `movies_reduced.tsv` contains the following table in KGTK format:

```bash
kgtk cat -i examples/docs/movies_reduced.tsv
```
| id | node1 | label | node2 |
| -- | -- | -- | -- |
| t1 | terminator | label | 'The Terminator'@en |
| t2 | terminator | instance_of | film |
| t3 | terminator | genre | action |
| t4 | terminator | genre | science_fiction |
| t5 | terminator | publication_date | ^1984-10-26T00:00:00Z/11 |
| t6 | t5 | location | united_states |
| t7 | terminator | publication_date | ^1985-02-08T00:00:00Z/11 |
| t8 | t7 | location | sweden |
| t9 | terminator | director | james_cameron |
| t10 | terminator | cast | arnold_schwarzenegger |
| t11 | t10 | role | terminator |
| t12 | terminator | cast | michael_biehn |
| t13 | t12 | role | kyle_reese |
| t14 | terminator | cast | linda_hamilton |
| t15 | t14 | role | sarah_connor |
| t16 | terminator | duration | 108 |
| t17 | terminator | award | national_film_registry |
| t18 | t17 | point_in_time | ^2008-01-01T00:00:00Z/9 |

Suppose that `tutorial_people_full.tsv` contains the following table in KGTK format:

```bash
kgtk cat -i examples/docs/tutorial_people_full.tsv
```
| id | node1 | label | node2 |
| -- | -- | -- | -- |
| h1 | james_cameron | label | "James Cameron" |
| h2 | james_cameron | instance_of | human |
| h3 | james_cameron | birth_date | ^1954-08-16T00:00:00Z/11 |
| h4 | james_cameron | country | Canada |
| h5 | arnold_schwarzenegger | label | "Arnold Schwarzenegger" |
| h6 | arnold_schwarzenegger | instance_of | human |
| h7 | arnold_schwarzenegger | birth_date | ^1947-07-30T00:00:00Z/11 |
| h8 | arnold_schwarzenegger | country | "Austria" |
| h9 | michael_biehn | label | "Michael Biehn" |
| h10 | michael_biehn | instance_of | human |
| h11 | michael_biehn | birth_date | ^1956-07-31T00:00:00Z/11 |
| h12 | michael_biehn | country | "United States of America" |
| h13 | linda_hamilton | label | "Linda Hamilton" |
| h14 | linda_hamilton | instance_of | human |
| h15 | linda_hamilton | birth_date | ^1956-09-26T00:00:00Z/11 |
| h16 | linda_hamilton | country | "United States of America" |
| h17 | edward_furlong | label | "Edward Furlong" |
| h18 | edward_furlong | instance_of | human |
| h19 | edward_furlong | birth_date | ^1977-08-02T00:00:00Z/11 |
| h20 | edward_furlong | country | "United States of America" |
| h21 | robert_patrick | label | "Robert Patrick" |
| h22 | robert_patrick | instance_of | human |
| h23 | robert_patrick | birth_date | ^1958-11-05T00:00:00Z/11 |
| h24 | robert_patrick | country | "United States of America" |

### Combine two KGTK files, sending the output to standard output.

These two files have only he 4 basic KGTK fields.

```bash
kgtk cat -i examples/docs/movies_reduced.tsv examples/docs/tutorial_people_full.tsv
```
The result will be the following file in KGTK format:

| id | node1 | label | node2 |
| -- | -- | -- | -- |
| t1 | terminator | label | 'The Terminator'@en |
| t2 | terminator | instance_of | film |
| t3 | terminator | genre | action |
| t4 | terminator | genre | science_fiction |
| t5 | terminator | publication_date | ^1984-10-26T00:00:00Z/11 |
| t6 | t5 | location | united_states |
| t7 | terminator | publication_date | ^1985-02-08T00:00:00Z/11 |
| t8 | t7 | location | sweden |
| t9 | terminator | director | james_cameron |
| t10 | terminator | cast | arnold_schwarzenegger |
| t11 | t10 | role | terminator |
| t12 | terminator | cast | michael_biehn |
| t13 | t12 | role | kyle_reese |
| t14 | terminator | cast | linda_hamilton |
| t15 | t14 | role | sarah_connor |
| t16 | terminator | duration | 108 |
| t17 | terminator | award | national_film_registry |
| t18 | t17 | point_in_time | ^2008-01-01T00:00:00Z/9 |
| h1 | james_cameron | label | "James Cameron" |
| h2 | james_cameron | instance_of | human |
| h3 | james_cameron | birth_date | ^1954-08-16T00:00:00Z/11 |
| h4 | james_cameron | country | Canada |
| h5 | arnold_schwarzenegger | label | "Arnold Schwarzenegger" |
| h6 | arnold_schwarzenegger | instance_of | human |
| h7 | arnold_schwarzenegger | birth_date | ^1947-07-30T00:00:00Z/11 |
| h8 | arnold_schwarzenegger | country | "Austria" |
| h9 | michael_biehn | label | "Michael Biehn" |
| h10 | michael_biehn | instance_of | human |
| h11 | michael_biehn | birth_date | ^1956-07-31T00:00:00Z/11 |
| h12 | michael_biehn | country | "United States of America" |
| h13 | linda_hamilton | label | "Linda Hamilton" |
| h14 | linda_hamilton | instance_of | human |
| h15 | linda_hamilton | birth_date | ^1956-09-26T00:00:00Z/11 |
| h16 | linda_hamilton | country | "United States of America" |
| h17 | edward_furlong | label | "Edward Furlong" |
| h18 | edward_furlong | instance_of | human |
| h19 | edward_furlong | birth_date | ^1977-08-02T00:00:00Z/11 |
| h20 | edward_furlong | country | "United States of America" |
| h21 | robert_patrick | label | "Robert Patrick" |
| h22 | robert_patrick | instance_of | human |
| h23 | robert_patrick | birth_date | ^1958-11-05T00:00:00Z/11 |
| h24 | robert_patrick | country | "United States of America" |


### Combine two gzipped KGTK files, sending the output to a bzip2 file.

```bash
kgtk cat -i examples/docs/movies_reduced.tsv.gz examples/docs/tutorial_people_full.tsv.gz -o ofile.tsv.bz2
```

### Expert Topic: Processing Files Not in KGTK Format

Suppose that `not-kgtk.tsv` contains the following data **not** in KGTK format:

```bash
kgtk cat -i examples/docs/not-kgtk.tsv --mode=NONE
```

| a | b | c | d |
| -- | -- | -- | -- |
| h21 | robert_patrick | label | "Robert Patrick" |
| h22 | robert_patrick | instance_of | human |
| h23 | robert_patrick | birth_date | ^1958-11-05T00:00:00Z/11 |
| h24 | robert_patrick | country | "United States of America" |

Trying to run the command:

    ```
    kgtk cat -i file1.tsv.gz file3.tsv 
    ```

will result in an error message:

    ```
    In input 2 header 'a	b	c	d': Missing required column: id | ID
    Exit requested
    ```

We can force the `kgtk cat` command to process the file by using the `--mode NONE` option,
as shown above.

### Expert Topic: Adding Column Names

Suppose that you have a TSV (tab-separated values) data file
that looks like a KGTK data file but without the header line.
You can supply a header line with the expert option `--force-column-names`.
You can also use this option when concatenating several
data files, so long as they are all missing header lines and
they should all have the same header line.

Consider the following input file:

```bash
kgtk cat -i examples/docs/no-header.tsv --mode=NONE
````
| a | b | c | d |
| -- | -- | -- | -- |
| h21 | robert_patrick | label | "Robert Patrick" |
| h22 | robert_patrick | instance_of | human |
| h23 | robert_patrick | birth_date | ^1958-11-05T00:00:00Z/11 |
| h24 | robert_patrick | country | "United States of America" |

We can supply a valid header line as follows:

```bash
kgtk cat -i examples/docs/no-header.tsv \
         --force-column-names id node1 label node2
```

The result will be the following file in KGTK format:

| id | node1 | label | node2 |
| -- | -- | -- | -- |
| a | b | c | d |
| h21 | robert_patrick | label | "Robert Patrick" |
| h22 | robert_patrick | instance_of | human |
| h23 | robert_patrick | birth_date | ^1958-11-05T00:00:00Z/11 |
| h24 | robert_patrick | country | "United States of America" |

!!! note
    `---force-column-names` takes place before the input file is checked
    to see if it is a valid KGTK edge or node file.  Since we supplied
    valid KGTK edge column names in the example above, `--mode=NONE` is
    no longer needed.

### Expert Topic: Renaming Column Names on Input

There is a special KGTK command, `kgtk rename_columns`, for renaming columns.
However, you may want to rename columns while also using other features of
the `kgtk cat` command, such as combining multiple input files or sampling
data lines.

You have two main choices: override the column names on input, or rename the
column names on output.

Overriding the column names on input can be done by skipping the existing
header record and supplying a replacement list of column names.

```bash
kgtk cat -i examples/docs/not-kgtk.tsv \
         --skip-header-record \
	 --force-column-names id node1 label node2
```

The result will be the following file in KGTK format:

| id | node1 | label | node2 |
| -- | -- | -- | -- |
| a | b | c | d |
| h21 | robert_patrick | label | "Robert Patrick" |
| h22 | robert_patrick | instance_of | human |
| h23 | robert_patrick | birth_date | ^1958-11-05T00:00:00Z/11 |
| h24 | robert_patrick | country | "United States of America" |

!!! note
    When you rename columns on input, the change applies to all input files: they
    all must have the same column layout, for which you will provide a new set of
    column names.

### Expert Topic: Renaming Column Names on Output

There is a special KGTK command, `kgtk rename_columns`, for renaming columns.
However, you may want to rename columns while also using other features of
the `kgtk cat` command, such as combining multiple input files or sampling
data lines.

You have two main choices: override the column names on input, or rename the
column names on output.

For example, suppose your input file contained the following table in almost KGTK format:

```bash
kgtk cat -i examples/docs/movies_origin_destination.tsv --mode=NONE
```

| origin | label | destination | years |
| -- | -- | -- | -- |
| terminator | label | 'The Terminator'@en | 4 |
| terminator | instance_of | film | 3 |

Renaming the column names on output can by done two ways.  First, you can name
all of the new column names using --output-columns.

```bash
kgtk cat -i examples/docs/movies_origin_destination.tsv --mode=NONE \
         --output-columns node1 label node2 years
```

The result will be the following table in KGTK format:

| node1 | label | node2 | years |
| -- | -- | -- | -- |
| terminator | label | 'The Terminator'@en | 4 |
| terminator | instance_of | film | 3 |

Second, you can rename individual columns using --old-columns and --new-columns.

You want to rename the `origin` column to `node1`, and the `destination`
column to `node2`, leaving the other column names alone.

```bash
kgtk cat -i examples/docs/movies_origin_destination.tsv --mode=NONE \
         --old-columns origin destination \
	 --new-columns node1 node2
```

The result will be the following table in KGTK format:

| node1 | label | node2 | years |
| -- | -- | -- | -- |
| terminator | label | 'The Terminator'@en | 4 |
| terminator | instance_of | film | 3 |

!!! note
    Renaming column names on output can be done when you combine a
    disparate set of KGTK files.  The rename applies to the merged set of column
    names computed by `kgtk cat`.

### Expert Topic: Data Sampling: head

Limit the number of records read (like `head`).

```bash
kgtk cat -i examples/docs/movies_reduced.tsv --record-limit 4
```

The result will be the following table in KGTK format:

| id | node1 | label | node2 |
| -- | -- | -- | -- |
| t1 | terminator | label | 'The Terminator'@en |
| t2 | terminator | instance_of | film |
| t3 | terminator | genre | action |
| t4 | terminator | genre | science_fiction |

### Expert Topic: Data Sampling: skip

Skip some number of initial records, then begin processing.

```bash
kgtk cat -i examples/docs/movies_reduced.tsv --initial-skip-count 4
```

The result will be the following table in KGTK format:

| id | node1 | label | node2 |
| -- | -- | -- | -- |
| t5 | terminator | publication_date | ^1984-10-26T00:00:00Z/11 |
| t6 | t5 | location | united_states |
| t7 | terminator | publication_date | ^1985-02-08T00:00:00Z/11 |
| t8 | t7 | location | sweden |
| t9 | terminator | director | james_cameron |
| t10 | terminator | cast | arnold_schwarzenegger |
| t11 | t10 | role | terminator |
| t12 | terminator | cast | michael_biehn |
| t13 | t12 | role | kyle_reese |
| t14 | terminator | cast | linda_hamilton |
| t15 | t14 | role | sarah_connor |
| t16 | terminator | duration | 108 |
| t17 | terminator | award | national_film_registry |
| t18 | t17 | point_in_time | ^2008-01-01T00:00:00Z/9 |

### Expert Topic: Data Sampling: last 5

Process the last n records relative to the end (like `tail`).
You must know the number of data records in the file (the number of lines
in the file minus the header line).

```bash
kgtk cat -i examples/docs/movies_reduced.tsv --record-limit 15 --tail-count 3
```

The result will be the following table in KGTK format:

| id | node1 | label | node2 |
| -- | -- | -- | -- |
| t13 | t12 | role | kyle_reese |
| t14 | terminator | cast | linda_hamilton |
| t15 | t14 | role | sarah_connor |

!!! note
    If both --initial-skip-count # and --record-limit # --tail-count #
    are specified, the number of records skipped will be the maximum of
    the initial skip count and (record limit minus tail count).

### Expert Topic: Data Sampling: every n

Process every nth record (after skipping, but calculated relative to
the count of data lines read before skipping).  The following example will
process every second line.

```bash
kgtk cat -i examples/docs/movies_reduced.tsv --every-nth-record 2
```

The result will be the following table in KGTK format:

| id | node1 | label | node2 |
| -- | -- | -- | -- |
| t2 | terminator | instance_of | film |
| t4 | terminator | genre | science_fiction |
| t6 | t5 | location | united_states |
| t8 | t7 | location | sweden |
| t10 | terminator | cast | arnold_schwarzenegger |
| t12 | terminator | cast | michael_biehn |
| t14 | terminator | cast | linda_hamilton |
| t16 | terminator | duration | 108 |
| t18 | t17 | point_in_time | ^2008-01-01T00:00:00Z/9 |
