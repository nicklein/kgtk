"""
Remove columns from a KGTK file.
"""
from argparse import Namespace, SUPPRESS
from pathlib import Path
import sys
import typing

from kgtk.cli_argparse import KGTKArgumentParser
from kgtk.io.kgtkreader import KgtkReader, KgtkReaderOptions, KgtkReaderMode
from kgtk.io.kgtkwriter import KgtkWriter
from kgtk.utils.argparsehelpers import optional_bool
from kgtk.value.kgtkvalueoptions import KgtkValueOptions

def parser():
    return {
        'help': 'Remove columns from a KGTK file.',
        'description': 'Remove specific columns from a KGTK file.'
    }


def add_arguments_extended(parser: KGTKArgumentParser, parsed_shared_args: Namespace):
    """
    Parse arguments
    Args:
        parser (argparse.ArgumentParser)
    """

    _expert: bool = parsed_shared_args._expert

    # '$label == "/r/DefinedAs" && $node2=="/c/en/number_zero"'
    parser.add_argument(      "input_kgtk_file", nargs="?", help="The KGTK file to filter. May be omitted or '-' for stdin.", type=Path, default="-")
    parser.add_argument("-o", "--output-file", dest="output_kgtk_file", help="The KGTK file to write records that pass the filter (default=%(default)s).",
                        type=Path, default="-")

    parser.add_argument('-c', "--columns", action="store", type=str, dest="columns", nargs='+', required=True,
                        help="Columns to remove as a comma- or space-separated strings, e.g., id,docid or id docid")

    parser.add_argument(      "--split-on-commas", dest="split_on_commas", help="Parse the list of columns, splitting on commas. (default=%(default)s).",
                              type=optional_bool, nargs='?', const=True, default=True)

    parser.add_argument(      "--split-on-spaces", dest="split_on_spaces", help="Parse the list of columns, splitting on spaces. (default=%(default)s).",
                              type=optional_bool, nargs='?', const=True, default=False)

    parser.add_argument(      "--strip-spaces", dest="strip_spaces", help="Parse the list of columns, stripping whitespace. (default=%(default)s).",
                              type=optional_bool, nargs='?', const=True, default=True)

    KgtkReader.add_debug_arguments(parser, expert=_expert)
    KgtkReaderOptions.add_arguments(parser, mode_options=True, default_mode=KgtkReaderMode.NONE, expert=_expert)
    KgtkValueOptions.add_arguments(parser, expert=_expert)

def run(input_kgtk_file: Path,
        output_kgtk_file: Path,

        columns: typing.Optional[typing.List[str]],

        split_on_commas: bool,
        split_on_spaces: bool,
        strip_spaces: bool,

        errors_to_stdout: bool = False,
        errors_to_stderr: bool = True,
        show_options: bool = False,
        verbose: bool = False,
        very_verbose: bool = False,

        **kwargs # Whatever KgtkFileOptions and KgtkValueOptions want.
)->int:
    # import modules locally
    from kgtk.exceptions import kgtk_exception_auto_handler, KGTKException

    # Select where to send error messages, defaulting to stderr.
    error_file: typing.TextIO = sys.stdout if errors_to_stdout else sys.stderr

    # Build the option structures.
    reader_options: KgtkReaderOptions = KgtkReaderOptions.from_dict(kwargs)
    value_options: KgtkValueOptions = KgtkValueOptions.from_dict(kwargs)

    # Show the final option structures for debugging and documentation.
    if show_options:
        print("input: %s" % str(input_kgtk_file), file=error_file)
        print("--output-file=%s" % str(output_kgtk_file), file=error_file)
        if columns is not None:
            print("--columns=%s" % " ".join(columns), file=error_file)
        print("--split-on-commas=%s" % str(split_on_commas), file=error_file)
        print("--split-on-spaces=%s" % str(split_on_spaces), file=error_file)
        print("--sprip-spaces=%s" % str(strip_spaces), file=error_file)
        reader_options.show(out=error_file)
        value_options.show(out=error_file)
        print("=======", file=error_file, flush=True)

    try:

        if columns is None:
            columns = [ ] # This simplifies matters.

        if split_on_spaces:
            # We will be very lenient, and allow space-seperated arguments
            # *inside* shell quoting, e.g.
            #
            # kgtk remove_columns -c 'name name2 name3'
            #
            # Do not enable this option if spaces are legal inside your
            # column names.
            columns = " ".join(columns).split()
        remove_columns: typing.List[str] = [ ]
        arg: str
        column_name: str
        for arg in columns:
            if split_on_commas:
                for column_name in arg.split(","):
                    if strip_spaces:
                        column_name = column_name.strip()
                    if len(column_name) > 0:
                        remove_columns.append(column_name)
            else:
                if strip_spaces:
                    arg = arg.strip()
                if len(arg) > 0:
                    remove_columns.append(arg)
        if verbose:
            print("Removing %d columns: %s" % (len(remove_columns), " ".join(remove_columns)), file=error_file, flush=True)
        if len(remove_columns) == 0:
            raise KGTKException("No columns to remove")

        if verbose:
            print("Opening the input file: %s" % str(input_kgtk_file), file=error_file, flush=True)
        kr: KgtkReader = KgtkReader.open(input_kgtk_file,
                                         options=reader_options,
                                         value_options = value_options,
                                         error_file=error_file,
                                         verbose=verbose,
                                         very_verbose=very_verbose,
        )

        output_column_names: typing.List[str] = kr.column_names.copy()

        trouble_column_names: typing.List[str] = [ ]
        for column_name in remove_columns:
            if column_name in output_column_names:
                output_column_names.remove(column_name)
            else:
                print("Error: cannot remove unknown column '%s'." % column_name, file=error_file, flush=True)
                trouble_column_names.append(column_name)
        if len(trouble_column_names) > 0:
            raise KGTKException("Unknown columns %s" % " ".join(trouble_column_names))

        if verbose:
            print("Opening the output file: %s" % str(output_kgtk_file), file=error_file, flush=True)
        kw: KgtkWriter = KgtkWriter.open(output_column_names,
                                         output_kgtk_file,
                                         mode=kr.mode,
                                         verbose=verbose,
                                         very_verbose=very_verbose)

        shuffle_list: typing.List[int] = kw.build_shuffle_list(kr.column_names)

        input_line_count: int = 0
        row: typing.List[str]
        for row in kr:
            input_line_count += 1
            kw.write(row, shuffle_list=shuffle_list)

        if verbose:
            print("Processed %d rows." % (input_line_count))

        kw.close()

        return 0

    except Exception as e:
        kgtk_exception_auto_handler(e)
        return 1
