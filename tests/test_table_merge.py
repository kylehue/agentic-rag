from app.plugins.text import TableFragment, merge_table_fragments


def frag(header, data, page=None):
    return TableFragment(header=header, data=data, page=page)


def names(table):
    return [row[0] for row in table.data]


# --- repeated header on the continuation page ---


def test_repeated_header_is_merged_not_duplicated():
    first = frag(["name", "age"], [["kyle", "25"], ["anne", "21"]], page=1)
    second = frag(["name", "age"], [["john", "23"]], page=2)

    tables = merge_table_fragments([first, second])

    assert len(tables) == 1
    table = tables[0]
    assert table.header == ["name", "age"]
    assert table.data == [
        ["kyle", "25"],
        ["anne", "21"],
        ["john", "23"],
    ]
    # The header must not survive as a data row.
    assert ["name", "age"] not in table.data
    assert table.fragment_count == 2
    assert table.page == 1


# --- different schema (different header) stays separate ---


def test_different_headers_stay_separate():
    first = frag(["name", "age"], [["kyle", "25"], ["anne", "21"]], page=1)
    second = frag(["part", "length"], [["petal", "5cm"], ["sepal", "2cm"]], page=2)

    tables = merge_table_fragments([first, second])

    assert len(tables) == 2
    assert names(tables[0]) == ["kyle", "anne"]
    assert tables[1].header == ["part", "length"]
    assert names(tables[1]) == ["petal", "sepal"]


def test_different_column_count_stays_separate():
    first = frag(["name", "age"], [["kyle", "25"]])
    second = frag(["name", "age", "city"], [["anne", "21", "oslo"]])

    assert len(merge_table_fragments([first, second])) == 2


def test_header_comparison_ignores_case_and_whitespace():
    first = frag(["Name", "Age"], [["kyle", "25"]])
    second = frag(["name", " age "], [["john", "23"]])

    tables = merge_table_fragments([first, second])

    assert len(tables) == 1
    # The first fragment's header spelling wins.
    assert tables[0].header == ["Name", "Age"]
    assert names(tables[0]) == ["kyle", "john"]


# --- missing header on the continuation page ---


def test_missing_header_same_types_is_merged():
    first = frag(["name", "age"], [["kyle", "25"], ["anne", "21"]], page=1)
    second = frag(None, [["john", "44"], ["anika", "16"]], page=2)

    tables = merge_table_fragments([first, second])

    assert len(tables) == 1
    table = tables[0]
    assert table.header == ["name", "age"]
    assert table.data == [
        ["kyle", "25"],
        ["anne", "21"],
        ["john", "44"],
        ["anika", "16"],
    ]
    assert table.fragment_count == 2


def test_misclassified_continuation_header_is_merged_as_data():
    # The parser marked the continuation's first data row as a header (a
    # table continuing across a page break without repeating its header).
    # The "header" row is type-compatible with the table's data, so the
    # fragment merges with that row kept as data.
    first = frag(
        ["Month", "Tomatoes", "Total"],
        [["April", "12 kg", "23 kg"]],
        page=1,
    )
    second = frag(
        ["May", "18 kg", "34 kg"],
        [["June", "24 kg", "46 kg"]],
        page=2,
    )

    tables = merge_table_fragments([first, second])

    assert len(tables) == 1
    table = tables[0]
    assert table.header == ["Month", "Tomatoes", "Total"]
    assert table.data == [
        ["April", "12 kg", "23 kg"],
        ["May", "18 kg", "34 kg"],
        ["June", "24 kg", "46 kg"],
    ]
    assert table.fragment_count == 2


def test_genuine_new_header_with_conflicting_label_stays_separate():
    # A real new table with its own header: the "code" label (text)
    # conflicts with the numeric age column, so it does not merge even
    # though the data values below it are numeric.
    first = frag(["name", "age"], [["kyle", "25"], ["anne", "21"]], page=1)
    second = frag(
        ["city", "code"],
        [["Oslo", "0179"], ["Bergen", "0555"]],
        page=2,
    )

    tables = merge_table_fragments([first, second])

    assert len(tables) == 2
    assert tables[1].header == ["city", "code"]
    assert tables[1].data == [["Oslo", "0179"], ["Bergen", "0555"]]


def test_missing_header_conflicting_types_stays_separate():
    first = frag(["name", "age"], [["kyle", "25"], ["anne", "21"]], page=1)
    # "4cm" / "1cm" are not numbers, conflicting with the numeric age column.
    second = frag(None, [["petal", "4cm"], ["sepal", "1cm"]], page=2)

    tables = merge_table_fragments([first, second])

    assert len(tables) == 2
    assert names(tables[0]) == ["kyle", "anne"]
    # The second table keeps its (missing) header state: no header, its rows
    # intact.
    assert tables[1].header is None
    assert tables[1].data == [["petal", "4cm"], ["sepal", "1cm"]]


def test_two_headerless_fragments_same_width_are_merged():
    first = frag(None, [["kyle", "25"], ["anne", "21"]])
    second = frag(None, [["john", "44"]])

    tables = merge_table_fragments([first, second])

    assert len(tables) == 1
    assert tables[0].header is None
    assert tables[0].data == [["kyle", "25"], ["anne", "21"], ["john", "44"]]


def test_headerless_then_headered_starts_new_table():
    first = frag(None, [["kyle", "25"]])
    second = frag(["name", "age"], [["anne", "21"]])

    tables = merge_table_fragments([first, second])

    assert len(tables) == 2


# --- chains and misc ---


def test_three_fragment_chain_merges():
    a = frag(["name", "age"], [["kyle", "25"]], page=1)
    b = frag(["name", "age"], [["anne", "21"]], page=2)
    c = frag(None, [["john", "44"]], page=3)

    tables = merge_table_fragments([a, b, c])

    assert len(tables) == 1
    assert names(tables[0]) == ["kyle", "anne", "john"]
    assert tables[0].fragment_count == 3


def test_fragment_count_tracks_split_tables():
    a = frag(["name", "age"], [["kyle", "25"]])
    b = frag(["name", "age"], [["anne", "21"]])
    c = frag(["part", "length"], [["petal", "5cm"]])
    d = frag(["part", "length"], [["sepal", "2cm"]])

    tables = merge_table_fragments([a, b, c, d])

    assert len(tables) == 2
    assert tables[0].fragment_count == 2
    assert tables[1].fragment_count == 2


def test_number_detection_allows_separators_currency_percent():
    first = frag(["amount"], [["1,200"], ["$50"], ["75%"]])
    second = frag(None, [["2,000"], ["$9"], ["50%"]])

    tables = merge_table_fragments([first, second])

    assert len(tables) == 1
    assert len(tables[0].data) == 6


def test_empty_cells_do_not_conflict():
    first = frag(["name", "age"], [["kyle", "25"], ["anne", ""]])
    second = frag(None, [["john", "44"], ["", ""]])

    tables = merge_table_fragments([first, second])

    assert len(tables) == 1


def test_header_only_fragment_merges_with_same_header():
    first = frag(["name", "age"], [])
    second = frag(["name", "age"], [["kyle", "25"]])

    tables = merge_table_fragments([first, second])

    assert len(tables) == 1
    assert tables[0].data == [["kyle", "25"]]
    assert tables[0].fragment_count == 2


def test_single_fragment_passthrough():
    only = frag(["name", "age"], [["kyle", "25"]], page=7)

    tables = merge_table_fragments([only])

    assert len(tables) == 1
    assert tables[0].header == ["name", "age"]
    assert tables[0].data == [["kyle", "25"]]
    assert tables[0].page == 7
    assert tables[0].fragment_count == 1


def test_empty_input():
    assert merge_table_fragments([]) == []
