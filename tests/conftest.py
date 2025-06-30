from stamp.preprocessing import ExtractorName


# This lets you choose which extractors to run on pytest. Useful for the
# CI pipeline as extractors like MUSK take an eternity on CPU.
def pytest_addoption(parser):
    parser.addoption(
        "--extractor",
        action="append",
        dest="extractor",
        default=None,
        help="Specify extractor(s) to test",
        choices=[e.value for e in ExtractorName],
    )


def pytest_generate_tests(metafunc):
    if "extractor" in metafunc.fixturenames:
        chosen = metafunc.config.getoption("extractor")
        if not chosen:
            chosen = [e.value for e in ExtractorName]  # default to all
        metafunc.parametrize("extractor", chosen, indirect=False, ids=lambda x: x)
