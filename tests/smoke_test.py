def test_package_imports():
    """
    Basic check that the installed wheel can be imported.
    """
    import adhteb

    assert adhteb is not None