from importlib.metadata import distribution


def test_installs_only_infostop_and_cpputils():
    # `top_level.txt` lists the top-level names the installed distribution provides
    top_level = distribution("infostop").read_text("top_level.txt").split()
    assert sorted(top_level) == ["cpputils", "infostop"]
