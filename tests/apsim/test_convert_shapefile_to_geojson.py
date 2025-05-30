import pytest

from vercye_ops.apsim.convert_shapefile_to_geojson import clean_region_name

# Adding some random tests to check the testing CI


def test_symbols_and_punctuation():
    assert clean_region_name("Area 51 / Nevada") == "area_51___nevada"
    assert clean_region_name("South@East*Asia!") == "south_east_asia_"
    assert clean_region_name("100% Legit") == "100__legit"


def test_removes_quotes_and_symbols():
    assert clean_region_name("L'Haÿ-les-Roses") == "lhay-les-roses"
    assert clean_region_name('"Ulan Bator"') == "ulan_bator"
    assert clean_region_name("Davao—City") == "davao_city"


def test_handles_accents_and_diacritics():
    assert clean_region_name("São Tomé") == "sao_tome"
    assert clean_region_name("Réunion") == "reunion"
    assert clean_region_name("České Budějovice") == "ceske_budejovice"
    assert clean_region_name("İstanbul") == "istanbul"
    assert clean_region_name("Zürich") == "zurich"


def test_edge_cases():
    assert clean_region_name("") == ""
    assert clean_region_name("___") == "___"
    assert clean_region_name("123-456") == "123-456"


def test_special_chars_at_start_and_end():
    assert clean_region_name("!Nairobi") == "_nairobi"
    assert clean_region_name("Rio de Janeiro*") == "rio_de_janeiro_"
    assert clean_region_name("¿Bogotá?") == "_bogota_"
    assert clean_region_name("“Kigali”") == "_kigali_"
    assert clean_region_name("`Seoul`") == "_seoul_"
    assert clean_region_name("/Osaka/") == "_osaka_"
    assert clean_region_name("  Tokyo  ") == "__tokyo__"
    assert clean_region_name("...Berlin...") == "___berlin___"
